import os
import torch
import numpy as np
import librosa
import soundfile as sf
import pickle
import gradio as gr
from scipy import signal
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from diffusers import StableAudioPipeline

# =========================
# 1. 全局配置
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

MUSICGEN_PATH = "/root/MusicSfx-Fusion/models/musicgen-medium"
STABLE_AUDIO_PATH = "/root/MusicSfx-Fusion/models/stable-audio-open-1.0"

MIX_MODEL_PATH = "/root/MusicSfx-Fusion/best_model.pth"
SCALER_PATH = "/root/MusicSfx-Fusion/models/scaler.pkl"

sr = 22050

# =========================
# 2. 加载模型（只加载一次）
# =========================
print("Loading models...")

# MusicGen
processor = AutoProcessor.from_pretrained(MUSICGEN_PATH, local_files_only=True)
musicgen = MusicgenForConditionalGeneration.from_pretrained(
    MUSICGEN_PATH, local_files_only=True
).to(device)
musicgen.eval()

# Stable Audio
pipe = StableAudioPipeline.from_pretrained(
    STABLE_AUDIO_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)

pipe.enable_model_cpu_offload()

# Mixing Model
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return torch.tanh(self.net(x))

mix_model = MLP()
mix_model.load_state_dict(torch.load(MIX_MODEL_PATH, map_location=device))
mix_model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Models loaded!")

# =========================
# 3. 工具函数
# =========================
def compute_rms(y):
    return np.mean(librosa.feature.rms(y=y)[0])

def spectral_features(y, sr):
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))

    low = S[freqs < 200].sum()
    mid = S[(freqs >= 200) & (freqs < 2000)].sum()
    total = S.sum() + 1e-6

    return centroid, bandwidth, low/total, mid/total

def match_length(music, sfx):
    if len(sfx) < len(music):
        sfx = np.tile(sfx, len(music)//len(sfx)+1)
    return sfx[:len(music)]

def apply_gain(y, db):
    return y * (10 ** (db / 20))

def high_pass(y, sr, cutoff):
    b, a = signal.butter(2, cutoff/(sr/2), btype='high')
    return signal.filtfilt(b, a, y)

# ===== score =====
def audibility_score(music, sfx):
    ratio = compute_rms(sfx)/(compute_rms(music)+1e-6)
    return float(1 - np.exp(-5 * ratio))

def balance_score(music, sfx):
    ratio = compute_rms(sfx)/(compute_rms(music)+1e-6)
    return float(np.exp(-5 * max(0, ratio-0.4)))

def spectral_separation_score(music, sfx):
    S1 = np.abs(librosa.stft(music))
    S2 = np.abs(librosa.stft(sfx))
    S1 /= (S1.sum()+1e-6)
    S2 /= (S2.sum()+1e-6)
    return float(-np.sum(np.minimum(S1, S2)))

def mixing_score(music, sfx):
    a = audibility_score(music, sfx)
    b = balance_score(music, sfx)
    c = spectral_separation_score(music, sfx)
    return round(1.0*a + 0.3*b + 0.5*c,4), {"audibility":a,"balance":b,"sep":c}

# =========================
# 4. 生成函数
# =========================
def generate_music(prompt):
    inputs = processor(text=[prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        audio = musicgen.generate(**inputs, max_new_tokens=512)[0].cpu().numpy()
    return (32000, audio.T)

def generate_sfx(prompt):
    audio = pipe(prompt, num_inference_steps=200, audio_end_in_s=10.0).audios[0]
    audio = audio.to(torch.float32).cpu().numpy()
    return (pipe.vae.sampling_rate, audio.T)

# =========================
# 5. 自动混音
# =========================

def load_audio(audio_tuple):
    y = audio_tuple[1]
    sr_in = audio_tuple[0]

    # 转 float
    if y.dtype != np.float32:
        y = y.astype(np.float32)

        # 如果是 int16，要归一化
        if np.max(np.abs(y)) > 1:
            y = y / 32768.0

    # stereo → mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # resample
    y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)

    return y

def auto_mix(music_tuple, sfx_tuple):
    
    music = load_audio(music_tuple)
    sfx = load_audio(sfx_tuple)

    sfx = match_length(music, sfx)

    # features
    rms_m = compute_rms(music)
    centroid_m, _, low_m, mid_m = spectral_features(music, sr)

    rms_s = compute_rms(sfx)
    centroid_s, bandwidth_s, low_s, _ = spectral_features(sfx, sr)

    energy_ratio = rms_s/(rms_m+1e-6)

    feature = [rms_m, centroid_m, low_m, mid_m,
               rms_s, centroid_s, low_s, bandwidth_s, energy_ratio]

    x = scaler.transform([feature])
    x = torch.tensor(x, dtype=torch.float32)

    pred = mix_model(x).detach().numpy()[0]

    gain = int(pred[0]*15)
    cutoff = int(pred[1]*150 + 100)

    processed = high_pass(apply_gain(sfx, gain), sr, cutoff)

    score, detail = mixing_score(music, processed)

    mixed = music + processed
    mixed = mixed/(np.max(np.abs(mixed))+1e-6)

    info = {
        "features": feature,
        "gain": gain,
        "cutoff": cutoff,
        "score": score,
        "detail": detail
    }

    return (sr, mixed), info, gain, cutoff, (sr, processed)

# =========================
# 6. 手动混音
# =========================
def manual_mix(music_tuple, sfx_tuple, gain, cutoff):
    
    music = load_audio(music_tuple)
    sfx = load_audio(sfx_tuple)

    sfx = match_length(music, sfx)

    processed = high_pass(apply_gain(sfx, gain), sr, cutoff)
    score, detail = mixing_score(music, processed)

    mixed = music + processed
    mixed = mixed/(np.max(np.abs(mixed))+1e-6)

    return (sr, mixed), (sr, processed), score, detail

# =========================
# 7. Gradio UI
# =========================
with gr.Blocks() as demo:

    gr.Markdown("# 🎵 AI Audio Mixing Demo")

    with gr.Row():
        music_prompt = gr.Textbox(label="Music Prompt")
        sfx_prompt = gr.Textbox(label="SFX Prompt")

    with gr.Row():
        gen_music_btn = gr.Button("Generate Music")
        gen_sfx_btn = gr.Button("Generate SFX")

    music_audio = gr.Audio(label="Music")
    sfx_audio = gr.Audio(label="SFX")

    gen_music_btn.click(generate_music, music_prompt, music_audio)
    gen_sfx_btn.click(generate_sfx, sfx_prompt, sfx_audio)

    mix_btn = gr.Button("Auto Mix")

    mixed_audio = gr.Audio(label="Mixed")
    processed_audio = gr.Audio(label="Processed SFX")

    info_box = gr.JSON(label="Mix Info")

    gain_slider = gr.Slider(-20, 20, value=0, step=1, label="Gain (dB)")
    cutoff_slider = gr.Slider(50, 500, value=200, step=10, label="Cutoff (Hz)")

    mix_btn.click(
        auto_mix,
        inputs=[music_audio, sfx_audio],
        outputs=[mixed_audio, info_box, gain_slider, cutoff_slider, processed_audio]
    )

    manual_btn = gr.Button("Manual Remix")

    score_box = gr.Number(label="Score")
    detail_box = gr.JSON(label="Detail")

    manual_btn.click(
        manual_mix,
        inputs=[music_audio, sfx_audio, gain_slider, cutoff_slider],
        outputs=[mixed_audio, processed_audio, score_box, detail_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
