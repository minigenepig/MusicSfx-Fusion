import torch
import pickle
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from IPython.display import Audio, display

# =====================
# 1. 路径
# =====================
music_path = "/root/autodl-tmp/audio_dataset/2_music/music_10.wav"
sfx_path = "/root/autodl-tmp/audio_dataset/2_sfx/sfx_10.wav"

sr = 22050

# =====================
# 2. 工具函数（和训练一致）
# =====================
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

    low_ratio = low / total
    mid_ratio = mid / total

    return centroid, bandwidth, low_ratio, mid_ratio

def match_length(music, sfx):
    if len(sfx) < len(music):
        times = len(music) // len(sfx) + 1
        sfx = np.tile(sfx, times)
    return sfx[:len(music)]

def apply_gain(y, db):
    return y * (10 ** (db / 20))

def high_pass(y, sr, cutoff=200):
    b, a = signal.butter(2, cutoff/(sr/2), btype='high')
    return signal.filtfilt(b, a, y)


# =====================
# ⭐ 评分函数（补上这个）
# =====================
def compute_rms(y):
    return np.mean(librosa.feature.rms(y=y)[0])

def audibility_score(music, sfx):
    rms_m = compute_rms(music)
    rms_s = compute_rms(sfx)
    ratio = rms_s / (rms_m + 1e-6)
    return float(1 - np.exp(-5 * ratio))

def balance_score(music, sfx):
    rms_m = compute_rms(music)
    rms_s = compute_rms(sfx)
    ratio = rms_s / (rms_m + 1e-6)

    target_max = 0.4
    diff = max(0, ratio - target_max)
    return float(np.exp(-5 * diff))

def spectral_separation_score(music, sfx, sr):
    S1 = np.abs(librosa.stft(music, n_fft=1024))
    S2 = np.abs(librosa.stft(sfx, n_fft=1024))

    S1 = S1 / (np.sum(S1) + 1e-6)
    S2 = S2 / (np.sum(S2) + 1e-6)

    overlap = np.sum(np.minimum(S1, S2))
    return float(-overlap)

def mixing_score(music, processed_sfx, sr):
    a = audibility_score(music, processed_sfx)
    b = balance_score(music, processed_sfx)
    c = spectral_separation_score(music, processed_sfx, sr)

    score = 1.0*a + 0.3*b + 0.5*c

    detail = {
        "audibility": round(a, 4),
        "balance": round(b, 4),
        "separation": round(c, 4)
    }

    return round(score, 4), detail


# =====================
# 3. 加载模型 & scaler
# =====================
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
        x = self.net(x)
        x = torch.tanh(x)   # ⭐和训练保持一致（如果你加了）
        return x

model = MLP()
model.load_state_dict(torch.load("/root/sfx/best_model.pth"))
model.eval()

with open("/root/autodl-tmp/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =====================
# 4. 读取音频
# =====================
music, _ = librosa.load(music_path, sr=sr)
sfx, _ = librosa.load(sfx_path, sr=sr)

sfx = match_length(music, sfx)

# =====================
# 5. 提取特征
# =====================
rms_m = compute_rms(music)
centroid_m, _, low_m, mid_m = spectral_features(music, sr)

rms_s = compute_rms(sfx)
centroid_s, bandwidth_s, low_s, _ = spectral_features(sfx, sr)

energy_ratio = rms_s / (rms_m + 1e-6)

feature = [
    rms_m,
    centroid_m,
    low_m,
    mid_m,
    rms_s,
    centroid_s,
    low_s,
    bandwidth_s,
    energy_ratio
]

# =====================
# 6. 模型预测
# =====================
x = scaler.transform([feature])
x = torch.tensor(x, dtype=torch.float32)

pred = model(x).detach().numpy()[0]

# 反归一化
gain = pred[0] * 15
cutoff = pred[1] * 150 + 100

gain = int(round(gain))
cutoff = int(round(cutoff))

print("🎯 模型预测参数：")
print("gain:", gain, "dB")
print("cutoff:", cutoff, "Hz")

# =====================
# 7. 应用处理
# =====================
processed_sfx = apply_gain(sfx, gain)
processed_sfx = high_pass(processed_sfx, sr, cutoff)

score, detail = mixing_score(music, processed_sfx, sr)

print("⭐ 模型混音评分:", score)
print(detail)

# =====================
# 8. 混音
# =====================
mixed = music + processed_sfx
mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)

# =====================
# 9. 播放
# =====================
print("🎵 Music:")
display(Audio(music, rate=sr))

print("🌧️ Processed SFX:")
display(Audio(processed_sfx, rate=sr))

print("🎧 Mixed:")
display(Audio(mixed, rate=sr))

# =====================
# 10. 保存
# =====================
sf.write("mixed_output.wav", mixed, sr)
print("✅ 已保存 mixed_output.wav")