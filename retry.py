import os
import json
import random
import re
import numpy as np
import librosa
from scipy import signal
from tqdm import tqdm

# ========= 路径 =========
music_dir = "/root/autodl-tmp/audio_dataset/1_music"
sfx_dir = "/root/autodl-tmp/audio_dataset/1_sfx"

fail_txt = "/root/autodl-tmp/fail_cases_1.txt"
output_json = "/root/autodl-tmp/mix_dataset_1_retry.json"

sr = 22050
MAX_TRY = 50
SCORE_THRESHOLD = 0.97

# ========= 读取已有数据 =========
if os.path.exists(output_json):
    with open(output_json, "r") as f:
        results = json.load(f)
else:
    results = []

# ========= 读取失败 index =========
fail_indices = []
with open(fail_txt, "r") as f:
    for line in f:
        match = re.search(r'index (\d+)', line)
        if match:
            fail_indices.append(int(match.group(1)))

print(f"需要重试的数量: {len(fail_indices)}")

# ========= 工具函数（和你原来一样） =========

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
        times = len(music) // len(sfx) + 1
        sfx = np.tile(sfx, times)
    return sfx[:len(music)]


def apply_gain(y, db):
    return y * (10 ** (db / 20))


def high_pass(y, sr, cutoff):
    b, a = signal.butter(2, cutoff/(sr/2), btype='high')
    return signal.filtfilt(b, a, y)


def audibility_score(music, sfx):
    return float(1 - np.exp(-5 * compute_rms(sfx) / (compute_rms(music)+1e-6)))


def balance_score(music, sfx):
    ratio = compute_rms(sfx) / (compute_rms(music)+1e-6)
    diff = max(0, ratio - 0.4)
    return float(np.exp(-5 * diff))


def spectral_separation_score(music, sfx, sr):
    S1 = np.abs(librosa.stft(music, n_fft=1024))
    S2 = np.abs(librosa.stft(sfx, n_fft=1024))
    S1 /= (np.sum(S1)+1e-6)
    S2 /= (np.sum(S2)+1e-6)
    return float(-np.sum(np.minimum(S1, S2)))


def mixing_score(music, processed_sfx, sr):
    a = audibility_score(music, processed_sfx)
    b = balance_score(music, processed_sfx)
    c = spectral_separation_score(music, processed_sfx, sr)
    return 1.0*a + 0.3*b + 0.5*c


# ========= 开始重试 =========

new_success = 0

for i in tqdm(fail_indices):

    music_path = os.path.join(music_dir, f"music_{i}.wav")
    sfx_path = os.path.join(sfx_dir, f"sfx_{i}.wav")

    if not os.path.exists(music_path) or not os.path.exists(sfx_path):
        continue

    music, _ = librosa.load(music_path, sr=sr)
    sfx, _ = librosa.load(sfx_path, sr=sr)

    sfx = match_length(music, sfx)

    # ===== 特征 =====
    rms_m = compute_rms(music)
    centroid_m, _, low_m, mid_m = spectral_features(music, sr)

    rms_s = compute_rms(sfx)
    centroid_s, bandwidth_s, low_s, _ = spectral_features(sfx, sr)

    energy_ratio = rms_s / (rms_m + 1e-6)

    for _ in range(MAX_TRY):

        # 🔥 可选优化：引导采样（强烈建议）
        if energy_ratio < 0.1:
            gain = random.randint(5, 15)
        else:
            gain = random.randint(-10, 5)

        cutoff = random.randint(100, 250)

        processed_sfx = apply_gain(sfx, gain)
        processed_sfx = high_pass(processed_sfx, sr, cutoff)

        score = mixing_score(music, processed_sfx, sr)

        if score > SCORE_THRESHOLD:
            new_success += 1

            results.append({
                "index": i,
                "music_feat": {
                    "rms": float(rms_m),
                    "centroid": float(centroid_m),
                    "low_ratio": float(low_m),
                    "mid_ratio": float(mid_m),
                },
                "sfx_feat": {
                    "rms": float(rms_s),
                    "centroid": float(centroid_s),
                    "low_ratio": float(low_s),
                    "bandwidth": float(bandwidth_s),
                },
                "energy_ratio": float(energy_ratio),
                "label": {
                    "gain": int(gain),
                    "high_pass_cutoff": int(cutoff)
                },
                "score": float(score)
            })

# ========= 保存 =========
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print("✅ 重试完成")
print(f"新增成功样本: {new_success}")