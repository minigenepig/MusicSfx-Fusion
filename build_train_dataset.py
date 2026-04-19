import os
import json
import random
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from tqdm import tqdm

# =====================
# 基础配置
# =====================
music_dir = "/root/autodl-tmp/audio_dataset/4_music"
sfx_dir = "/root/autodl-tmp/audio_dataset/4_sfx"

output_json = "/root/autodl-tmp/mix_dataset_4.json"
fail_txt = "/root/autodl-tmp/fail_cases_4.txt"

sr = 22050
MAX_TRY = 40
SCORE_THRESHOLD = 0.97

results = []
fail_cases = []

# =====================
# 工具函数
# =====================

def compute_rms(y):
    return np.mean(librosa.feature.rms(y=y)[0])


def spectral_features(y, sr):
    n_fft = 1024

    S = np.abs(librosa.stft(y, n_fft=n_fft))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))

    # 频带划分
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
# 评分函数
# =====================

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
# 主流程
# =====================

for i in tqdm(range(1, 143)):
    music_path = os.path.join(music_dir, f"music_{i}.wav")
    sfx_path = os.path.join(sfx_dir, f"sfx_{i}.wav")

    if not os.path.exists(music_path) or not os.path.exists(sfx_path):
        continue

    music, _ = librosa.load(music_path, sr=sr)
    sfx, _ = librosa.load(sfx_path, sr=sr)

    sfx = match_length(music, sfx)

    # ===== 提取特征（固定，只算一次）=====
    rms_m = compute_rms(music)
    centroid_m, _, low_m, mid_m = spectral_features(music, sr)

    rms_s = compute_rms(sfx)
    centroid_s, bandwidth_s, low_s, _ = spectral_features(sfx, sr)

    energy_ratio = rms_s / (rms_m + 1e-6)

    found = False

    # ===== 多次尝试 =====
    for _ in range(MAX_TRY):

        gain = random.randint(-15, 15)
        cutoff = random.randint(100, 250)

        processed_sfx = apply_gain(sfx, gain)
        processed_sfx = high_pass(processed_sfx, sr, cutoff)

        # 混音
        mixed = music + processed_sfx
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)

        score, detail = mixing_score(music, processed_sfx, sr)

        if score > SCORE_THRESHOLD:
            found = True

            data = {
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
            }

            results.append(data)

    if not found:
        fail_cases.append(f"index {i} 没有找到满足条件的组合")

# =====================
# 保存结果
# =====================

with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

with open(fail_txt, "w") as f:
    for line in fail_cases:
        f.write(line + "\n")

print("✅ 数据生成完成")
print(f"成功样本数: {len(results)}")
print(f"失败样本数: {len(fail_cases)}")