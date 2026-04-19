import os
import json
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ======================
# 1. 配置路径
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

local_path = "/root/sfx/models/musicgen-medium"
json_path = "paired_prompt_dataset.json"

output_dir = "/root/autodl-tmp/audio_dataset/4_music"
os.makedirs(output_dir, exist_ok=True)

# ======================
# 2. 加载模型
# ======================
processor = AutoProcessor.from_pretrained(
    local_path,
    local_files_only=True
)

model = MusicgenForConditionalGeneration.from_pretrained(
    local_path,
    local_files_only=True
).to(device)

model.eval()

# ======================
# 3. 读取 JSON 数据
# ======================
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# ======================
# 4. 遍历生成
# ======================
for item in data:
    prompt = item["music_prompt_en"]
    index = item["index"]

    print(f"Generating index {index} ...")

    # 编码输入
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 推理
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=512
        )

    # 转 numpy
    audio = audio_values[0].cpu().numpy()

    # 保存路径
    out_path = os.path.join(output_dir, f"music_{index}.wav")

    # 写文件（MusicGen一般是 32kHz）
    sf.write(out_path, audio.T, samplerate=32000)

    print(f"Saved: {out_path}")

print("全部生成完成！")