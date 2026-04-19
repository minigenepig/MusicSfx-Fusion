import os
import json
import torch
import soundfile as sf
from diffusers import StableAudioPipeline

# ======================
# 1. 基本配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "/root/sfx/models/stable-audio-open-1.0"
json_path = "paired_prompt_dataset.json"

output_dir = "/root/autodl-tmp/audio_dataset/4_sfx"
os.makedirs(output_dir, exist_ok=True)

# ======================
# 2. 加载模型
# ======================
pipe = StableAudioPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
)

pipe = pipe.to(device)

# 可选优化（显存更稳）
pipe.enable_model_cpu_offload()  # 如果显存紧张可以打开

# ======================
# 3. 读取 JSON
# ======================
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# ======================
# 4. 遍历生成 SFX
# ======================
for item in data:
    prompt = item["sfx_prompt_en"]
    index = item["index"]

    out_path = os.path.join(output_dir, f"sfx_{index}.wav")

    # 👉 跳过已生成（很重要）
    if os.path.exists(out_path):
        print(f"Skip {index}, already exists")
        continue

    print(f"Generating SFX {index} ...")

    try:
        # 生成音频
        audio = pipe(
            prompt,
            num_inference_steps=200,
            audio_end_in_s=10.0   # ✅ 精确 10 秒
        ).audios[0]

        # 保存
        sf.write(
            out_path,
            audio.to(torch.float32).cpu().numpy().T,
            pipe.vae.sampling_rate
        )

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error at index {index}: {e}")
        continue

    # 👉 防止显存累积
    if device == "cuda":
        torch.cuda.empty_cache()

print("全部生成完成！")