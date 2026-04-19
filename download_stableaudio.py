import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    local_dir="models/stable-audio-open-1.0",
    max_workers=2,
    resume_download=True
)