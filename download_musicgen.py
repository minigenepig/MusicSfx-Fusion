import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/musicgen-medium",
    local_dir="models/musicgen-medium",
    max_workers=2,
    resume_download=True
)