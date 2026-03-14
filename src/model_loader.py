"""
Model Loader — download YOLO weights from Hugging Face Hub.

On first run the weights are fetched from the HF repo and cached locally
inside `LOCAL_MODEL_DIR` (default: ``weights/``).  Subsequent runs reuse
the cached file.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def get_model_path(repo_id: str, filename: str,
                   local_dir: str = "weights") -> str:
    """
    Return a local path to the YOLO ``.pt`` file.

    If the file already exists in *local_dir* it is returned immediately.
    Otherwise it is downloaded from the Hugging Face Hub.

    Args:
        repo_id:   HF repository, e.g. ``"username/helmetguard-ai"``.
        filename:  Name of the weights file inside the repo, e.g. ``"best.pt"``.
        local_dir: Directory where the weights are cached locally.

    Returns:
        Absolute path to the downloaded (or cached) weights file.
    """
    local_path = Path(local_dir) / filename

    if local_path.exists():
        print(f"✅ Model already cached → {local_path}")
        return str(local_path)

    print(f"⬇️  Downloading model from HuggingFace: {repo_id}/{filename} …")
    os.makedirs(local_dir, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )

    print(f"✅ Model saved → {downloaded}")
    return str(downloaded)
