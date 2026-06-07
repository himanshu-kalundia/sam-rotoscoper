import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
WORKSPACE_DIR = BASE_DIR / "workspaces"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "exports"

# Create directories if they do not exist
for directory in [UPLOAD_DIR, WORKSPACE_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# SAM 2 Model Configuration Mappings
SAM2_MODELS = {
    "tiny": {
        "model_type": "sam2.1_hiera_tiny",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "label": "SAM 2.1 Tiny (Fastest / Low VRAM)"
    },
    "small": {
        "model_type": "sam2.1_hiera_small",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "label": "SAM 2.1 Small"
    },
    "medium": {
        "model_type": "sam2.1_hiera_base_plus",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "label": "SAM 2.1 Medium (Base+)"
    },
    "large": {
        "model_type": "sam2.1_hiera_large",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "label": "SAM 2.1 Large (High Accuracy)"
    }
}

DEFAULT_MODEL = "large"

# Server Configuration
PORT = 8000
HOST = "0.0.0.0"
