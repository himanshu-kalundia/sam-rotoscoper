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

# SAM 2 Model Configuration
# We are using the high-accuracy Large SAM 2.1 model
SAM2_MODEL_TYPE = "sam2.1_hiera_large"
SAM2_CHECKPOINT = CHECKPOINT_DIR / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Matches configs in sam2 repo

# Server Configuration
PORT = 8000
HOST = "0.0.0.0"
