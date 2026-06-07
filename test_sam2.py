import torch
import numpy as np
import sys
from pathlib import Path

print("=========================================")
print(" Running SAM 2 Large Diagnostic Test")
print("=========================================")

print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) Available: {cuda_available}")

if not cuda_available:
    print("ERROR: CUDA is not available! SAM 2 Large requires GPU acceleration.")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"Detected GPU: {device_name}")
print(f"VRAM Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

checkpoint_path = Path("checkpoints/sam2.1_hiera_large.pt")
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint file not found at {checkpoint_path}! Wait for weight download to finish.")
    sys.exit(1)

try:
    print("\nLoading Segment Anything 2 Hiera Large model...")
    from sam2.build_sam import build_sam2_video_predictor
    
    # SAM 2 config and checkpoint path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    predictor = build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=str(checkpoint_path),
        device="cuda"
    )
    print("SUCCESS: Predictor compiled and loaded successfully on GPU!")
    print("System is 100% ready for video rotoscoping.")
    
except Exception as e:
    print(f"ERROR: Failed to initialize SAM 2 Predictor: {e}")
    sys.exit(1)
