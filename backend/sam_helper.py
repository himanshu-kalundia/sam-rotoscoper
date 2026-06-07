import torch
import numpy as np
import cv2
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from backend.config import SAM2_MODELS, DEFAULT_MODEL, CHECKPOINT_DIR
from sam2.build_sam import build_sam2_video_predictor

# Global predictor instance
_predictor = None
_current_model_size = None

# Global dict to store active session inference states: session_id -> inference_state
_inference_states = {}

# Global dict to store active session model sizes: session_id -> model_size
_session_model_sizes = {}

def download_weights(model_size: str, dest_path: Path, progress_callback=None):
    """
    Downloads model weights from Meta's server dynamically on demand.
    """
    if model_size not in SAM2_MODELS:
        raise ValueError(f"Invalid model size: {model_size}")
        
    model_info = SAM2_MODELS[model_size]
    url = model_info["url"]
    
    print("\n" + "=" * 50)
    print(f" Downloading SAM 2.1 {model_size.upper()} Weights")
    print(f" Source: {url}")
    print(f" Destination: {dest_path}")
    print("=" * 50)
    
    def progress_hook(count, block_size, total_size):
        total_size_val = total_size if total_size > 0 else 1
        percent = int(count * block_size * 100 / total_size_val)
        percent = min(100, max(0, percent))
        current_mb = count * block_size // 1024 // 1024
        total_mb = total_size // 1024 // 1024 if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading... {percent}% ({current_mb}MB / {total_mb}MB)")
        sys.stdout.flush()
        if progress_callback:
            try:
                progress_callback(percent, current_mb, total_mb)
            except Exception:
                pass
        
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest_path), progress_hook)
        print("\nDownload completed successfully!")
    except Exception as e:
        # If download fails, delete the partially downloaded file so we can retry later
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to download model weights: {e}")

def get_predictor(model_size: str = None):
    """
    Lazy loads and returns the SAM 2 Video Predictor for the specified model_size.
    Forces CUDA usage for performance. If switching models, cleans up the old one first.
    """
    global _predictor, _current_model_size
    
    if model_size is None:
        model_size = _current_model_size or DEFAULT_MODEL
        
    if model_size not in SAM2_MODELS:
        model_size = DEFAULT_MODEL
        
    # If the requested model matches the currently loaded model, return it
    if _predictor is not None and _current_model_size == model_size:
        return _predictor
        
    # Otherwise, clean up the currently loaded model to save VRAM
    if _predictor is not None:
        print(f"Switching SAM 2 model from {_current_model_size} to {model_size}...")
        _predictor = None
        _current_model_size = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Previous model released and CUDA memory cache cleared.")
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SAM 2 ({model_size}) model on {device}...")
    
    if device == "cpu":
        print(f"WARNING: CUDA is not available. SAM 2 ({model_size}) will run EXTREMELY slowly on CPU!")
        
    model_info = SAM2_MODELS[model_size]
    ckpt_path = CHECKPOINT_DIR / model_info["checkpoint"]
    config_file = model_info["config"]
    
    # Check if checkpoint exists, download if missing
    if not ckpt_path.exists():
        download_weights(model_size, ckpt_path)
        
    # Build predictor
    _predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(ckpt_path),
        device=device
    )
    _current_model_size = model_size
    print(f"SAM 2 {model_size.upper()} Model loaded successfully!")
    
    return _predictor

def init_session_state(session_id: str, frames_dir: Path, model_size: str = "large"):
    """
    Initializes a new tracking state for the video frames in frames_dir.
    """
    # Close any existing session with the same ID first
    cleanup_session_state(session_id)
    
    # Store session model size
    _session_model_sizes[session_id] = model_size
    
    # Load correct predictor
    predictor = get_predictor(model_size)
    
    # Initialize the video predictor inference state
    inference_state = predictor.init_state(video_path=str(frames_dir))
    _inference_states[session_id] = inference_state
    return inference_state

def get_session_state(session_id: str):
    """
    Retrieves the tracking state for a given session.
    """
    if session_id not in _inference_states:
        raise ValueError(f"Session {session_id} not initialized or has been cleaned up.")
    return _inference_states[session_id]

def add_click(
    session_id: str,
    frame_idx: int,
    obj_id: int,
    coords: List[Tuple[float, float]],
    labels: List[int]
) -> np.ndarray:
    """
    Adds interactive points (clicks) on a specific frame and returns the updated mask.
    coords: List of [x, y] coordinates in original resolution
    labels: List of integers (1 = positive click, 0 = negative click)
    """
    model_size = _session_model_sizes.get(session_id, DEFAULT_MODEL)
    predictor = get_predictor(model_size)
    inference_state = get_session_state(session_id)
    
    # Format inputs for PyTorch
    points_np = np.array(coords, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    
    # Perform standard automatic float casting (e.g. bfloat16 or float16) depending on hardware capability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else torch.autocast(device_type=device, enabled=False)
    
    with torch.inference_mode(), autocast_ctx:
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np
        )
        
    # Process output logits to binary mask (height, width)
    # out_mask_logits shape is typically (num_objects, 1, height, width) or (num_objects, height, width)
    mask_logits = out_mask_logits[0]
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.cpu().numpy()
        
    binary_mask = (mask_logits > 0.0).astype(np.uint8) * 255
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[0]
        
    return binary_mask

def clear_frame_prompts(
    session_id: str,
    frame_idx: int,
    obj_id: int
):
    """
    Clears all prompts (clicks/masks) on a specific frame for a given session.
    """
    model_size = _session_model_sizes.get(session_id, DEFAULT_MODEL)
    predictor = get_predictor(model_size)
    inference_state = get_session_state(session_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else torch.autocast(device_type=device, enabled=False)
    
    with torch.inference_mode(), autocast_ctx:
        predictor.clear_all_prompts_in_frame(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            need_output=False
        )

def propagate(session_id: str, mask_output_dir: Path, progress_callback=None) -> int:
    """
    Propagates prompts throughout the entire video and writes mask PNG files to disk.
    progress_callback: optional callable receiving (current_frame, total_frames)
    """
    model_size = _session_model_sizes.get(session_id, DEFAULT_MODEL)
    predictor = get_predictor(model_size)
    inference_state = get_session_state(session_id)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_frames = len(inference_state["images"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else torch.autocast(device_type=device, enabled=False)
    
    frame_count = 0
    with torch.inference_mode(), autocast_ctx:
        # Generator for temporal tracking
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # Process output to single channel mask (height, width)
            mask_logits = out_mask_logits[0]
            if isinstance(mask_logits, torch.Tensor):
                mask_logits = mask_logits.cpu().numpy()
                
            binary_mask = (mask_logits > 0.0).astype(np.uint8) * 255
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[0]
                
            # Create a 4-channel BGRA image for true alpha transparency
            bgra_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
            bgra_mask[:, :, :3] = 255  # RGB = White
            bgra_mask[:, :, 3] = binary_mask  # Alpha channel is the binary mask itself
            
            # Write mask to disk
            mask_path = mask_output_dir / f"{out_frame_idx:05d}.png"
            cv2.imwrite(str(mask_path), bgra_mask)
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)
                
    return frame_count

def cleanup_session_state(session_id: str):
    """
    Cleans up predictor state for a session and frees up GPU memory.
    """
    global _inference_states, _predictor, _session_model_sizes
    if session_id in _inference_states:
        print(f"Cleaning up SAM 2 session state: {session_id}...")
        try:
            model_size = _session_model_sizes.get(session_id, DEFAULT_MODEL)
            predictor = get_predictor(model_size)
            if predictor is not None:
                predictor.reset_state(_inference_states[session_id])
        except Exception as e:
            print(f"Error resetting SAM 2 state: {e}")
            
        del _inference_states[session_id]
        if session_id in _session_model_sizes:
            del _session_model_sizes[session_id]
        
        # Clear CUDA memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared.")
