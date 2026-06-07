import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from backend.config import SAM2_CONFIG, SAM2_CHECKPOINT
from sam2.build_sam import build_sam2_video_predictor

# Global predictor instance
_predictor = None

# Global dict to store active session inference states: session_id -> inference_state
_inference_states = {}

def get_predictor():
    """
    Lazy loads and returns the SAM 2 Video Predictor.
    Forces CUDA usage for performance.
    """
    global _predictor
    if _predictor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 2 model on {device}...")
        
        if device == "cpu":
            print("WARNING: CUDA is not available. SAM 2 Large will run EXTREMELY slowly on CPU!")
            
        # Build predictor
        # Model config is resolved from installed package since we used editable mode
        _predictor = build_sam2_video_predictor(
            config_file=SAM2_CONFIG,
            ckpt_path=str(SAM2_CHECKPOINT),
            device=device
        )
        print("SAM 2 Large Model loaded successfully!")
    return _predictor

def init_session_state(session_id: str, frames_dir: Path):
    """
    Initializes a new tracking state for the video frames in frames_dir.
    """
    predictor = get_predictor()
    # Close any existing session with the same ID first
    cleanup_session_state(session_id)
    
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
    predictor = get_predictor()
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
    predictor = get_predictor()
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
    predictor = get_predictor()
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
    global _inference_states, _predictor
    if session_id in _inference_states:
        print(f"Cleaning up SAM 2 session state: {session_id}...")
        try:
            if _predictor is not None:
                _predictor.reset_state(_inference_states[session_id])
        except Exception as e:
            print(f"Error resetting SAM 2 state: {e}")
            
        del _inference_states[session_id]
        
        # Clear CUDA memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared.")
