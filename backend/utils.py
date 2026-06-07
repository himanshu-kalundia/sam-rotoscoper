import cv2
import os
import shutil
import zipfile
import numpy as np
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path):
    """
    Extracts all frames from a video file as JPEGs and returns video metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_name = f"{frame_idx:05d}.jpg"
        frame_path = output_dir / frame_name
        # Save as high-quality JPEG
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1
        
    cap.release()
    
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "frame_count": frame_idx
    }

def create_video_from_frames(
    frames_dir: Path,
    mask_dir: Path,
    output_path: Path,
    fps: float,
    export_mode: str,
    overlay_color=(0, 255, 128),  # BGR for neon mint green
    alpha=0.5
):
    """
    Stitches frame images together according to export_mode:
    - 'overlay': overlays mask in a transparent color on original frame
    - 'green_screen': subject kept, background replaced with pure green (0, 255, 0)
    - 'cutout': subject kept, background replaced with pure black (0, 0, 0)
    """
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
        
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Setup VideoWriter - MP4V is universally compatible for web browsers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            
            # Check if mask exists for this frame
            mask_file = mask_dir / f"{idx:05d}.png"
            if mask_file.exists():
                # Read mask as BGRA to preserve Alpha transparency channel
                mask_bgra = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask_bgra is not None:
                    if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 4:
                        mask = mask_bgra[:, :, 3]  # Use Alpha channel as the mask
                    else:
                        mask = cv2.cvtColor(mask_bgra, cv2.COLOR_BGR2GRAY)
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                
            # Perform stitching depending on export_mode
            if export_mode == 'overlay':
                # Create colored overlay
                overlay = frame.copy()
                overlay[mask > 0] = overlay_color
                # Blend original frame and colored overlay
                output_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
            elif export_mode == 'green_screen':
                # Green background BGR = (0, 255, 0)
                bg = np.zeros_like(frame)
                bg[:] = [0, 255, 0]
                # Apply mask: keep frame where mask is active, green otherwise
                output_frame = np.where(mask[:, :, np.newaxis] > 0, frame, bg)
                
            elif export_mode == 'cutout':
                # Black background BGR = (0, 0, 0)
                bg = np.zeros_like(frame)
                # Keep subject, black elsewhere
                output_frame = np.where(mask[:, :, np.newaxis] > 0, frame, bg)
                
            else:
                # Default fallback is original frame
                output_frame = frame
                
            out.write(output_frame)
    finally:
        out.release()

def create_transparent_png_sequence_zip(
    frames_dir: Path,
    mask_dir: Path,
    output_zip_path: Path
):
    """
    Creates a zip archive containing high-quality transparent PNGs (BGRA) of the segmented objects.
    This is extremely useful for professional video editing workflows (After Effects, Resolve).
    """
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
        
    temp_dir = output_zip_path.parent / "temp_png_seq"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            height, width, _ = frame.shape
            
            # Check if mask exists for this frame
            mask_file = mask_dir / f"{idx:05d}.png"
            if mask_file.exists():
                # Read mask as BGRA to preserve Alpha transparency channel
                mask_bgra = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask_bgra is not None:
                    if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 4:
                        mask = mask_bgra[:, :, 3]  # Use Alpha channel as the mask
                    else:
                        mask = cv2.cvtColor(mask_bgra, cv2.COLOR_BGR2GRAY)
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                
            # Create a 4-channel BGRA image
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            # Set alpha channel based on mask (255 inside mask, 0 outside)
            bgra[:, :, 3] = mask
            
            # Save transparent PNG to temporary folder
            png_name = f"cutout_{idx:05d}.png"
            cv2.imwrite(str(temp_dir / png_name), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
        # Zip the directory contents
        with zipfile.ZipFile(str(output_zip_path), 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for png_file in temp_dir.glob("*.png"):
                zip_file.write(str(png_file), arcname=png_file.name)
                
    finally:
        # Cleanup temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
