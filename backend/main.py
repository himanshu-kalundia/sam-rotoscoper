import os
import uuid
import shutil
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict

# Import local backend modules
from backend.config import UPLOAD_DIR, WORKSPACE_DIR, OUTPUT_DIR, BASE_DIR
from backend import utils
from backend import sam_helper

app = FastAPI(title="SAM 2 Video Rotoscoper")

# Allow CORS for development versatility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track background task progress: session_id -> progress_dict
task_progress: Dict[str, dict] = {}

class ClickRequest(BaseModel):
    session_id: str
    frame_idx: int
    obj_id: int
    coords: List[Tuple[float, float]]
    labels: List[int]

class PropagateRequest(BaseModel):
    session_id: str

class ExportRequest(BaseModel):
    session_id: str
    export_mode: str  # 'overlay', 'green_screen', 'cutout', 'zip'

@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """
    Endpoint to upload a video, parse properties, extract frames,
    and initialize the SAM 2 tracking state.
    """
    try:
        session_id = uuid.uuid4().hex
        
        # Save uploaded video file
        temp_video_path = UPLOAD_DIR / f"{session_id}.mp4"
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        # Create workspace directories
        session_workspace = WORKSPACE_DIR / session_id
        frames_dir = session_workspace / "raw_frames"
        masks_dir = session_workspace / "masks"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames and get video properties
        metadata = utils.extract_frames(temp_video_path, frames_dir)
        
        # Initialize SAM 2 inference state
        sam_helper.init_session_state(session_id, frames_dir)
        
        # Register progress status
        task_progress[session_id] = {
            "status": "idle",
            "current": 0,
            "total": metadata["frame_count"],
            "message": ""
        }
        
        return {
            "session_id": session_id,
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"]
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/frames/{session_id}/{frame_idx}")
async def get_frame(session_id: str, frame_idx: int):
    """
    Serves raw frame image from disk.
    """
    frame_path = WORKSPACE_DIR / session_id / "raw_frames" / f"{frame_idx:05d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path)

@app.get("/api/masks/{session_id}/{frame_idx}")
async def get_mask(session_id: str, frame_idx: int):
    """
    Serves predicted mask PNG image from disk if it exists.
    Otherwise returns 404 so client knows there is no mask.
    """
    mask_path = WORKSPACE_DIR / session_id / "masks" / f"{frame_idx:05d}.png"
    if not mask_path.exists():
        return JSONResponse(status_code=404, content={"detail": "Mask not generated for this frame"})
    return FileResponse(mask_path)

@app.post("/api/click")
async def register_click(req: ClickRequest):
    """
    Registers a new set of points/labels for a frame, runs inference,
    saves the mask, and returns success.
    """
    try:
        session_workspace = WORKSPACE_DIR / req.session_id
        masks_dir = session_workspace / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{req.frame_idx:05d}.png"
        
        # If coords are empty, it means we are clearing prompts for this frame
        if not req.coords:
            sam_helper.clear_frame_prompts(
                session_id=req.session_id,
                frame_idx=req.frame_idx,
                obj_id=req.obj_id
            )
            # Delete mask file if it exists, so the client renders a blank frame correctly
            if mask_path.exists():
                mask_path.unlink()
            return {"status": "success", "frame_idx": req.frame_idx}
        
        # Add click and run frame predictor
        binary_mask = sam_helper.add_click(
            session_id=req.session_id,
            frame_idx=req.frame_idx,
            obj_id=req.obj_id,
            coords=req.coords,
            labels=req.labels
        )
        
        # Create a 4-channel BGRA image for true alpha transparency
        import numpy as np
        import cv2
        bgra_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
        bgra_mask[:, :, :3] = 255  # RGB = White
        bgra_mask[:, :, 3] = binary_mask  # Alpha channel is the binary mask itself
        
        # Save mask as a transparent PNG
        cv2.imwrite(str(mask_path), bgra_mask)
        
        return {"status": "success", "frame_idx": req.frame_idx}
        
    except Exception as e:
        print(f"Click prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_propagation_background(session_id: str, masks_dir: Path):
    """
    Background worker for running SAM 2 tracking.
    """
    try:
        task_progress[session_id]["status"] = "propagating"
        task_progress[session_id]["message"] = "Segmenting and propagating mask across video..."
        
        def progress_callback(current, total):
            task_progress[session_id]["current"] = current
            task_progress[session_id]["total"] = total
            
        sam_helper.propagate(
            session_id=session_id,
            mask_output_dir=masks_dir,
            progress_callback=progress_callback
        )
        
        task_progress[session_id]["status"] = "completed"
        task_progress[session_id]["message"] = "Mask propagation completed successfully!"
        
    except Exception as e:
        print(f"Background propagation error: {e}")
        task_progress[session_id]["status"] = "failed"
        task_progress[session_id]["message"] = f"Propagation failed: {str(e)}"

@app.post("/api/propagate")
async def trigger_propagation(req: PropagateRequest, background_tasks: BackgroundTasks):
    """
    Triggers mask tracking throughout the video in a background thread.
    """
    session_workspace = WORKSPACE_DIR / req.session_id
    masks_dir = session_workspace / "masks"
    
    if req.session_id not in task_progress:
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    background_tasks.add_task(run_propagation_background, req.session_id, masks_dir)
    return {"status": "started"}

@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """
    Retrieves progress status of active background operations (propagation/export).
    """
    if session_id not in task_progress:
        return {"status": "not_found"}
    return task_progress[session_id]

def run_export_background(session_id: str, export_mode: str, fps: float):
    """
    Background worker for stitching frames/masks into finished rotoscoped products.
    """
    try:
        task_progress[session_id]["status"] = "exporting"
        task_progress[session_id]["message"] = f"Stitching video in {export_mode} mode..."
        task_progress[session_id]["current"] = 50  # Mock percentage midpoint
        
        session_workspace = WORKSPACE_DIR / session_id
        frames_dir = session_workspace / "raw_frames"
        masks_dir = session_workspace / "masks"
        
        if export_mode == "zip":
            # ZIP PNG sequence export
            output_zip = OUTPUT_DIR / f"rotoscope_cutout_{session_id}.zip"
            utils.create_transparent_png_sequence_zip(frames_dir, masks_dir, output_zip)
        else:
            # Video compilation exports
            output_mp4 = OUTPUT_DIR / f"rotoscope_{export_mode}_{session_id}.mp4"
            utils.create_video_from_frames(
                frames_dir=frames_dir,
                mask_dir=masks_dir,
                output_path=output_mp4,
                fps=fps,
                export_mode=export_mode
            )
            
        task_progress[session_id]["status"] = "completed"
        task_progress[session_id]["message"] = "Export compiled successfully!"
        task_progress[session_id]["current"] = 100
        
    except Exception as e:
        print(f"Background export error: {e}")
        task_progress[session_id]["status"] = "failed"
        task_progress[session_id]["message"] = f"Export compilation failed: {str(e)}"

@app.post("/api/export")
async def trigger_export(req: ExportRequest, background_tasks: BackgroundTasks):
    """
    Triggers rendering the final rotoscoped output.
    """
    try:
        session_workspace = WORKSPACE_DIR / req.session_id
        if not session_workspace.exists():
            raise HTTPException(status_code=400, detail="Invalid session workspace")
            
        # Extract FPS from video path in UPLOAD_DIR
        video_path = UPLOAD_DIR / f"{req.session_id}.mp4"
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            fps = 30.0
            
        # Reset progress bar for export phase
        task_progress[req.session_id] = {
            "status": "exporting",
            "current": 10,
            "total": 100,
            "message": "Initializing video stitching..."
        }
        
        background_tasks.add_task(run_export_background, req.session_id, req.export_mode, fps)
        return {"status": "started"}
        
    except Exception as e:
        print(f"Export launch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{session_id}/{export_mode}")
async def download_export(session_id: str, export_mode: str):
    """
    Endpoint to retrieve the compiled media attachment.
    """
    if export_mode == "zip":
        file_path = OUTPUT_DIR / f"rotoscope_cutout_{session_id}.zip"
        filename = "rotoscope_transparent_sequence.zip"
    else:
        file_path = OUTPUT_DIR / f"rotoscope_{export_mode}_{session_id}.mp4"
        filename = f"rotoscope_{export_mode}.mp4"
        
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested export file not found.")
        
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    Cleans up workspace directories, uploads, and releases SAM 2 GPU state.
    """
    try:
        # Free SAM 2 memory
        sam_helper.cleanup_session_state(session_id)
        
        # Delete directories
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)
        
        temp_video = UPLOAD_DIR / f"{session_id}.mp4"
        if temp_video.exists():
            temp_video.unlink()
            
        # Clean export media files
        for f in OUTPUT_DIR.glob(f"*{session_id}*"):
            try:
                f.unlink()
            except Exception:
                pass
                
        # Clean global progress
        if session_id in task_progress:
            del task_progress[session_id]
            
        return {"status": "cleaned"}
        
    except Exception as e:
        print(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount Frontend static files
frontend_path = BASE_DIR / "frontend"
if not frontend_path.exists():
    frontend_path.mkdir(parents=True, exist_ok=True)
    
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
