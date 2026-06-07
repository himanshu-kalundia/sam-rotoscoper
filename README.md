# SAM Rotoscoper 🪄✨
> Next-Generation Interactive Video Segmentation Powered by Meta SAM 2.1 Large

SAM Rotoscoper is a premium, web-based interactive video segmentation tool. It leverages Meta's **Segment Anything Model 2.1 (SAM 2.1 Hiera Large)** to track and segment subjects in videos with professional precision. This application is optimized for Windows with NVIDIA CUDA acceleration (bfloat16 enabled) and provides a clean, responsive dark-themed workspace.

---

## 🌟 Key Features

* **Interactive Point Prompts**: Left-click to include your subject (positive prompt), right-click to exclude backgrounds or incorrect selections (negative prompt).
* **Temporal Tracking**: Propagate your clicks backward and forward across the entire video. SAM 2 utilizes video memory to predict the mask on adjacent frames.
* **Multiple Output Render Modes**:
  * **Green Mint Mask Overlay (MP4)**: Semi-transparent visual overlay of the segmented mask on top of the original video.
  * **Chroma Key / Green Screen (MP4)**: Background replaced with solid green, perfect for compositing in editing programs.
  * **Cutout on Pure Black (MP4)**: Excludes the background entirely, leaving only the subject over a black background.
  * **Transparent PNG Sequence (ZIP)**: Exports a ZIP archive of high-quality transparent PNGs (BGRA), ideal for professional post-production software like DaVinci Resolve, Adobe After Effects, and Premiere Pro.
* **GPU Accelerated (RTX 4060 / CUDA)**: Implemented with PyTorch and CUDA 12.4 utilizing `bfloat16` precision for fast inference.
* **Interactive Dark Mode Workspace**: Outfitted with a custom canvas editor, a frame timeline slider, keyframe indicators, status badges, and keyboard shortcuts.

---

## 📂 Project Structure

```
sam-rotoscoper/
├── backend/
│   ├── config.py         # App configurations, path resolution & port settings
│   ├── main.py           # FastAPI server endpoints (Upload, Click, Propagate, Export, Clean)
│   ├── sam_helper.py     # SAM 2 predictor inference state and GPU memory manager
│   └── utils.py          # OpenCV frame extraction and video compiler utils
├── frontend/
│   ├── index.html        # Main app UI structure
│   ├── style.css         # Custom dark glassmorphism styling
│   └── app.js            # Interactive Canvas logic, API requests & player controls
├── checkpoints/          # Stores the downloaded SAM 2.1 Large model weights
├── sam2_repo/            # Meta's Segment Anything 2 library
├── install.ps1           # Full automated Windows installation script
├── test_sam2.py          # Diagnostic script to verify CUDA & SAM 2 model loading
└── requirements.txt      # Python dependencies (FastAPI, OpenCV, etc.)
```

---

## ⚙️ Prerequisites

* **OS**: Windows 10/11
* **GPU**: NVIDIA GPU (e.g. RTX 4060 or higher) with [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads) installed.
* **Python**: Python 3.10+ (ensure it is added to your System PATH)
* **Git**: Installed and available in terminal

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   Clone the repository recursively to automatically fetch the `sam2_repo` submodule:
   ```powershell
   git clone --recursive https://github.com/himanshu-kalundia/sam-rotoscoper.git
   cd sam-rotoscoper
   ```
   *Note: If you have already cloned the repository without `--recursive`, run `git submodule update --init --recursive` to pull the submodule.*

2. **Run the Installer**:
   Open a PowerShell window in the project root folder and execute the installation script. This script automatically sets up the Python virtual environment (`venv`), installs PyTorch with CUDA 12.4 support, initializes and pulls the Meta SAM 2 submodule, installs additional dependencies, and downloads the Hiera Large weights.
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process
   .\install.ps1
   ```

3. **Verify Installation**:
   Confirm PyTorch has CUDA access and that SAM 2 initializes correctly by running the diagnostic test script:
   ```powershell
   .\venv\Scripts\python.exe test_sam2.py
   ```

---

## 🏃 Running the Application

1. **Start the FastAPI Backend**:
   Launch the server using the virtual environment's `uvicorn` runner:
   ```powershell
   .\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the Web App**:
   Open your browser and navigate to `http://localhost:8000`.

---

## 🎬 How to Use (Workflow)

1. **Upload Video**:
   Drag and drop or browse to select a video (up to 10 seconds long, preferably in `.mp4`, `.mov`, or `.webm`). The backend will extract individual frames and prepare the tracking session.
2. **Add Prompts (Keyframes)**:
   * Navigate to the frame where your object is most visible.
   * **Left-Click** on the object to add green selection points (to include the subject).
   * **Right-Click** on any incorrectly selected region to add red selection points (to exclude).
   * Keyframe tick marks will appear on the timeline slider indicating where prompts have been placed.
3. **Propagate Tracking**:
   Click **Propagate Tracking** on the control panel. The model will track the segmented subject backward and forward across the entire video.
4. **Review & Refine**:
   Play back the video to review the result. If the tracking slips on some frames, navigate to that frame, add additional correction prompts, and run **Propagate Tracking** again.
5. **Render and Export**:
   Select your preferred **Render Mode** (e.g., Chroma Key or Transparent PNG Sequence) and click **Render Output**. Once completed, click the **Download Rotoscope** button to download the finalized assets.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Space` | Play / Pause video playback |
| `Right Arrow` | Step forward 1 frame |
| `Left Arrow` | Step backward 1 frame |
| `Left-Click` | Add positive prompt (Include area) |
| `Right-Click` | Add negative prompt (Exclude area) |

---

## 🔒 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
The Segment Anything 2 core library is licensed under the Apache 2.0 License.