/* ==========================================================================
   SAM 2 ROTOSCOPER - CLIENT APPLICATION ENGINE (app.js)
   ========================================================================== */

// --- APPLICATION STATE ---
const state = {
    sessionId: null,
    width: 0,
    height: 0,
    fps: 30.0,
    frameCount: 0,
    currentFrame: 0,
    isPlaying: false,
    playInterval: null,
    
    // Prompts state: frameIdx -> { coords: [[x,y], ...], labels: [1, 0, ...] }
    prompts: {},
    activeMode: 'add', // 'add' (positive click) or 'remove' (negative click)
    
    // Undo/Redo stacks: frameIdx -> Array of prompt snapshots
    undoStack: {},
    redoStack: {},
    
    // Image cache to prevent flashing while scrubbing/playing
    imageCache: {},
    maskCache: {}
};

// --- DOM ELEMENTS ---
const elements = {
    dropZone: document.getElementById('drop-zone'),
    videoInput: document.getElementById('video-input'),
    browseBtn: document.getElementById('browse-btn'),
    videoDetails: document.getElementById('video-details'),
    videoName: document.getElementById('video-name'),
    videoDuration: document.getElementById('video-duration'),
    videoFrames: document.getElementById('video-frames'),
    
    interactionCard: document.getElementById('interaction-card'),
    modeAddBtn: document.getElementById('mode-add-btn'),
    modeRemoveBtn: document.getElementById('mode-remove-btn'),
    clearFramePrompts: document.getElementById('clear-frame-prompts'),
    clearAllPrompts: document.getElementById('clear-all-prompts'),
    
    propagationCard: document.getElementById('propagation-card'),
    propagateBtn: document.getElementById('propagate-btn'),
    progressContainer: document.getElementById('progress-container'),
    progressBar: document.getElementById('progress-bar'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressMessage: document.getElementById('progress-message'),
    
    canvasPlaceholder: document.getElementById('canvas-placeholder'),
    canvasWrapper: document.getElementById('canvas-wrapper'),
    editorCanvas: document.getElementById('editor-canvas'),
    workspaceLoader: document.getElementById('workspace-loader'),
    loaderText: document.getElementById('loader-text'),
    
    videoControls: document.getElementById('video-controls'),
    controlPrevBtn: document.getElementById('control-prev-btn'),
    controlPlayBtn: document.getElementById('control-play-btn'),
    controlNextBtn: document.getElementById('control-next-btn'),
    timelineSlider: document.getElementById('timeline-slider'),
    keyframeTicks: document.getElementById('keyframe-ticks'),
    currentFrameLbl: document.getElementById('current-frame-lbl'),
    totalFramesLbl: document.getElementById('total-frames-lbl'),
    
    infoResolution: document.getElementById('info-resolution'),
    infoFps: document.getElementById('info-fps'),
    keyframesCard: document.getElementById('keyframes-card'),
    keyframesList: document.getElementById('keyframes-list'),
    
    exportCard: document.getElementById('export-card'),
    exportModeSelect: document.getElementById('export-mode-select'),
    exportBtn: document.getElementById('export-btn'),
    exportProgressContainer: document.getElementById('export-progress-container'),
    exportProgressBar: document.getElementById('export-progress-bar'),
    exportProgressMessage: document.getElementById('export-progress-message'),
    downloadLink: document.getElementById('download-link'),
    
    statusBadge: document.getElementById('status-badge'),
    statusText: document.getElementById('status-text')
};

// Canvas 2D context
const ctx = elements.editorCanvas.getContext('2d');

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    setupUploadHandlers();
    setupInteractionHandlers();
    setupPlaybackHandlers();
    setupKeyboardNavigation();
    setupExportHandlers();
});

// --- NOTIFICATION UTILITY ---
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' : type === 'success' ? 'toast-success' : ''}`;
    
    let icon = '<i class="fa-solid fa-circle-info text-success"></i>';
    if (type === 'error') icon = '<i class="fa-solid fa-triangle-exclamation text-danger"></i>';
    else if (type === 'success') icon = '<i class="fa-solid fa-circle-check text-success"></i>';
    
    toast.innerHTML = `${icon}<span>${message}</span>`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function updateStatus(text, statusClass) {
    elements.statusText.textContent = text;
    elements.statusBadge.className = `badge status-badge ${statusClass}`;
}

// --- TASK 1: VIDEO UPLOAD HANDLERS ---
function setupUploadHandlers() {
    elements.browseBtn.addEventListener('click', () => elements.videoInput.click());
    elements.videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleVideoUpload(e.target.files[0]);
    });
    
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('dragover');
    });
    
    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('dragover');
    });
    
    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleVideoUpload(e.dataTransfer.files[0]);
    });
}

async function handleVideoUpload(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    // Set UI Loading State
    showLoader(true, "Uploading video and extracting frames...");
    updateStatus("Uploading...", "processing");
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to upload video");
        }
        
        const data = await response.json();
        
        // Initialize State with returned metadata
        state.sessionId = data.session_id;
        state.width = data.width;
        state.height = data.height;
        state.fps = data.fps;
        state.frameCount = data.frame_count;
        state.currentFrame = 0;
        state.prompts = {};
        state.undoStack = {};
        state.redoStack = {};
        
        // Clear UI keyframe ticks and items
        updateKeyframeIndicators();
        
        // Cache clear
        state.imageCache = {};
        state.maskCache = {};
        
        // Update Video Details Card
        elements.videoName.textContent = file.name;
        elements.videoDuration.textContent = `${(state.frameCount / state.fps).toFixed(1)}s`;
        elements.videoFrames.textContent = state.frameCount;
        elements.videoDetails.classList.remove('hidden');
        
        // Update Info Card
        elements.infoResolution.textContent = `${state.width} × ${state.height}`;
        elements.infoFps.textContent = `${state.fps.toFixed(1)} fps`;
        
        // Set up Timeline Slider
        elements.timelineSlider.max = state.frameCount - 1;
        elements.timelineSlider.value = 0;
        elements.totalFramesLbl.textContent = String(state.frameCount).padStart(3, '0');
        elements.currentFrameLbl.textContent = "000";
        
        // Enable workspace UI cards
        enableWorkspace(true);
        showLoader(false);
        showToast("Video uploaded and pre-processed successfully!", "success");
        updateStatus("Ready", "ready");
        
        // Display first frame
        await renderFrame(0);
        
    } catch (e) {
        showLoader(false);
        showToast(e.message, "error");
        updateStatus("Upload Failed", "idle");
        console.error("Upload error:", e);
    }
}

function showLoader(show, text = "") {
    if (show) {
        elements.loaderText.textContent = text;
        elements.workspaceLoader.classList.remove('hidden');
    } else {
        elements.workspaceLoader.classList.add('hidden');
    }
}

function enableWorkspace(enable) {
    if (enable) {
        elements.canvasPlaceholder.classList.add('hidden');
        elements.canvasWrapper.classList.remove('hidden');
        elements.interactionCard.classList.remove('disabled');
        elements.propagationCard.classList.remove('disabled');
        elements.videoControls.classList.remove('disabled');
        elements.keyframesCard.classList.remove('disabled');
        elements.exportCard.classList.remove('disabled');
    } else {
        elements.canvasPlaceholder.classList.remove('hidden');
        elements.canvasWrapper.classList.add('hidden');
        elements.interactionCard.classList.add('disabled');
        elements.propagationCard.classList.add('disabled');
        elements.videoControls.classList.add('disabled');
        elements.keyframesCard.classList.add('disabled');
        elements.exportCard.classList.add('disabled');
    }
}

// --- CANVAS RENDERING AND IMAGE PRELOADING ---
async function preloadFrameImage(frameIdx) {
    if (state.imageCache[frameIdx]) return state.imageCache[frameIdx];
    
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            state.imageCache[frameIdx] = img;
            resolve(img);
        };
        img.onerror = () => reject(new Error("Failed to load frame image"));
        img.src = `/api/frames/${state.sessionId}/${frameIdx}`;
    });
}

async function preloadMaskImage(frameIdx) {
    // Check cache
    if (state.maskCache[frameIdx] !== undefined) return state.maskCache[frameIdx];
    
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            state.maskCache[frameIdx] = img;
            resolve(img);
        };
        img.onerror = () => {
            // A 404 indicates no mask has been created for this frame, which is expected
            state.maskCache[frameIdx] = null;
            resolve(null);
        };
        // Add cache bust string
        img.src = `/api/masks/${state.sessionId}/${frameIdx}?t=${new Date().getTime()}`;
    });
}

async function renderFrame(frameIdx) {
    try {
        // Load image and mask concurrently
        const [img, mask] = await Promise.all([
            preloadFrameImage(frameIdx),
            preloadMaskImage(frameIdx)
        ]);
        
        // Match canvas dimensions to the video resolution
        elements.editorCanvas.width = state.width;
        elements.editorCanvas.height = state.height;
        
        // Draw raw video frame
        ctx.drawImage(img, 0, 0, state.width, state.height);
        
        // Overlay mask if available
        if (mask) {
            // Draw transparent colored mask using standard source-over compositing
            ctx.save();
            ctx.globalAlpha = 0.5; // 50% opacity
            
            // Create a temporary canvas to colorize the grayscale mask
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = state.width;
            tempCanvas.height = state.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Draw original mask on temp canvas
            tempCtx.drawImage(mask, 0, 0);
            
            // Colorize: source-in fills non-transparent pixels with our neon green/mint color
            tempCtx.globalCompositeOperation = 'source-in';
            tempCtx.fillStyle = '#00ff87'; // Bright Neon Mint Green
            tempCtx.fillRect(0, 0, state.width, state.height);
            
            // Draw colorized mask onto main editor canvas
            ctx.drawImage(tempCanvas, 0, 0);
            ctx.restore();
        }
        
        // Draw interactive prompt dots
        const framePrompts = state.prompts[frameIdx];
        if (framePrompts && framePrompts.coords.length > 0) {
            framePrompts.coords.forEach((coord, i) => {
                const label = framePrompts.labels[i];
                const x = coord[0];
                const y = coord[1];
                
                ctx.save();
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2 * Math.PI);
                
                if (label === 1) {
                    ctx.fillStyle = '#00ff87'; // Positive click
                    ctx.strokeStyle = '#ffffff';
                    ctx.shadowColor = '#00ff87';
                } else {
                    ctx.fillStyle = '#ff0055'; // Negative click
                    ctx.strokeStyle = '#ffffff';
                    ctx.shadowColor = '#ff0055';
                }
                
                ctx.shadowBlur = 8;
                ctx.lineWidth = 1.5;
                ctx.fill();
                ctx.stroke();
                ctx.restore();
            });
        }
        
        // Update Timeline display
        elements.currentFrameLbl.textContent = String(frameIdx).padStart(3, '0');
        elements.timelineSlider.value = frameIdx;
        
    } catch (e) {
        console.error("Frame render error:", e);
    }
}

// --- TASK 3: INTERACTIVE CANVAS CLICKS ---
function setupInteractionHandlers() {
    // Mode toggles
    elements.modeAddBtn.addEventListener('click', () => {
        state.activeMode = 'add';
        elements.modeAddBtn.classList.add('active');
        elements.modeRemoveBtn.classList.remove('active');
    });
    
    elements.modeRemoveBtn.addEventListener('click', () => {
        state.activeMode = 'remove';
        elements.modeRemoveBtn.classList.add('active');
        elements.modeAddBtn.classList.remove('active');
    });
    
    // Canvas click detection
    elements.editorCanvas.addEventListener('click', handleCanvasClick);
    
    // Disable right-click menu on canvas to allow easy subtract clicks
    elements.editorCanvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        // Trigger right-click as subtraction prompt directly
        handleCanvasClick(e, true);
    });
    
    // Clear buttons
    elements.clearFramePrompts.addEventListener('click', clearCurrentFramePrompts);
    elements.clearAllPrompts.addEventListener('click', resetAllSessionPrompts);
}

async function handleCanvasClick(e, forceSubtract = false) {
    if (!state.sessionId) return;
    
    // Get mouse position relative to canvas layout box
    const rect = elements.editorCanvas.getBoundingClientRect();
    const scaleX = state.width / rect.width;
    const scaleY = state.height / rect.height;
    
    // Compute pixel coordinate in original video resolution
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;
    
    // Clamp to boundaries
    const originalX = Math.max(0, Math.min(state.width - 1, Math.round(canvasX)));
    const originalY = Math.max(0, Math.min(state.height - 1, Math.round(canvasY)));
    
    // Determine prompt label (1 = positive click, 0 = negative/subtract click)
    // Left-click with subtract active OR right-click triggers subtract
    const isSubtract = forceSubtract || state.activeMode === 'remove' || e.button === 2;
    const label = isSubtract ? 0 : 1;
    
    const frameIdx = state.currentFrame;
    
    // Save state for undo before modifying
    saveUndoState(frameIdx);
    
    // Append prompt to our active frame state
    if (!state.prompts[frameIdx]) {
        state.prompts[frameIdx] = { coords: [], labels: [] };
    }
    
    state.prompts[frameIdx].coords.push([originalX, originalY]);
    state.prompts[frameIdx].labels.push(label);
    
    // Send full coordinate set of the current frame to backend for interactive refine
    showLoader(true, "Refining segment selection...");
    updateStatus("Predicting...", "processing");
    
    try {
        const response = await fetch('/api/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                frame_idx: frameIdx,
                obj_id: 1, // Single object tracking
                coords: state.prompts[frameIdx].coords,
                labels: state.prompts[frameIdx].labels
            })
        });
        
        if (!response.ok) {
            throw new Error("Selection refinement failed");
        }
        
        // Remove specific frame mask cache to force download reload
        delete state.maskCache[frameIdx];
        
        // Redraw canvas
        await renderFrame(frameIdx);
        
        // Re-draw keyframe timeline marks
        updateKeyframeIndicators();
        
        showLoader(false);
        updateStatus("Ready", "ready");
        
    } catch (err) {
        showLoader(false);
        updateStatus("Prediction Failed", "ready");
        showToast(err.message, "error");
        
        // Revert local prompt entry on error
        state.prompts[frameIdx].coords.pop();
        state.prompts[frameIdx].labels.pop();
        
        // Also pop from undo stack so it matches
        if (state.undoStack[frameIdx]) {
            state.undoStack[frameIdx].pop();
        }
    }
}

async function clearCurrentFramePrompts() {
    const frameIdx = state.currentFrame;
    if (!state.prompts[frameIdx]) return;
    
    // Save undo state before clearing
    saveUndoState(frameIdx);
    
    delete state.prompts[frameIdx];
    delete state.maskCache[frameIdx];
    
    updateKeyframeIndicators();
    showLoader(true, "Clearing frame prompts...");
    
    try {
        // Send a delete session/re-initialize click request or clear state on server
        // The simplest way to clear prompts on a frame in SAM 2 is to clear clicks locally
        // and tell the server there are no clicks.
        const response = await fetch('/api/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                frame_idx: frameIdx,
                obj_id: 1,
                coords: [],
                labels: []
            })
        });
        
        if (!response.ok) throw new Error("Failed to clear prompts on frame");
        
        await renderFrame(frameIdx);
        showLoader(false);
        showToast("Frame prompts cleared.", "success");
        
    } catch (e) {
        showLoader(false);
        showToast(e.message, "error");
    }
}

// --- UNDO AND REDO HELPER FUNCTIONS ---
function saveUndoState(frameIdx) {
    if (!state.undoStack[frameIdx]) {
        state.undoStack[frameIdx] = [];
    }
    
    // Create a deep copy of the current prompts for this frame
    const currentPrompts = state.prompts[frameIdx] 
        ? { 
            coords: state.prompts[frameIdx].coords.map(c => [...c]), 
            labels: [...state.prompts[frameIdx].labels] 
          } 
        : null;
        
    state.undoStack[frameIdx].push(currentPrompts);
    
    // Limit stack size to 50 edits
    if (state.undoStack[frameIdx].length > 50) {
        state.undoStack[frameIdx].shift();
    }
    
    // Clear redo stack when a new action is performed
    state.redoStack[frameIdx] = [];
}

async function undoAction() {
    const frameIdx = state.currentFrame;
    if (!state.undoStack[frameIdx] || state.undoStack[frameIdx].length === 0) {
        showToast("Nothing to undo on this frame", "info");
        return;
    }
    
    showToast("Undoing last prompt...", "success");
    
    if (!state.redoStack[frameIdx]) {
        state.redoStack[frameIdx] = [];
    }
    
    // Save current state to redo stack
    const currentPrompts = state.prompts[frameIdx]
        ? {
            coords: state.prompts[frameIdx].coords.map(c => [...c]),
            labels: [...state.prompts[frameIdx].labels]
          }
        : null;
    state.redoStack[frameIdx].push(currentPrompts);
    
    // Restore previous state from undo stack
    const previousPrompts = state.undoStack[frameIdx].pop();
    if (previousPrompts) {
        state.prompts[frameIdx] = previousPrompts;
    } else {
        delete state.prompts[frameIdx];
    }
    
    // Send updated state to backend
    await syncPromptsWithBackend(frameIdx);
}

async function redoAction() {
    const frameIdx = state.currentFrame;
    if (!state.redoStack[frameIdx] || state.redoStack[frameIdx].length === 0) {
        showToast("Nothing to redo on this frame", "info");
        return;
    }
    
    showToast("Redoing last prompt...", "success");
    
    if (!state.undoStack[frameIdx]) {
        state.undoStack[frameIdx] = [];
    }
    
    // Save current state to undo stack
    const currentPrompts = state.prompts[frameIdx]
        ? {
            coords: state.prompts[frameIdx].coords.map(c => [...c]),
            labels: [...state.prompts[frameIdx].labels]
          }
        : null;
    state.undoStack[frameIdx].push(currentPrompts);
    
    // Restore state from redo stack
    const nextPrompts = state.redoStack[frameIdx].pop();
    if (nextPrompts) {
        state.prompts[frameIdx] = nextPrompts;
    } else {
        delete state.prompts[frameIdx];
    }
    
    // Send updated state to backend
    await syncPromptsWithBackend(frameIdx);
}

async function syncPromptsWithBackend(frameIdx) {
    if (!state.sessionId) return;
    
    showLoader(true, "Updating segment selection...");
    updateStatus("Predicting...", "processing");
    
    const framePrompts = state.prompts[frameIdx];
    const coords = framePrompts ? framePrompts.coords : [];
    const labels = framePrompts ? framePrompts.labels : [];
    
    try {
        const response = await fetch('/api/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                frame_idx: frameIdx,
                obj_id: 1,
                coords: coords,
                labels: labels
            })
        });
        
        if (!response.ok) {
            throw new Error("Failed to update prompts");
        }
        
        // Remove specific frame mask cache to force download reload
        delete state.maskCache[frameIdx];
        
        // Redraw canvas
        await renderFrame(frameIdx);
        
        // Re-draw keyframe timeline marks
        updateKeyframeIndicators();
        
        showLoader(false);
        updateStatus("Ready", "ready");
        
    } catch (err) {
        showLoader(false);
        updateStatus("Prediction Failed", "ready");
        showToast(err.message, "error");
    }
}

async function resetAllSessionPrompts() {
    if (!confirm("Are you sure you want to delete ALL click selections across all frames and start fresh?")) return;
    
    showLoader(true, "Resetting tracking session...");
    updateStatus("Resetting...", "processing");
    
    try {
        // Standard clean reset is accomplished by calling DELETE on session
        if (state.sessionId) {
            const response = await fetch(`/api/session/${state.sessionId}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error("Failed to reset session on server");
        }
        
        // Reset local state variables
        state.sessionId = null;
        state.width = 0;
        state.height = 0;
        state.fps = 30.0;
        state.frameCount = 0;
        state.currentFrame = 0;
        state.prompts = {};
        state.undoStack = {};
        state.redoStack = {};
        state.maskCache = {};
        state.imageCache = {};
        
        // Reset file input value to allow re-uploading the same file
        elements.videoInput.value = "";
        
        // Hide details card and reset values
        elements.videoDetails.classList.add('hidden');
        elements.videoName.textContent = "-";
        elements.videoDuration.textContent = "-";
        elements.videoFrames.textContent = "-";
        
        // Reset timeline slider
        elements.timelineSlider.value = 0;
        elements.timelineSlider.max = 100;
        elements.currentFrameLbl.textContent = "000";
        elements.totalFramesLbl.textContent = "000";
        
        // Reset info card values
        elements.infoResolution.textContent = "-";
        elements.infoFps.textContent = "-";
        
        // Clear timeline markers and keyframe badges
        updateKeyframeIndicators();
        
        showLoader(false);
        showToast("Session reset successfully! You can now upload a video to start fresh.", "success");
        enableWorkspace(false);
        updateStatus("Idle", "idle");
        
    } catch (e) {
        showLoader(false);
        showToast("Reset failed: " + e.message, "error");
    }
}

function updateKeyframeIndicators() {
    // 1. Draw timeline marks
    elements.keyframeTicks.innerHTML = '';
    const ticksContainerWidth = elements.keyframeTicks.getBoundingClientRect().width;
    
    // 2. Draw lists
    elements.keyframesList.innerHTML = '';
    
    const activeFrameIndices = Object.keys(state.prompts).map(Number).sort((a,b)=>a-b);
    
    if (activeFrameIndices.length === 0) {
        elements.keyframesList.innerHTML = '<div class="no-keyframes">No prompts placed yet</div>';
        return;
    }
    
    activeFrameIndices.forEach(idx => {
        // Draw tick mark on timeline
        const percent = (idx / (state.frameCount - 1)) * 100;
        const tick = document.createElement('div');
        tick.className = 'keyframe-tick';
        tick.style.left = `${percent}%`;
        elements.keyframeTicks.appendChild(tick);
        
        // Add badge list item
        const badge = document.createElement('div');
        badge.className = 'keyframe-badge';
        badge.innerHTML = `<i class="fa-solid fa-key"></i> Frame ${idx}`;
        badge.addEventListener('click', () => {
            state.currentFrame = idx;
            renderFrame(idx);
        });
        elements.keyframesList.appendChild(badge);
    });
}

// --- TIMELINE PLAYBACK AND SCRUBBING ---
function setupPlaybackHandlers() {
    // Scrubbing on timeline
    elements.timelineSlider.addEventListener('input', (e) => {
        if (state.isPlaying) stopPlayback();
        state.currentFrame = parseInt(e.target.value);
        renderFrame(state.currentFrame);
    });
    
    // Video buttons
    elements.controlPlayBtn.addEventListener('click', togglePlayback);
    elements.controlPrevBtn.addEventListener('click', stepPrevFrame);
    elements.controlNextBtn.addEventListener('click', stepNextFrame);
}

function togglePlayback() {
    if (state.isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!state.sessionId) return;
    state.isPlaying = true;
    elements.controlPlayBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
    elements.controlPlayBtn.classList.add('playing');
    
    // Interval based on FPS
    const intervalMs = 1000 / state.fps;
    state.playInterval = setInterval(() => {
        state.currentFrame++;
        if (state.currentFrame >= state.frameCount) {
            state.currentFrame = 0; // Loop around
        }
        renderFrame(state.currentFrame);
    }, intervalMs);
}

function stopPlayback() {
    state.isPlaying = false;
    elements.controlPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    elements.controlPlayBtn.classList.remove('playing');
    if (state.playInterval) {
        clearInterval(state.playInterval);
        state.playInterval = null;
    }
}

function stepPrevFrame() {
    if (state.isPlaying) stopPlayback();
    state.currentFrame = Math.max(0, state.currentFrame - 1);
    renderFrame(state.currentFrame);
}

function stepNextFrame() {
    if (state.isPlaying) stopPlayback();
    state.currentFrame = Math.min(state.frameCount - 1, state.currentFrame + 1);
    renderFrame(state.currentFrame);
}

function setupKeyboardNavigation() {
    window.addEventListener('keydown', (e) => {
        if (!state.sessionId) return;
        
        // Skip hotkeys if focused on form elements
        if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {
            return;
        }
        
        // Detect Ctrl+Z and Ctrl+Y / Ctrl+Shift+Z for undo/redo
        if (e.ctrlKey || e.metaKey) {
            const isZ = e.key === 'z' || e.key === 'Z' || e.code === 'KeyZ' || e.keyCode === 90;
            const isY = e.key === 'y' || e.key === 'Y' || e.code === 'KeyY' || e.keyCode === 89;
            
            if (e.shiftKey && isZ) {
                e.preventDefault();
                e.stopPropagation();
                console.log("Redo triggered via Ctrl+Shift+Z");
                redoAction();
                return;
            } else if (isZ) {
                e.preventDefault();
                e.stopPropagation();
                console.log("Undo triggered via Ctrl+Z");
                undoAction();
                return;
            } else if (isY) {
                e.preventDefault();
                e.stopPropagation();
                console.log("Redo triggered via Ctrl+Y");
                redoAction();
                return;
            }
        }
        
        if (e.key === ' ') {
            e.preventDefault();
            togglePlayback();
        } else if (e.key === 'ArrowLeft' || e.key === 'a') {
            stepPrevFrame();
        } else if (e.key === 'ArrowRight' || e.key === 'd') {
            stepNextFrame();
        } else if (e.key === 'Escape') {
            clearCurrentFramePrompts();
        }
    });
}

// --- TASK 4: PROPAGATION AND EXPORT CHANNELS ---
document.getElementById('propagate-btn').addEventListener('click', runMaskPropagation);

async function runMaskPropagation() {
    if (!state.sessionId) return;
    
    // Verify there is at least one prompt frame
    const hasPrompts = Object.keys(state.prompts).length > 0;
    if (!hasPrompts) {
        showToast("Please place at least one click prompt on the video before propagating tracking!", "error");
        return;
    }
    
    if (state.isPlaying) stopPlayback();
    
    elements.propagateBtn.disabled = true;
    elements.progressContainer.classList.remove('hidden');
    elements.progressBar.style.width = '0%';
    elements.progressPercentage.textContent = '0%';
    elements.progressMessage.textContent = 'Initializing propagation...';
    
    updateStatus("Propagating...", "processing");
    
    try {
        const response = await fetch('/api/propagate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId })
        });
        
        if (!response.ok) throw new Error("Propagation start failed");
        
        // Begin progress polling loop
        pollPropagationProgress();
        
    } catch (e) {
        elements.propagateBtn.disabled = false;
        elements.progressContainer.classList.add('hidden');
        updateStatus("Prop Failed", "ready");
        showToast(e.message, "error");
    }
}

function pollPropagationProgress() {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/progress/${state.sessionId}`);
            if (!response.ok) return;
            
            const progress = await response.json();
            
            if (progress.status === "propagating" || progress.status === "exporting") {
                const percent = Math.round((progress.current / progress.total) * 100);
                elements.progressBar.style.width = `${percent}%`;
                elements.progressPercentage.textContent = `${percent}%`;
                elements.progressMessage.textContent = progress.message;
            } 
            else if (progress.status === "completed") {
                clearInterval(pollInterval);
                elements.progressBar.style.width = '100%';
                elements.progressPercentage.textContent = '100%';
                elements.progressMessage.textContent = progress.message;
                
                // Clear maskCache to force reload propagated frames
                state.maskCache = {};
                
                // Re-render current frame with its newly propagated mask!
                await renderFrame(state.currentFrame);
                
                showToast("Mask propagation complete across all frames!", "success");
                updateStatus("Ready", "ready");
                elements.propagateBtn.disabled = false;
            } 
            else if (progress.status === "failed") {
                clearInterval(pollInterval);
                elements.progressContainer.classList.add('hidden');
                elements.propagateBtn.disabled = false;
                updateStatus("Prop Failed", "ready");
                showToast(progress.message, "error");
            }
            
        } catch (e) {
            console.error("Progress polling error:", e);
        }
    }, 500);
}

function setupExportHandlers() {
    elements.exportBtn.addEventListener('click', runVideoExport);
}

async function runVideoExport() {
    if (!state.sessionId) return;
    
    elements.exportBtn.disabled = true;
    elements.exportProgressContainer.classList.remove('hidden');
    elements.exportProgressBar.style.width = '15%';
    elements.exportProgressMessage.textContent = "Launching render task...";
    elements.downloadLink.classList.add('hidden');
    
    updateStatus("Exporting...", "processing");
    const mode = elements.exportModeSelect.value;
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                export_mode: mode
            })
        });
        
        if (!response.ok) throw new Error("Video export failed to launch");
        
        pollExportProgress(mode);
        
    } catch (e) {
        elements.exportBtn.disabled = false;
        elements.exportProgressContainer.classList.add('hidden');
        updateStatus("Ready", "ready");
        showToast(e.message, "error");
    }
}

function pollExportProgress(mode) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/progress/${state.sessionId}`);
            if (!response.ok) return;
            
            const progress = await response.json();
            
            if (progress.status === "exporting") {
                elements.exportProgressBar.style.width = '50%';
                elements.exportProgressMessage.textContent = progress.message;
            } 
            else if (progress.status === "completed") {
                clearInterval(pollInterval);
                elements.exportProgressBar.style.width = '100%';
                elements.exportProgressMessage.textContent = "Composition ready!";
                
                // Configure Download Button
                elements.downloadLink.href = `/api/download/${state.sessionId}/${mode}`;
                elements.downloadLink.classList.remove('hidden');
                
                showToast("Composition finished! Download is ready.", "success");
                updateStatus("Ready", "ready");
                elements.exportBtn.disabled = false;
            } 
            else if (progress.status === "failed") {
                clearInterval(pollInterval);
                elements.exportProgressContainer.classList.add('hidden');
                elements.exportBtn.disabled = false;
                updateStatus("Export Failed", "ready");
                showToast(progress.message, "error");
            }
            
        } catch (e) {
            console.error("Export progress polling error:", e);
        }
    }, 500);
}
