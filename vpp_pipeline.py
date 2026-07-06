# --- CELL 2: THE HOLLYWOOD MASTER PIPELINE (LOSSLESS + POLYGON Z-DEPTH + CACHE FIX) ---
import torch
import cv2
import numpy as np
import os
import shutil
import subprocess
import time
from ultralytics import YOLO


# ==========================================
# 1. USER CONFIGURATION & VFX DASHBOARD
# ==========================================
VIDEO_FILENAME = "TrialVideo.mp4" 
OBJECT_FILENAME = "object.png" 
BOX = (900, 260, 1000, 360) 

# --- 🎬 THE VFX DASHBOARD 🎬 ---
OBJECT_BLUR = 5             # Match the blur of the background
COLOR_MATCH_STRENGTH = 0.6  # Tints the object to match street lighting
SHADOW_OPACITY = 0.4        # Lowered to 0.4 for realistic outdoor ambient shadow
NOISE_AMOUNT = 5            

MAX_FRAMES = 150 
AI_SCALE = 0.5   
# ==========================================

# 1. Create a clean temporary directory for Lossless PNG frames
FRAME_DIR = "lossless_frames"
if os.path.exists(FRAME_DIR):
    shutil.rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR)

# --- VFX FUNCTION 1: Dynamic Contrast & Lighting ---
def match_lighting_and_grain(object_img, bg_roi):
    if OBJECT_BLUR > 1:
        harmonized = cv2.GaussianBlur(object_img, (OBJECT_BLUR, OBJECT_BLUR), 0)
    else:
        harmonized = object_img.copy()
        
    bgr = harmonized[..., :3].astype(np.float32)
    alpha = (harmonized[..., 3] > 0).astype(np.uint8)
    
    if np.sum(alpha) > 0 and bg_roi.size > 0:
        # Shift overall color tint
        bg_mean = cv2.mean(bg_roi)[:3]
        obj_mean = cv2.mean(bgr, mask=alpha)[:3]
        for c in range(3):
            shift = bg_mean[c] - obj_mean[c]
            bgr[..., c] = np.clip(bgr[..., c] + (shift * COLOR_MATCH_STRENGTH), 0, 255)
            
        # DYNAMIC CONTRAST: Match the shadow depth of the street
        bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        obj_gray = cv2.cvtColor(harmonized[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, bg_std = cv2.meanStdDev(bg_gray)
        _, obj_std = cv2.meanStdDev(obj_gray, mask=alpha)
        
        if obj_std[0][0] > 0:
            contrast_ratio = (bg_std[0][0] / obj_std[0][0]) * 0.9 
            bgr = ((bgr - obj_mean) * contrast_ratio) + obj_mean
            bgr = np.clip(bgr, 0, 255)

    if NOISE_AMOUNT > 0:
        noise = np.random.normal(0, NOISE_AMOUNT, bgr.shape).astype(np.float32)
        bgr = np.clip(bgr + noise, 0, 255)
    
    harmonized[..., :3] = bgr.astype(np.uint8)
    return harmonized

# --- VFX FUNCTION 2: Contact Shadow ---
def draw_contact_shadow(background, x, y, w, h):
    shadow_overlay = background.copy()
    center_x = x + (w // 2)
    center_y = y + h - (h // 12)
    axes = (w // 2, h // 6) 
    
    cv2.ellipse(shadow_overlay, (center_x, center_y), axes, 0, 0, 360, (10, 10, 10), -1)
    shadow_overlay = cv2.GaussianBlur(shadow_overlay, (45, 45), 0)
    
    return cv2.addWeighted(shadow_overlay, SHADOW_OPACITY, background, 1.0 - SHADOW_OPACITY, 0)

# --- VFX FUNCTION 3: Pixel-Perfect Overlay ---
def overlay_transparent(background, overlay, x, y, occlusion_mask_full):
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + w < 0 or y + h < 0: return background
    clip_x_left, clip_y_top = max(0, -x), max(0, -y)
    clip_x_right, clip_y_bottom = min(w, bg_w - x), min(h, bg_h - y)
    
    overlay_cropped = overlay[clip_y_top:clip_y_bottom, clip_x_left:clip_x_right]
    target_x, target_y = max(0, x), max(0, y)
    
    bg_target = background[target_y:target_y + (clip_y_bottom-clip_y_top), 
                           target_x:target_x + (clip_x_right-clip_x_left)]

    overlay_harmonized = match_lighting_and_grain(overlay_cropped, bg_target)

    alpha_mask = overlay_harmonized[..., 3].astype(np.float32)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0) / 255.0
    
    occ_target = occlusion_mask_full[target_y:target_y + (clip_y_bottom-clip_y_top), 
                                     target_x:target_x + (clip_x_right-clip_x_left)]
    
    alpha_mask = alpha_mask * (1.0 - occ_target)
    alpha_inv = 1.0 - alpha_mask
    bgr = overlay_harmonized[..., :3]

    for c in range(3):
        bg_target[..., c] = (alpha_mask * bgr[..., c] + alpha_inv * bg_target[..., c])
    return background

# ==========================================
# MAIN SCRIPT EXECUTION
# ==========================================
print("1️⃣ Loading AI Trackers...")
predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda")
seg_model = YOLO("yolov8n-seg.pt") 

print("2️⃣ Processing Video Frames...")
cap = cv2.VideoCapture(VIDEO_FILENAME)
original_frames = []
ai_frames = []
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while len(original_frames) < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret: break
    original_frames.append(frame) 
    small_frame = cv2.resize(frame, (0, 0), fx=AI_SCALE, fy=AI_SCALE)
    ai_frames.append(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
cap.release()

video_tensor = torch.tensor(np.array(ai_frames)).permute(0, 3, 1, 2).unsqueeze(0).float().to("cuda") / 255.0

print("3️⃣ Tracking camera movement...")
grid_points = []
for x in np.linspace(BOX[0], BOX[2], 4):
    for y in np.linspace(BOX[1], BOX[3], 4):
        grid_points.append([0, x * AI_SCALE, y * AI_SCALE])

queries = torch.tensor(grid_points, dtype=torch.float32).unsqueeze(0).to("cuda")

with torch.no_grad():
    tracks, visibilities = predictor(video_tensor, queries=queries)
    tracks = tracks[0].cpu().numpy() / AI_SCALE
    visibilities = visibilities[0].cpu().numpy()

# Free up VRAM
del predictor
torch.cuda.empty_cache()

print("4️⃣ Compositing (Polygon Z-Depth & Lossless Output)...")
object_raw = cv2.imread(OBJECT_FILENAME, cv2.IMREAD_UNCHANGED)
native_h, native_w = object_raw.shape[:2]
box_w = BOX[2] - BOX[0]
scale_factor = box_w / native_w
new_w, new_h = int(native_w * scale_factor), int(native_h * scale_factor)
object_resized = cv2.resize(object_raw, (new_w, new_h))

initial_points = tracks[0]
last_dx, last_dy = 0, 0 

for t, frame in enumerate(original_frames):
    current_points = tracks[t]
    current_vis = visibilities[t]
    
    vis_mask = current_vis > 0.0 
    valid_current = current_points[vis_mask]
    valid_initial = initial_points[vis_mask]
    
    if len(valid_current) > 2: 
        shift = np.median(valid_current - valid_initial, axis=0)
        last_dx, last_dy = int(shift[0]), int(shift[1])
        
    current_x = BOX[0] + last_dx
    current_y = BOX[3] + last_dy - new_h 
    
    # Calculate exactly where the cone touches the ground
    cone_bottom_y = current_y + new_h
    
    # Run YOLO Masking
    results = seg_model(frame, classes=[0, 1, 2, 3, 5, 7], verbose=False)
    occlusion_mask_full = np.zeros((height, width), dtype=np.float32)
    
    if results[0].masks is not None:
        # THE POLYGON FIX: Loop through the exact 2D boundaries of every object
        for points in results[0].masks.xy:
            if len(points) > 0:
                # Find the absolute lowest pixel (the shoe/tire) of this object
                lowest_y = np.max(points[:, 1])
                
                # Z-DEPTH LOGIC: If their lowest pixel is lower than the cone's base, 
                # they are in front of the cone. Draw their mask!
                if lowest_y > (cone_bottom_y - 15): 
                    pts = np.int32([points])
                    cv2.fillPoly(occlusion_mask_full, pts, 1.0)
                    
        occlusion_mask_full = cv2.GaussianBlur(occlusion_mask_full, (7, 7), 0)

    # Render Frame
    frame_with_shadow = draw_contact_shadow(frame.copy(), current_x, current_y, new_w, new_h)
    final_frame = overlay_transparent(frame_with_shadow, object_resized, current_x, current_y, occlusion_mask_full)
        
    # THE QUALITY FIX: Save as uncompressed, lossless PNG
    cv2.imwrite(f"{FRAME_DIR}/frame_{t:04d}.png", final_frame)
    
    if t % 30 == 0:
        print(f"   ↳ Rendered {t}/{len(original_frames)} frames")

# ==========================================
# 5. THE CACHE FIX (Dynamic Naming)
# ==========================================
print("5️⃣ Stitching Cinema-Quality Video via FFmpeg...")

# Generate a unique filename using the current timestamp
timestamp = int(time.time())
output_filename = f"final_master_vfx_{timestamp}.mp4"

# Use CRF 12 (Visually Lossless) to preserve 100% of the original video's quality
cmd = f"ffmpeg -y -framerate {fps} -i {FRAME_DIR}/frame_%04d.png -c:v libx264 -crf 12 -preset slow -pix_fmt yuv420p {output_filename}"
subprocess.run(cmd, shell=True, check=True)

print(f"✅ DONE! Look in your sidebar for the new file: '{output_filename}'")
