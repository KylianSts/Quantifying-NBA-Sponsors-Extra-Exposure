import os
import cv2
import pandas as pd
import yt_dlp
import torch
from ultralytics import YOLO
from tqdm import tqdm
import threading
import queue
import time
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "Models/models_results/modelisation_v10/yolo11s-obb_fine_tuned_v10_1280/weights/best.pt"
INPUT_CSV = "Data/urls/game_highlight_urls.csv"
OUTPUT_CSV = "Data/exposure_and_game_info/exposure_results_normalized.csv" 
TEMP_DIR = "Data/temp_videos"

# Detection Settings
CONF_THRESH = 0.6
TARGET_FPS = 5
BALL_CLASS_NAME = "basketball"
INPUT_SIZE = 1024

# Distance calculation logic 
PERSISTENCE_WINDOW = TARGET_FPS 

# Performance Settings
BATCH_SIZE = 32
NUM_DOWNLOADERS = 1
DOWNLOAD_QUEUE_SIZE = 3
SAVE_EVERY = 1

# GPU Optimizations
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# ============================================================================
# HELPER: FILE MANAGEMENT
# ============================================================================

def load_existing_results(output_path):
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            if 'video_id' in df.columns:
                processed_video_ids = set(df['video_id'].unique())
                print(f"✓ Resuming: {len(processed_video_ids)} videos already processed.")
                return df, processed_video_ids
        except Exception as e:
            print(f"⚠ CSV reading error: {e}")
    return pd.DataFrame(), set()

save_lock = threading.Lock()

def save_results_incremental(new_results, output_path):
    if not new_results: return
    new_df = pd.DataFrame(new_results)
    with save_lock:
        try:
            mode = 'a' if os.path.exists(output_path) else 'w'
            header = not os.path.exists(output_path)
            new_df.to_csv(output_path, mode=mode, header=header, index=False)
            return True
        except Exception as e:
            print(f"\n[SAVE ERROR] {e}")
            return False

# ============================================================================
# WORKER: DOWNLOAD
# ============================================================================

def download_worker(task_queue, ready_queue, worker_id):
    os.makedirs(TEMP_DIR, exist_ok=True)
    ydl_opts = {
        'format': 'best[height<=480]/best',
        'quiet': True, 'no_warnings': True,
        'cookiesfrombrowser': ('firefox',),
        #'concurrent_fragment_downloads': 8,
    }

    while True:
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty: continue
        if task is None:
            ready_queue.put(None)
            break
            
        output_path = os.path.join(TEMP_DIR, f"{task['video_id']}.mp4")
        current_opts = ydl_opts.copy()
        current_opts['outtmpl'] = output_path
        
        try:
            if not os.path.exists(output_path):
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    ydl.download([task['url']])
            ready_queue.put({'path': output_path, 'meta': task})
        except Exception:
            pass # Ignore download errors to continue
        finally:
            task_queue.task_done()

# ============================================================================
# CORE: ANALYZER WITH "SMART REFERENCE" LOGIC
# ============================================================================

def process_video_batched(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {}, {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_interval = max(1, round(fps / TARGET_FPS))
    
    video_meta = {'fps': fps, 'skip_interval': skip_interval, 'width': width, 'height': height}
    
    # --- MEMORY STATE (SMART REFERENCE) ---
    # Initialize ball state for this video
    ball_state = {
        'last_center': None,  # Coordinates (x, y)
        'lost_count': 9999,   # Number of frames since loss
        'image_center': np.array([width / 2, height / 2], dtype=np.float32)
    }

    results = {}
    batch_frames = []
    frame_idx = 0

    while True:
        if frame_idx % skip_interval == 0:
            success, frame = cap.read()
            if not success: break
            batch_frames.append(frame)
        else:
            if not cap.grab(): break
        
        if len(batch_frames) >= BATCH_SIZE:
            # Pass ball_state so it gets updated frame by frame
            _process_batch(batch_frames, model, results, ball_state)
            batch_frames = []
        
        frame_idx += 1

    if batch_frames:
        _process_batch(batch_frames, model, results, ball_state)

    cap.release()
    return results, video_meta

def _process_batch(frames, model, results, ball_state):
    """
    Process a batch but apply sequential logic for ball memory.
    """
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        batch_results = model(frames, imgsz=INPUT_SIZE, conf=CONF_THRESH, verbose=False, half=True)
    
    # Iterate SEQUENTIALLY over batch results to maintain temporal coherence
    for res in batch_results:
        
        img_h, img_w = res.orig_shape
        img_diag = np.sqrt(img_w**2 + img_h**2)
        img_area = img_w * img_h

        # Retrieve YOLO data
        if res.obb is None: 
            # If no detection, just increment loss counter
            ball_state['lost_count'] += 1
            # No logos detected, so nothing to calculate for this frame
            continue

        classes = res.obb.cls.cpu().numpy().astype(int)
        boxes = res.obb.xyxyxyxy.cpu().numpy()
        confs = res.obb.conf.cpu().numpy()
        
        # Centers of all boxes
        box_centers = boxes.mean(axis=1).astype(np.float32)
        
        # --- 1. SMART BALL LOGIC (Inspired by provided code) ---
        
        # Find index of best ball (max confidence)
        ball_indices = [i for i, c in enumerate(classes) if model.names[c] == BALL_CLASS_NAME]
        
        current_ball_center = None
        if ball_indices:
            best_idx = ball_indices[np.argmax(confs[ball_indices])]
            current_ball_center = box_centers[best_idx]

        # Determine Reference Point (REF)
        ref_point = None
        
        if current_ball_center is not None:
            # CASE 1: Ball visible -> It's the reference
            ref_point = current_ball_center
            ball_state['last_center'] = current_ball_center
            ball_state['lost_count'] = 0
            
        elif ball_state['last_center'] is not None and ball_state['lost_count'] < PERSISTENCE_WINDOW:
            # CASE 2: Ball recently lost -> Use memory (Ghost Point)
            ref_point = ball_state['last_center']
            ball_state['lost_count'] += 1
            
        else:
            # CASE 3: Ball lost for long time -> Use image center
            ref_point = ball_state['image_center']
            ball_state['lost_count'] += 1

        # --- 2. STATISTICS CALCULATION ---

        areas_px = np.array([cv2.contourArea(box) for box in boxes])
        
        detected_in_this_frame = set()

        for i, cls_id in enumerate(classes):
            name = model.names[cls_id]
            
            if name == BALL_CLASS_NAME:
                continue
            
            if name not in results:
                results[name] = {
                    'frames': 0,
                    'norm_area_acc': 0.0,
                    'detections': 0,
                    'norm_dist_acc': 0.0,
                    'dist_samples': 0  # Counter for distance average
                }
            
            # Area (Always calculated)
            results[name]['norm_area_acc'] += float(areas_px[i] / img_area)
            results[name]['detections'] += 1
            
            # Distance (Calculated relative to "ref_point" determined above)
            # Since we ALWAYS have a ref_point (Ball, Memory, or Center), we always calculate.
            dist_px = np.linalg.norm(box_centers[i] - ref_point)
            norm_dist = dist_px / img_diag
            
            results[name]['norm_dist_acc'] += float(norm_dist)
            results[name]['dist_samples'] += 1
            
            detected_in_this_frame.add(name)
        
        for name in detected_in_this_frame:
            results[name]['frames'] += 1

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading on {device.upper()}...")
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(f" Model error: {e}")
        return

    # Load CSV
    existing_df, processed_ids = load_existing_results(OUTPUT_CSV)
    try:
        df = pd.read_csv(INPUT_CSV)
        video_tasks = [t for t in df.to_dict('records') if str(t['video_id']) not in processed_ids]
    except FileNotFoundError: return

    if not video_tasks:
        print("✓ Everything already processed.")
        return

    # Thread Pipeline
    task_queue = queue.Queue()
    ready_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    
    downloaders = []
    for i in range(NUM_DOWNLOADERS):
        t = threading.Thread(target=download_worker, args=(task_queue, ready_queue, i), daemon=True)
        t.start()
        downloaders.append(t)

    for task in video_tasks: task_queue.put(task)
    for _ in range(NUM_DOWNLOADERS): task_queue.put(None)

    # Processing loop
    pending_results = []
    processed_count = 0
    
    print(f"\nProcessing {len(video_tasks)} videos...")
    print(f"Distance Strategy: Ball -> Memory ({PERSISTENCE_WINDOW} frames) -> Center")
    print("="*60)

    with tqdm(total=len(video_tasks), unit="vid") as pbar:
        active_downloaders = NUM_DOWNLOADERS
        
        while processed_count < len(video_tasks):
            try:
                item = ready_queue.get(timeout=2)
            except queue.Empty:
                if active_downloaders == 0: break
                continue

            if item is None:
                active_downloaders -= 1
                continue

            path = item['path']
            meta = item['meta']
            
            try:
                metrics, vid_meta = process_video_batched(path, model)
                
                for class_name, stats in metrics.items():
                    # Final calculations
                    real_exposure = stats['frames'] * (vid_meta['skip_interval'] / vid_meta['fps']) if vid_meta['fps'] else 0
                    
                    avg_area = 0
                    if stats['detections'] > 0:
                        avg_area = (stats['norm_area_acc'] / stats['detections']) * 100
                    
                    # Distance average (now calculated on almost all frames)
                    avg_dist = 0
                    if stats['dist_samples'] > 0:
                        avg_dist = (stats['norm_dist_acc'] / stats['dist_samples']) * 100

                    pending_results.append({
                        'game_id': meta.get('game_id'),
                        'video_id': meta.get('video_id'),
                        'exposure_zone': class_name,
                        'exposure_time_seconds': round(real_exposure, 2),
                        'avg_area_pct': round(avg_area, 4),
                        'avg_dist_ref_pct': round(avg_dist, 2), # Renamed for clarity
                        'video_url': meta.get('url')
                    })

            except Exception as e:
                print(f"Error {meta.get('video_id')}: {e}")
            finally:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
                processed_count += 1
                pbar.update(1)
                
                if processed_count % SAVE_EVERY == 0:
                    if save_results_incremental(pending_results, OUTPUT_CSV):
                        pending_results = []

    if pending_results:
        save_results_incremental(pending_results, OUTPUT_CSV)

    print("\n✓ COMPLETED")

if __name__ == "__main__":
    main()