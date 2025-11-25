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


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "Models/models_results/modelisation_v6/yolo11s-obb_fine_tuned_v6/weights/best.pt"
INPUT_CSV = "Data/urls/game_highlight_urls.csv"
OUTPUT_CSV = "Data/exposure_and_game_info/exposure_results_final.csv"
TEMP_DIR = "Data/temp_videos"

# Detection Settings
CONF_THRESH = 0.75
TARGET_FPS = 5

BATCH_SIZE = 16
DOWNLOAD_QUEUE_SIZE = 10

# Save frequency (number of videos)
SAVE_EVERY = 3

# ============================================================================
# HELPER: LOAD EXISTING RESULTS
# ============================================================================

def load_existing_results(output_path):
    """
    Load existing results CSV and return both the dataframe and a set of processed video_ids.
    """
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            processed_video_ids = set(df['video_id'].unique())
            print(f"Found existing results with {len(processed_video_ids)} already processed videos.")
            return df, processed_video_ids
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()

# ============================================================================
# HELPER: SAVE RESULTS INCREMENTALLY
# ============================================================================

def save_results_incremental(new_results, output_path):
    """
    Append new results to the CSV file (or create if doesn't exist).
    """
    if not new_results:
        return
    
    new_df = pd.DataFrame(new_results)
    
    try:
        if os.path.exists(output_path):
            # Append to existing file
            new_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            # Create new file with header
            new_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"\n[SAVE ERROR] Could not save results: {e}")

# ============================================================================
# WORKER: VIDEO DOWNLOADER (Runs in separate thread)
# ============================================================================

def download_worker(task_queue, ready_queue):
    """
    Constantly pulls URLs from task_queue, downloads them, 
    and puts the file path into ready_queue.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Fast download options
    ydl_opts = {
        #'format': 'bestvideo[ext=mp4][height<=720]/best[ext=mp4]',
        'format': 'best[height<=720]/best',
        'quiet': True,
        'no_warnings': True,
        #'extractor_args': {'youtube': {'player_client': ['android']}},
        'cookiesfrombrowser': ('firefox',),
        'concurrent_fragment_downloads': 8
    }


    while True:
        task = task_queue.get()
        if task is None: # Sentinel signal to stop
            ready_queue.put(None)
            break
            
        video_id = task['video_id']
        url = task['url']
        output_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")
        
        # Configure output for this specific file
        current_opts = ydl_opts.copy()
        current_opts['outtmpl'] = output_path
        
        try:
            # Only download if not exists (resuming support)
            if not os.path.exists(output_path):
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    ydl.download([url])
            
            # Send to Analyzer
            ready_queue.put({'path': output_path, 'meta': task})
            
        except Exception as e:
            print(f"\n[DOWNLOAD ERROR] {video_id}: {e}")
            # Even if failed, we mark task as done so pipeline continues
        
        task_queue.task_done()

# ============================================================================
# CORE: BATCH ANALYZER (Runs on Main Thread/GPU)
# ============================================================================

def process_video_batched(video_path, model):
    """
    Reads video, aggregates frames into batches, runs inference, returns stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {}

    # Video Props
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_interval = max(1, round(fps / TARGET_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stats Storage
    results = {} 
    
    # Batch Storage
    batch_frames = []
    
    # Progress Bar
    pbar = tqdm(total=total_frames, desc="  GPU Inference", leave=False, unit="frames")
    
    frame_idx = 0
    while True:
        # Optimized Reading: Grab is faster than Read if we skip
        if frame_idx % skip_interval == 0:
            success, frame = cap.read()
            if not success: break
            batch_frames.append(frame)
        else:
            if not cap.grab(): break # End of video
            
        # If Batch is full or End of Video, Run Inference
        if len(batch_frames) == BATCH_SIZE or (not success and len(batch_frames) > 0):
            if batch_frames:
                # RUN INFERENCE ON BATCH (Massive Speedup)
                # verbose=False prevents console spam
                batch_results = model(batch_frames, conf=CONF_THRESH, verbose=False)
                
                # Process Batch Results
                for res in batch_results:
                    if res.obb is not None:
                        classes = res.obb.cls.cpu().numpy().astype(int)
                        boxes = res.obb.xyxyxyxy.cpu().numpy()
                        
                        # Use set to ensure we count 1 frame per class even if multiple logos appear
                        detected_in_frame = set()
                        
                        for cls_id, box in zip(classes, boxes):
                            name = model.names[cls_id]
                            area = cv2.contourArea(box)
                            
                            if name not in results: 
                                results[name] = {'frames': 0, 'area': 0, 'detections': 0}
                            
                            results[name]['area'] += area
                            results[name]['detections'] += 1
                            detected_in_frame.add(name)
                        
                        for name in detected_in_frame:
                            results[name]['frames'] += 1
                
                # Clear batch
                batch_frames = []
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()

    # 1. Load Model (Force GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing YOLO on {device.upper()} (RTX Optimization Enabled)...")
    if device == 'cpu': print("WARNING: GPU not detected. Processing will be slow.")
    
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load existing results and get already processed video IDs
    existing_df, processed_video_ids = load_existing_results(OUTPUT_CSV)

    # 3. Prepare Queues
    task_queue = queue.Queue()
    ready_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE) 

    # 4. Load Data and filter out already processed videos
    try:
        df = pd.read_csv(INPUT_CSV)
        all_tasks = df.to_dict('records')
        
        # Filter out already processed videos
        video_tasks = [task for task in all_tasks if task['video_id'] not in processed_video_ids]
        
        if len(video_tasks) < len(all_tasks):
            skipped = len(all_tasks) - len(video_tasks)
            print(f"Skipping {skipped} already processed videos.")
        
        if len(video_tasks) == 0:
            print("All videos have already been processed!")
            return
            
    except FileNotFoundError:
        print("Input CSV not found.")
        return

    # 5. Fill Task Queue
    print(f"Queuing {len(video_tasks)} videos for processing...")
    for task in video_tasks:
        task_queue.put(task)
    # Add sentinel for the downloader
    task_queue.put(None) 

    # 6. Start Download Thread
    downloader = threading.Thread(target=download_worker, args=(task_queue, ready_queue))
    downloader.daemon = True
    downloader.start()

    # 7. Main Loop (Consumer)
    pending_results = []  # Buffer for results before saving
    processed_count = 0
    total_processed_overall = len(processed_video_ids)  # Track total including previously processed
    
    print("\n" + "="*50)
    print("STARTING PARALLEL PIPELINE")
    print(f"Batch Size: {BATCH_SIZE} | Target FPS: {TARGET_FPS}")
    print(f"Save Frequency: Every {SAVE_EVERY} videos")
    print("="*50)

    with tqdm(total=len(video_tasks), desc="Total Progress") as main_pbar:
        while processed_count < len(video_tasks):
            # Get next downloaded video (blocks if download is slower than GPU)
            item = ready_queue.get()
            
            if item is None: # Sentinel
                break
                
            path = item['path']
            meta = item['meta']
            
            # ANALYZE
            main_pbar.set_description(f"Processing: {meta['video_id']}")
            
            try:
                metrics = process_video_batched(path, model)
                
                # Format Results
                for class_name, stats in metrics.items():
                    pending_results.append({
                        'game_id': meta.get('game_id'),
                        'video_id': meta.get('video_id'),
                        'exposure_zone': class_name,
                        'exposure_time_seconds': round(stats['frames'] / TARGET_FPS, 2),
                        'average_box_area_pixels': round(stats['area'] / stats['detections'], 2),
                        'video_url': meta.get('url')
                    })
                    
            except Exception as e:
                print(f"\nError analyzing {meta['video_id']}: {e}")
            finally:
                # CLEANUP FILE IMMEDIATELY
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except: pass
                
                processed_count += 1
                total_processed_overall += 1
                main_pbar.update(1)
                
                # SAVE INCREMENTALLY
                if processed_count % SAVE_EVERY == 0 or processed_count == len(video_tasks):
                    if pending_results:
                        save_results_incremental(pending_results, OUTPUT_CSV)
                        main_pbar.set_description(f"✓ Saved {len(pending_results)} results ({total_processed_overall} total)")
                        pending_results = []  # Clear buffer after saving

    # 8. Final save for any remaining results
    if pending_results:
        save_results_incremental(pending_results, OUTPUT_CSV)
        print(f"\n✓ Saved final {len(pending_results)} results")

    print(f"\nDone! All results saved to {OUTPUT_CSV}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format time nicely
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print("\n" + "="*50)
    print("EXECUTION TIME")
    print("="*50)
    if hours > 0:
        print(f"Total time: {hours}h {minutes}m {seconds:.2f}s")
    elif minutes > 0:
        print(f"Total time: {minutes}m {seconds:.2f}s")
    else:
        print(f"Total time: {seconds:.2f}s")
    print(f"Videos processed (this session): {processed_count}")
    print(f"Total videos in results: {total_processed_overall}")
    if processed_count > 0:
        print(f"Average time per video: {elapsed_time/processed_count:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()