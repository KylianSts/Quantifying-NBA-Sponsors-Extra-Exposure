"""
Quantifies Sponsor Exposure in YouTube Videos (Sequential & Simple Version)

This script processes a list of YouTube video URLs one by one, quantifies
sponsor exposure by analyzing frames sequentially at a target FPS, and
aggregates the results into a single output CSV file.
"""

import os
import pandas as pd
import cv2
import yt_dlp
import torch
from ultralytics import YOLO
from tqdm import tqdm
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "Models/models_results/modelisation_v6/yolov8s-obb_fine_tuned_v6/weights/best.pt"
URLS_INPUT_CSV = "Data/urls/game_highlight_urls.csv"
OUTPUT_CSV_PATH = "Data/exposure_results_final.csv"
TEMP_VIDEO_DIR = "temp_videos"

CONFIDENCE_THRESHOLD = 0.75
TARGET_FPS = 5

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def download_video(url: str, output_dir: str) -> Tuple[str, Dict]:
    """Downloads a single video from YouTube and returns its path and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'extractor_args': {'youtube': {'player_client': ['android']}},
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        metadata = {
            'video_id': info.get('id', 'N/A'), 'title': info.get('title', 'N/A'), 'url': url,
            'view_count': info.get('view_count', 0), 'like_count': info.get('like_count', 0),
            'comment_count': info.get('comment_count', 0), 'duration_seconds': info.get('duration', 0),
        }
        return filepath, metadata

def analyze_video_frames(video_path: str, model: YOLO, target_fps: int) -> Dict[str, Dict]:
    """
    Processes a video frame by frame sequentially to quantify sponsor exposure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(original_fps / target_fps) if original_fps > 0 and original_fps > target_fps else 1
    
    class_names = list(model.names.values())
    exposure_data = {name: {'frame_count': 0, 'total_area': 0, 'detection_count': 0} for name in class_names}

    frames_to_analyze_count = total_frames // frame_skip
    pbar = tqdm(total=frames_to_analyze_count, desc=f"  Analyzing {os.path.basename(video_path)}", leave=False)

    for frame_index in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        
        if frame_index % frame_skip != 0:
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        
        detected_classes_in_frame = set()
        if result.obb is not None:
            for i in range(len(result.obb)):
                class_id = int(result.obb.cls[i].item())
                class_name = model.names[class_id]
                detected_classes_in_frame.add(class_name)
                
                points = result.obb.xyxyxyxy[i].cpu().numpy()
                area = cv2.contourArea(points)
                exposure_data[class_name]['total_area'] += area
                exposure_data[class_name]['detection_count'] += 1
        
        for class_name in detected_classes_in_frame:
            exposure_data[class_name]['frame_count'] += 1
        
        pbar.update(1)

    pbar.close()
    cap.release()

    final_metrics = {}
    for class_name, data in exposure_data.items():
        exposure_time_sec = data['frame_count'] / target_fps if target_fps > 0 else 0
        avg_area = data['total_area'] / data['detection_count'] if data['detection_count'] > 0 else 0
        final_metrics[class_name] = {'exposure_time_seconds': round(exposure_time_sec, 2), 'average_box_area_pixels': round(avg_area, 2)}
        
    return final_metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main sequential pipeline for batch processing."""
    print("="*60)
    print("Quantify Sponsor Exposure - Sequential Version")
    print("="*60)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device.upper()}")
        model = YOLO(MODEL_PATH)
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load model: {e}")
        return
        
    try:
        urls_df = pd.read_csv(URLS_INPUT_CSV)
        video_tasks = urls_df.to_dict('records')
        print(f"Found {len(video_tasks)} videos to process from '{URLS_INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"[FATAL ERROR] Input file not found: '{URLS_INPUT_CSV}'.")
        return

    all_results = []
    print("\nStarting video processing one by one...")
    
    for task in tqdm(video_tasks, desc="Overall Progress"):
        video_url = task['url']
        print(f"\nProcessing video: {video_url}")
        video_path = None
        try:
            video_path, metadata = download_video(video_url, TEMP_VIDEO_DIR)
            exposure_metrics = analyze_video_frames(video_path, model, TARGET_FPS)

            for class_name, metrics in exposure_metrics.items():
                row = {
                    'game_id': task['game_id'], 'video_id': metadata['video_id'],
                    'exposure_zone': class_name, 'exposure_time_seconds': metrics['exposure_time_seconds'],
                    'average_box_area_pixels': metrics['average_box_area_pixels'],
                    'view_count': metadata['view_count'], 'like_count': metadata['like_count'],
                    'comment_count': metadata['comment_count'], 'duration_seconds': metadata['duration_seconds'],
                    'video_title': metadata['title'], 'video_url': metadata['url']
                }
                all_results.append(row)
            
            print(f"  > Finished processing: {os.path.basename(video_path)}")

        except Exception as e:
            print(f"  > [ERROR] Failed to process video {video_url}: {e}")
        
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols_order = ['game_id', 'video_id', 'exposure_zone', 'exposure_time_seconds', 'average_box_area_pixels', 'view_count', 'like_count', 'comment_count', 'video_duration_seconds', 'video_title', 'video_url']
        results_df = results_df.reindex(columns=cols_order)
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        # Afficher un résumé plus clair
        num_videos_processed = len(results_df['video_id'].unique())
        print(f"\n{'='*60}")
        print(f"✅ Results for {num_videos_processed} videos saved to '{OUTPUT_CSV_PATH}'")
        print(f"{'='*60}")
    else:
        print("\nNo videos were successfully processed. No output file was created.")

if __name__ == "__main__":
    main()