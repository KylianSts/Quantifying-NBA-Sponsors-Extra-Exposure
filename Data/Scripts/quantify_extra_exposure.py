"""
Brand Exposure Analysis System for Basketball Videos

This system processes basketball game highlight videos to detect and measure
brand logo exposure using computer vision and deep learning techniques.

Key Features:
- Automated video downloading from URLs
- YOLO-based object detection (OBB - Oriented Bounding Boxes)
- Multi-threaded video processing
- Advanced quality scoring v combining attention, size, clutter, and legibility
- Financial Valuation: Calculates Media Value based on View Count and QI
- Incremental CSV output with fault tolerance
"""

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
import sys
import math
import logging
from typing import Dict, List, Tuple, Optional, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and file paths
MODEL_PATH = "Models/models_results/modelisation_v10/yolo11s-obb_fine_tuned_v10_1280/weights/best.pt"
INPUT_CSV = "Data/urls/game_highlight_urls_2025_26.csv"
OUTPUT_CSV = "Data/exposure_and_game_info/exposure_results_2025_26_test.csv"
TEMP_DIR = "Data/temp_videos"

# Detection parameters
CONF_THRESH = 0.6              # Minimum confidence threshold for detections
TARGET_FPS = 5                 # Target frame rate for processing
BALL_CLASS_NAME = "basketball" # Class name for basketball detection
INPUT_SIZE = 1024              # Input size for YOLO model

# Valuation parameters
CPV_REF = 0.00033              # Cost Per View reference
FRAME_DURATION = 1.0 / TARGET_FPS # Duration of one frame (0.2s)

# Tracking and fusion parameters
PERSISTENCE_WINDOW = TARGET_FPS * 1.5  # Frames to persist ball location
MERGE_IOU_THRESH = 0.10                # IoU threshold for merging overlapping detections
MERGE_CONTAINMENT_THRESH = 0.50        # Containment threshold for merging

# Performance parameters
BATCH_SIZE = 32           # Number of frames to process in batch
NUM_DOWNLOADERS = 3       # Number of concurrent download threads
DOWNLOAD_QUEUE_SIZE = 6   # Maximum videos in download queue
SAVE_EVERY = 5             # Save results every N videos

# Timeout parameters
DOWNLOAD_TIMEOUT = 600      # Timeout for video download (seconds)
VIDEO_PROCESS_TIMEOUT = 3000  # Timeout for video processing (seconds)
MAX_DOWNLOAD_RETRIES = 1   # Maximum download retry attempts

# GPU optimization
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Data/exposure_and_game_info/brand_exposure_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FILE INITIALIZATION
# ============================================================================

def initialize_files() -> None:
    """
    Initialize required files and directories for the analysis pipeline.
    """
    logger.info("Checking and creating required files...")
    
    # Create directories
    for path in [INPUT_CSV, OUTPUT_CSV]:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    # Create temp directory for videos
    os.makedirs(TEMP_DIR, exist_ok=True)
            
    # Check/create input CSV
    if not os.path.exists(INPUT_CSV):
        logger.warning(f"Input file not found: {INPUT_CSV}")
        # Added view_count to template
        df_template = pd.DataFrame(columns=['video_id', 'url', 'game_id', 'view_count'])
        df_template.to_csv(INPUT_CSV, index=False)
        logger.info(f"Template file created. Please populate it with video data.")
        sys.exit(0)

    # Create output CSV with all metric columns
    if not os.path.exists(OUTPUT_CSV):
        logger.info(f"Creating results file: {OUTPUT_CSV}")
        cols = [
            'game_id', 'video_id', 'exposure_zone', 
            'exposure_seconds', 'total_detections',
            
            # Financial Metric
            'total_media_value',                        
            
            # Scientific metrics
            'qi_score_avg', 'qi_score_std',
            
            # QI Score components
            'size_score_avg', 'size_score_std',         # Sigmoid size score
            'sov_weighted_avg', 'sov_weighted_std',     # Weighted share of voice
            'dist_score_avg', 'dist_score_std',         # Gaussian attention score
            'legi_score_avg', 'legi_score_std',         # Legibility score

            # Raw metrics for context
            'conf_avg', 'conf_std',                     # Model confidence
            'laplacian_avg', 'laplacian_std',           # Sharpness raw
            'dist_raw_pct_avg', 'dist_raw_pct_std',     # Raw distance to ball (%)
            'area_pct_avg', 'area_pct_std',             # Raw size (% of screen)
            
            'video_url'
        ]
        pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)
    else:
        logger.info("Results file found.")

    # Validate model exists
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_stats(sum_val: float, sum_sq_val: float, n: int) -> Tuple[float, float]:
    """Calculate mean and standard deviation from accumulated statistics."""
    if n == 0:
        return 0.0, 0.0
    
    mean = sum_val / n
    variance = (sum_sq_val / n) - (mean ** 2)
    std = math.sqrt(max(0.0, variance))
    
    return mean, std


def load_existing_results(output_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """Load previously processed results to avoid reprocessing."""
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            if 'video_id' in df.columns:
                processed = set(df['video_id'].astype(str).unique())
                logger.info(f"Found {len(processed)} previously processed videos.")
                return df, processed
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    return pd.DataFrame(), set()


# Thread-safe saving
save_lock = threading.Lock()

def save_results_incremental(new_results: List[Dict], output_path: str) -> bool:
    """Save results incrementally to CSV file in a thread-safe manner."""
    if not new_results:
        return True
        
    new_df = pd.DataFrame(new_results)
    
    with save_lock:
        try:
            file_exists = os.path.exists(output_path)
            header = not file_exists or os.stat(output_path).st_size == 0
            mode = 'a' if file_exists else 'w'
            new_df.to_csv(output_path, mode=mode, header=header, index=False)
            logger.debug(f"Saved {len(new_results)} results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

# ============================================================================
# DETECTION MERGING LOGIC
# ============================================================================

def should_merge(box1: np.ndarray, box2: np.ndarray) -> bool:
    """Determine if two bounding boxes should be merged based on IoU and containment."""
    # Get bounding rectangles
    x_min1, y_min1 = box1.min(axis=0)
    x_max1, y_max1 = box1.max(axis=0)
    x_min2, y_min2 = box2.min(axis=0)
    x_max2, y_max2 = box2.max(axis=0)
    
    # Check if boxes overlap
    if (x_max1 < x_min2 or x_max2 < x_min1 or 
        y_max1 < y_min2 or y_max2 < y_min1):
        return False
    
    # Calculate areas
    area1 = cv2.contourArea(box1)
    area2 = cv2.contourArea(box2)
    
    if area1 == 0 or area2 == 0:
        return False
    
    # Calculate intersection
    try:
        inter, _ = cv2.intersectConvexConvex(
            box1.astype(np.float32), 
            box2.astype(np.float32)
        )
    except:
        return False
    
    if inter <= 0:
        return False
    
    # Calculate union
    union = area1 + area2 - inter
    if union <= 0:
        return False
    
    # Check IoU threshold
    iou = inter / union
    if iou > MERGE_IOU_THRESH:
        return True
    
    # Check containment threshold
    containment = inter / min(area1, area2)
    if containment > MERGE_CONTAINMENT_THRESH:
        return True
    
    return False


def consolidate_detections(
    boxes: np.ndarray, 
    classes: np.ndarray, 
    confs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consolidate overlapping detections of the same class using graph-based merging."""
    if len(boxes) == 0:
        return boxes, classes, confs
    
    final_boxes = []
    final_cls = []
    final_conf = []
    
    # Process each class separately
    for cls in np.unique(classes):
        idxs = np.where(classes == cls)[0]
        cls_boxes = boxes[idxs]
        cls_confs = confs[idxs]
        
        n = len(idxs)
        
        # Build adjacency list for overlapping boxes
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge(cls_boxes[i], cls_boxes[j]):
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find connected components using DFS
        visited = [False] * n
        
        for i in range(n):
            if not visited[i]:
                # DFS to find all connected boxes
                stack = [i]
                visited[i] = True
                group = []
                
                while stack:
                    curr = stack.pop()
                    group.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                
                # Merge all boxes in group
                all_pts = np.vstack(cls_boxes[group]).astype(np.float32)
                rect = cv2.minAreaRect(all_pts)
                merged_box = cv2.boxPoints(rect).astype(int)
                
                final_boxes.append(merged_box)
                final_cls.append(cls)
                final_conf.append(np.max(cls_confs[group]))
    
    return np.array(final_boxes), np.array(final_cls), np.array(final_conf)

# ============================================================================
# QUALITY METRICS CALCULATION (V3)
# ============================================================================

def calculate_laplacian_variance(img_crop: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    """
    try:
        if img_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0.0


def sigmoid_size_score_final(ratio: float) -> float:
    """
    Calculate size score using UPDATED parameters for V3.
    
    New parameters:
    - k (slope): 100
    - x0 (center): 0.015 (1.5% of screen area)
    
    Args:
        ratio: Logo area as fraction of total screen area (0-1)
        
    Returns:
        Size score from 0 to 1
    """
    k = 100       # Modified slope for V3
    x0 = 0.015    # Modified center (1.5%) for V3
    
    score = 1 / (1 + np.exp(-k * (ratio - x0)))
    
    # Absolute noise threshold
    if ratio < 0.0015:  # < 0.15% considered invisible
        return 0.0
        
    return score


def sigmoid_legibility_score(variance: float) -> float:
    """
    Calculate legibility score based on Laplacian variance (sharpness).
    
    Parameters:
    - k (slope): 0.05
    - x0 (center): 100.0 (Variance threshold for sharp images)
    
    Args:
        variance: The raw Laplacian variance score
        
    Returns:
        Legibility score from 0 to 1
    """
    k = 0.05
    x0 = 100.0
    
    score = 1 / (1 + np.exp(-k * (variance - x0)))
    return score


def calculate_weighted_share_of_voice(
    target_area: float,
    target_center: np.ndarray,
    all_boxes_data: List[Dict],
    diag_px: float
) -> float:
    """
    Calculate weighted Share of Voice considering spatial crowding effect.
    """
    weighted_competitor_area = 0.0
    sigma_crowd = 0.10 
    
    for box_data in all_boxes_data:
        dist_px = np.linalg.norm(target_center - box_data['center'])
        
        # Skip self (distance < 1px)
        if dist_px < 1.0:
            continue
            
        dist_norm = dist_px / diag_px
        
        # Gaussian weight
        weight = np.exp(-(dist_norm**2) / (2 * sigma_crowd**2))
        
        weighted_competitor_area += (box_data['area'] * weight)
    
    total_effective_area = target_area + weighted_competitor_area
    
    if total_effective_area == 0:
        return 0.0
    
    sov = target_area / total_effective_area
    
    # Square root smoothing
    return np.sqrt(sov)


def calculate_final_scores(
    logo_box: np.ndarray,
    logo_center: np.ndarray,
    focal_point: np.ndarray,
    diag_px: float,
    total_area_px: float,
    all_boxes_data: List[Dict],
    laplacian_val: float
) -> Dict[str, float]:
    """
    Calculate comprehensive quality scores 
    """
    # Raw measurements
    dist_px = np.linalg.norm(logo_center - focal_point)
    dist_raw_pct = dist_px / diag_px 
    area = cv2.contourArea(logo_box)
    ratio = area / total_area_px

    # 1. Attention Score (Gaussian with sigma=0.20)
    s_attn = np.exp(-(dist_raw_pct**2) / (2 * 0.20**2))
    
    # 2. Size Score (Updated Sigmoid)
    s_size = sigmoid_size_score_final(ratio)
    
    # 3. Clutter Score (Weighted Share of Voice)
    s_sov = calculate_weighted_share_of_voice(
        area, logo_center, all_boxes_data, diag_px
    )

    # 4. Legibility Score (Sigmoid on Laplacian)
    s_legi = sigmoid_legibility_score(laplacian_val)
    
    # Final QI Score (product of 4 components)
    qi = s_attn * s_size * s_sov * s_legi

    return {
        'qi_score': min(qi, 1.0),
        's_attn': s_attn,
        's_size': s_size,
        's_sov': s_sov,
        's_legi': s_legi,
        'dist_raw_pct': dist_raw_pct
    }

# ============================================================================
# VIDEO DOWNLOAD WORKER
# ============================================================================

def download_worker(
    task_queue: queue.Queue,
    ready_queue: queue.Queue,
    worker_id: int
) -> None:
    """Worker thread for downloading videos from URLs."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # yt-dlp configuration
    ydl_opts = {
        'format': 'best[height<=720]/best',
        'quiet': True,
        'no_warnings': True,
        'cookiesfrombrowser': ('firefox',),
        'socket_timeout': DOWNLOAD_TIMEOUT,
        'retries': MAX_DOWNLOAD_RETRIES
    }
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # Poison pill to stop worker
        if task is None:
            ready_queue.put(None)
            break
        
        video_id = task.get('video_id', 'unknown')
        output_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")
        
        # Skip if already downloaded
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.debug(f"Video {video_id} already exists, skipping download")
            ready_queue.put({'path': output_path, 'meta': task})
            task_queue.task_done()
            continue
        
        # Attempt download with retry logic
        current_opts = ydl_opts.copy()
        current_opts['outtmpl'] = output_path
        
        success = False
        last_error = None
        
        for attempt in range(MAX_DOWNLOAD_RETRIES):
            try:
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    ydl.download([task['url']])
                
                # Verify download
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    ready_queue.put({'path': output_path, 'meta': task})
                    success = True
                    logger.debug(f"Successfully downloaded video {video_id}")
                    break
                else:
                    raise Exception("Downloaded file is empty or corrupted")
                    
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Download attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES} "
                    f"failed for video {video_id}: {str(e)}"
                )
                time.sleep(2 ** attempt)  # Exponential backoff
        
        if not success:
            logger.error(
                f"Failed to download video {video_id} after "
                f"{MAX_DOWNLOAD_RETRIES} attempts. Last error: {last_error}"
            )
            # Put placeholder in ready queue to maintain count
            ready_queue.put({'path': None, 'meta': task, 'error': str(last_error)})
        
        task_queue.task_done()

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_video_batched(
    video_path: str,
    model: YOLO,
    view_count: int
) -> Tuple[Dict, Dict]:
    """Process a video file to detect and analyze brand logo exposure."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}, {}

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    diag_px = np.sqrt(width**2 + height**2)
    total_area_px = width * height
    skip_interval = max(1, round(fps / TARGET_FPS))
    
    video_meta = {
        'fps': fps,
        'skip_interval': skip_interval,
        'width': width,
        'height': height,
        'total_frames': total_frames
    }
    
    # Ball tracking state
    ball_state = {
        'last_center': None,
        'lost_count': 9999,
        'image_center': np.array([width / 2, height / 2], dtype=np.float32)
    }

    results = {}
    batch_frames = []
    frame_idx = 0
    
    try:
        while True:
            # Read frame if it's a target frame
            if frame_idx % skip_interval == 0:
                success, frame = cap.read()
                if not success:
                    break
                batch_frames.append(frame)
            else:
                # Skip frame without decoding
                if not cap.grab():
                    break
            
            # Process batch when full
            if len(batch_frames) >= BATCH_SIZE:
                _process_batch(
                    batch_frames, model, results, ball_state,
                    diag_px, total_area_px, view_count
                )
                batch_frames = []
            
            frame_idx += 1
        
        # Process remaining frames
        if batch_frames:
            _process_batch(
                batch_frames, model, results, ball_state,
                diag_px, total_area_px, view_count
            )
            
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
    finally:
        cap.release()
    
    return results, video_meta


def _process_batch(
    frames: List[np.ndarray],
    model: YOLO,
    results: Dict,
    ball_state: Dict,
    diag_px: float,
    total_area_px: float,
    view_count: int
) -> None:
    """Process a batch of frames through the detection model and calculate value."""
    # Run inference with automatic mixed precision
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        batch_results = model(
            frames,
            imgsz=INPUT_SIZE,
            conf=CONF_THRESH,
            verbose=False,
            half=True
        )
    
    for i, res in enumerate(batch_results):
        # Skip if no detections
        if res.obb is None:
            ball_state['lost_count'] += 1
            continue
        
        # Extract raw detections
        raw_classes = res.obb.cls.cpu().numpy().astype(int)
        raw_boxes = res.obb.xyxyxyxy.cpu().numpy().astype(int)
        raw_confs = res.obb.conf.cpu().numpy()
        
        # Consolidate overlapping detections
        boxes, classes, confs = consolidate_detections(
            raw_boxes, raw_classes, raw_confs
        )
        
        if len(boxes) == 0:
            ball_state['lost_count'] += 1
            continue
        
        # Calculate box centers
        box_centers = boxes.mean(axis=1).astype(np.float32)

        # Determine focal point (basketball position with persistence)
        ball_indices = [
            idx for idx, c in enumerate(classes)
            if model.names[c] == BALL_CLASS_NAME
        ]
        
        current_ball_center = None
        if ball_indices:
            # Use highest confidence ball detection
            best_idx = ball_indices[np.argmax(confs[ball_indices])]
            current_ball_center = box_centers[best_idx]
        
        # Focal point logic with persistence
        if current_ball_center is not None:
            focal_point = current_ball_center
            ball_state['last_center'] = current_ball_center
            ball_state['lost_count'] = 0
        elif (ball_state['last_center'] is not None and
              ball_state['lost_count'] < PERSISTENCE_WINDOW):
            focal_point = ball_state['last_center']
            ball_state['lost_count'] += 1
        else:
            focal_point = ball_state['image_center']
            ball_state['lost_count'] += 1

        # Prepare neighbor data for Share of Voice calculation
        all_boxes_data = []
        for k, b in enumerate(boxes):
            if model.names[classes[k]] != BALL_CLASS_NAME:
                all_boxes_data.append({
                    'area': cv2.contourArea(b),
                    'center': box_centers[k]
                })

        # Track which brands detected in this frame
        detected_in_this_frame = set()
        current_frame_img = frames[i]

        # Process each detection
        for j, cls_id in enumerate(classes):
            name = model.names[cls_id]
            
            # Skip basketball detections
            if name == BALL_CLASS_NAME:
                continue
            
            # Initialize brand entry if needed
            if name not in results:
                results[name] = {
                    'frames': 0,
                    'detections': 0,
                    'media_value_sum': 0.0, 
                    
                    'qi_score_acc': 0.0, 'qi_score_sq': 0.0,
                    'size_score_acc': 0.0, 'size_score_sq': 0.0,
                    'sov_weighted_acc': 0.0, 'sov_weighted_sq': 0.0,
                    'dist_score_acc': 0.0, 'dist_score_sq': 0.0,
                    'legi_score_acc': 0.0, 'legi_score_sq': 0.0,
                    
                    'conf_acc': 0.0, 'conf_sq': 0.0,
                    'laplacian_acc': 0.0, 'laplacian_sq': 0.0,
                    'dist_raw_pct_acc': 0.0, 'dist_raw_pct_sq': 0.0,
                    'area_pct_acc': 0.0, 'area_pct_sq': 0.0
                }
            
            # --- 1. CALCULATE LAPLACIAN ---
            x_coords = boxes[j][:, 0]
            y_coords = boxes[j][:, 1]
            x_min = max(0, int(min(x_coords)))
            x_max = min(current_frame_img.shape[1], int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(current_frame_img.shape[0], int(max(y_coords)))
            
            laplacian_val = 0.0
            if x_max > x_min and y_max > y_min:
                crop = current_frame_img[y_min:y_max, x_min:x_max]
                laplacian_val = calculate_laplacian_variance(crop)

            # --- 2. CALCULATE FINAL SCORES ---
            scores = calculate_final_scores(
                boxes[j], box_centers[j], focal_point,
                diag_px, total_area_px, all_boxes_data,
                laplacian_val
            )
            
            # --- 3. CALCULATE INSTANT MEDIA VALUE ---
            # Formula: V_{i,t} = N_{views} * CPV_{ref} * (1/5) * QI_{i,t}
            instant_media_value = view_count * CPV_REF * FRAME_DURATION * scores['qi_score']
            results[name]['media_value_sum'] += instant_media_value

            # Calculate percentage metrics
            area_val_pct = (cv2.contourArea(boxes[j]) / total_area_px) * 100
            dist_raw_val_pct = scores['dist_raw_pct'] * 100

            # Accumulate statistics
            def update(key: str, val: float) -> None:
                results[name][f'{key}_acc'] += val
                results[name][f'{key}_sq'] += (val ** 2)

            # Scientific metrics
            update('qi_score', scores['qi_score'])
            update('size_score', scores['s_size'])
            update('sov_weighted', scores['s_sov'])
            update('dist_score', scores['s_attn'])
            update('legi_score', scores['s_legi'])
            
            # Raw metrics
            update('conf', confs[j])
            update('laplacian', laplacian_val)
            update('dist_raw_pct', dist_raw_val_pct)
            update('area_pct', area_val_pct)
            
            results[name]['detections'] += 1
            detected_in_this_frame.add(name)
        
        # Count frames for each detected brand
        for name in detected_in_this_frame:
            results[name]['frames'] += 1

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main() -> None:
    """Main entry point for the brand exposure analysis pipeline."""
    # Initialize environment
    initialize_files()
    logger.info("=== Starting Brand Exposure Analysis ===")
    
    # Load YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading model on {device.upper()}...")
    
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    # Load existing results
    existing_df, processed = load_existing_results(OUTPUT_CSV)
    
    # Load input tasks
    try:
        df = pd.read_csv(INPUT_CSV)
        df['video_id'] = df['video_id'].astype(str)
        tasks = [
            task for task in df.to_dict('records')
            if task['video_id'] not in processed
        ]
    except Exception as e:
        logger.error(f"Error loading input CSV: {e}")
        return

    if not tasks:
        logger.info("No new videos to process.")
        return

    # Initialize queues
    task_queue = queue.Queue()
    ready_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    
    # Start download worker threads
    logger.info(f"Starting {NUM_DOWNLOADERS} download worker(s)...")
    for i in range(NUM_DOWNLOADERS):
        threading.Thread(
            target=download_worker,
            args=(task_queue, ready_queue, i),
            daemon=True,
            name=f"Downloader-{i}"
        ).start()
    
    # Enqueue all tasks
    for task in tasks:
        task_queue.put(task)
    
    # Send poison pills to stop workers
    for _ in range(NUM_DOWNLOADERS):
        task_queue.put(None)

    # Process videos
    pending_results = []
    processed_count = 0
    failed_count = 0
    active_workers = NUM_DOWNLOADERS
    
    logger.info(f"Processing {len(tasks)} videos...")
    
    with tqdm(total=len(tasks), desc="Processing videos") as pbar:
        while processed_count < len(tasks):
            try:
                item = ready_queue.get(timeout=5)
            except queue.Empty:
                if active_workers == 0:
                    break
                continue
            
            if item is None:
                active_workers -= 1
                continue
            
            if item.get('path') is None or item.get('error'):
                video_id = item['meta'].get('video_id', 'unknown')
                logger.warning(
                    f"Skipping video {video_id} due to download failure: "
                    f"{item.get('error', 'Unknown error')}"
                )
                failed_count += 1
                processed_count += 1
                pbar.update(1)
                continue

            # Process video
            video_path = item['path']
            video_id = item['meta'].get('video_id', 'unknown')
            
            # Safely extract view_count
            try:
                raw_views = item['meta'].get('view_count', 0)
                # Handle potentially formatted strings like "1,000" or NaN
                if pd.isna(raw_views):
                    view_count = 0
                elif isinstance(raw_views, str):
                    # Basic cleanup for common formats (remove commas)
                    view_count = int(float(raw_views.replace(',', '')))
                else:
                    view_count = int(raw_views)
            except Exception:
                logger.warning(f"Could not parse view_count for {video_id}, default to 0")
                view_count = 0
            
            try:
                # Pass view_count to processing function
                metrics, v_meta = process_video_batched(video_path, model, view_count)
                
                # Generate results for each detected brand
                for brand, stats in metrics.items():
                    exp_sec = (
                        stats['frames'] * (v_meta['skip_interval'] / v_meta['fps'])
                        if v_meta['fps'] else 0
                    )
                    
                    d = stats['detections']
                    
                    # Calculate mean and std for all metrics
                    def ms(key: str) -> Tuple[float, float]:
                        return calculate_stats(
                            stats[f'{key}_acc'],
                            stats[f'{key}_sq'],
                            d
                        )

                    # Scientific metrics
                    qi_m, qi_s = ms('qi_score')
                    sz_m, sz_s = ms('size_score')
                    sov_m, sov_s = ms('sov_weighted')
                    ds_m, ds_s = ms('dist_score')
                    lg_m, lg_s = ms('legi_score')
                    
                    # Raw metrics
                    cf_m, cf_s = ms('conf')
                    lp_m, lp_s = ms('laplacian')
                    dr_m, dr_s = ms('dist_raw_pct')
                    ar_m, ar_s = ms('area_pct')

                    # Append result
                    pending_results.append({
                        'game_id': item['meta'].get('game_id'),
                        'video_id': item['meta'].get('video_id'),
                        'brand_name': brand,
                        'exposure_seconds': round(exp_sec, 2),
                        'total_detections': d,
                        
                        # Financial Value (Sum of all frames)
                        'total_media_value': round(stats['media_value_sum'], 4),
                        
                        # Scientific Metrics
                        'qi_score_avg': round(qi_m, 4),
                        'qi_score_std': round(qi_s, 4),
                        'size_score_avg': round(sz_m, 4),
                        'size_score_std': round(sz_s, 4),
                        'sov_weighted_avg': round(sov_m, 4),
                        'sov_weighted_std': round(sov_s, 4),
                        'dist_score_avg': round(ds_m, 4),
                        'dist_score_std': round(ds_s, 4),
                        'legi_score_avg': round(lg_m, 4),
                        'legi_score_std': round(lg_s, 4),
                        
                        # Raw Metrics
                        'conf_avg': round(cf_m, 4),
                        'conf_std': round(cf_s, 4),
                        'laplacian_avg': round(lp_m, 2),
                        'laplacian_std': round(lp_s, 2),
                        'dist_raw_pct_avg': round(dr_m, 2),
                        'dist_raw_pct_std': round(dr_s, 2),
                        'area_pct_avg': round(ar_m, 3),
                        'area_pct_std': round(ar_s, 3),
                        
                        'video_url': item['meta'].get('url')
                    })
                    
                logger.info(
                    f"Successfully processed video {video_id}: "
                    f"{len(metrics)} brands detected. Views: {view_count}"
                )
                    
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
                failed_count += 1
                
            finally:
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {video_path}: {e}")
                
                processed_count += 1
                pbar.update(1)
                
                if processed_count % SAVE_EVERY == 0:
                    if save_results_incremental(pending_results, OUTPUT_CSV):
                        logger.info(f"Saved {len(pending_results)} results")
                        pending_results = []
    
    if pending_results:
        save_results_incremental(pending_results, OUTPUT_CSV)
        logger.info(f"Saved final {len(pending_results)} results")
    
    logger.info("=== Processing Complete ===")
    logger.info(f"Successfully processed: {processed_count - failed_count}/{len(tasks)}")
    logger.info(f"Failed: {failed_count}/{len(tasks)}")
    logger.info(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()