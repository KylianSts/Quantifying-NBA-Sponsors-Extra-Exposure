import os
import json
import shutil
import yaml
import math
import gc
import torch
import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF
from collections import Counter

# Attempt to import Shapely for accurate polygon IoU calculations.
# If not present, the code falls back to basic logic or skips certain checks.
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    print("WARNING: 'shapely' library not found. Falling back to simple box matching.")
    SHAPELY_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "modelisation_v10"
SUB_MODEL = "yolo11s-obb_fine_tuned_v10_1280"

# Paths for input data and model weights
MODEL_WEIGHTS = f'Models/models_results/{VERSION}/{SUB_MODEL}/weights/best.pt'
IMAGES_FOLDER = 'Data/images/test_images_quality'
GROUND_TRUTH_JSON = 'Data/json_files/ground_truth.json'

# Output directories
BASE_OUTPUT_DIR = Path(f'Models/models_results/{VERSION}/evaluation')
TEMP_DATASET_DIR = Path('Data/temp_test_dataset')

# Hyperparameters for Inference
IMG_SIZE = 1280
CONF_THRES = 0.6   # Minimum confidence score to consider a detection
IOU_THRES = 0.7    # NMS (Non-Maximum Suppression) IoU threshold
BATCH_SIZE = 4     

# --- PARAMETERS FOR MERGING LOGIC ---
# Thresholds used to decide if two overlapping boxes should be merged into one
MERGE_IOU_THRESH = 0.10 
MERGE_CONTAINMENT_THRESH = 0.50

# Class Definitions
CLASS_NAMES = [
    "back-court-logo",      # 0
    "basket-logo",          # 1
    "mid-court-logo",       # 2
    "side-court-led-logo",  # 3
    "side-court-logo",      # 4
    "basketball"            # 5
]
FIXED_CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BASKETBALL_CLASS_ID = 5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_box_corners_from_xywhr(x, y, w, h, r):
    """
    Calculates the 4 corner points of a rotated rectangle.
    
    Args:
        x, y (float): Center coordinates (percentage 0-100).
        w, h (float): Width and Height (percentage 0-100).
        r (float): Rotation angle in degrees.

    Returns:
        list: Flat list of [x1, y1, x2, y2, ..., x4, y4] normalized to 0-1.
    """
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w / 2, h / 2
    # Define corners relative to the center (unrotated)
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    poly_points = []
    
    # Convert degrees to radians for rotation matrix
    rad = math.radians(r)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    
    for (px, py) in corners:
        # Apply 2D rotation matrix
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        # Normalize to 0-1 range based on assumed 0-100 input scale
        final_x = max(0, min(1, (cx + rx) / 100.0))
        final_y = max(0, min(1, (cy + ry) / 100.0))
        poly_points.extend([final_x, final_y])
    return poly_points

def prepare_dataset(json_path, source_img_dir, dest_root, class_map):
    """
    Parses a Label Studio style JSON ground truth file and organizes images/labels
    into a YOLO OBB compatible directory structure.

    Args:
        json_path (str): Path to ground truth JSON.
        source_img_dir (str): Folder containing source images.
        dest_root (Path): Where to create the temporary dataset.
        class_map (dict): Mapping of class names to IDs.

    Returns:
        dict: Cache of ground truth objects for later analysis {filename: [objects]}.
    """
    print(">> Preparing dataset...")
    images_dest = dest_root / 'images'
    labels_dest = dest_root / 'labels'
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f: data = json.load(f)
    processed_count = 0
    # Map filenames to paths for quick lookup
    source_files = {p.name: p for p in Path(source_img_dir).glob('*')}
    gt_cache = {}

    for item in data:
        if not item.get('annotations'): continue
        
        # Locate the source image file
        orig_name = Path(item['data']['image']).name
        src_path = source_files.get(orig_name)
        
        # Fallback: try matching by stem if exact name fails
        if not src_path:
            stem = Path(orig_name).stem
            for f in source_files.values():
                if f.stem == stem: src_path = f; break
        if not src_path: continue

        txt_content = []
        gt_objects = []

        # Process annotations
        for ann in item['annotations']:
            for res in ann.get('result', []):
                val = res.get('value', {})
                label = val.get('rectanglelabels', [None])[0]
                
                if label not in class_map: continue
                
                points = []
                # Handle Polygon format
                if 'points' in val:
                      for pt in val['points']: points.extend([pt[0]/100.0, pt[1]/100.0])
                # Handle Rotated Bounding Box (XYWHR) format
                elif 'x' in val:
                    points = get_box_corners_from_xywhr(val['x'], val['y'], val['width'], val['height'], val.get('rotation', 0))
                
                # Verify we have a valid quad (4 points, 2 coords each = 8 values)
                if len(points) == 8:
                    cid = class_map[label]
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    txt_content.append(f"{cid} " + " ".join(f"{p:.6f}" for p in points))
                    gt_objects.append({'class_id': cid, 'points': points})

        # Save image and label file if valid annotations exist
        if txt_content:
            shutil.copy(src_path, images_dest / src_path.name)
            with open(labels_dest / (src_path.stem + '.txt'), 'w') as f:
                f.write('\n'.join(txt_content))
            gt_cache[src_path.name] = gt_objects
            processed_count += 1
            
    print(f">> Dataset ready: {processed_count} images.")
    return gt_cache

def clear_gpu():
    """Forces garbage collection and clears CUDA cache to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ============================================================================
# MERGE LOGIC (FUSION)
# ============================================================================

def should_merge(box1, box2):
    """
    Determines if two boxes should be merged based on overlap or containment.
    
    Args:
        box1, box2 (np.array): Arrays of shape (4, 2) representing box corners.
    
    Returns:
        bool: True if they satisfy merge criteria.
    """
    # Quick bounding rect check to avoid expensive operations
    x_min1, y_min1 = box1.min(axis=0); x_max1, y_max1 = box1.max(axis=0)
    x_min2, y_min2 = box2.min(axis=0); x_max2, y_max2 = box2.max(axis=0)
    
    # If bounding rectangles don't intersect, polygons definitely don't
    if (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1): return False
    
    area1 = cv2.contourArea(box1)
    area2 = cv2.contourArea(box2)
    if area1 == 0 or area2 == 0: return False 
    
    try: 
        # Calculate intersection area using OpenCV
        inter_area, _ = cv2.intersectConvexConvex(box1.astype(np.float32), box2.astype(np.float32))
    except: return False
    
    if inter_area <= 0: return False
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return False
    
    iou = inter_area / union_area
    containment = inter_area / min(area1, area2) # % of smaller box inside larger box
    
    if iou > MERGE_IOU_THRESH or containment > MERGE_CONTAINMENT_THRESH: return True
    return False

def merge_boxes_points(boxes_list):
    """
    Combines a list of boxes into a single minimum bounding rectangle.
    
    Args:
        boxes_list (np.array): Array of shape (N, 4, 2).
        
    Returns:
        np.array: The 4 corners of the merged box.
    """
    if len(boxes_list) == 1: return boxes_list[0]
    # Stack all points and find the MinAreaRect enclosing all of them
    all_points = np.vstack(boxes_list).astype(np.float32)
    rect = cv2.minAreaRect(all_points)
    return cv2.boxPoints(rect).astype(int)

def consolidate_detections(boxes, classes, confs):
    """
    Groups overlapping detections of the same class using a graph-based approach
    (connected components) and merges them.

    Args:
        boxes (np.array): Shape (N, 4, 2).
        classes (np.array): Shape (N,).
        confs (np.array): Shape (N,).

    Returns:
        tuple: (final_boxes, final_classes, final_confs, merge_groups)
    """
    if len(boxes) == 0: return [], [], [], []
    final_boxes, final_classes, final_confs = [], [], []
    merge_groups = [] 
    
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        # Get indices for current class
        global_idxs = np.where(classes == cls)[0]
        cls_boxes = boxes[global_idxs]
        cls_confs = confs[global_idxs]
        n = len(global_idxs)
        
        # Build adjacency list: adj[i] contains indices j that overlap with i
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge(cls_boxes[i], cls_boxes[j]):
                    adj[i].append(j); adj[j].append(i)
        
        # Find connected components (groups of merging boxes)
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                stack = [i]; visited[i] = True; local_group = []
                while stack:
                    curr = stack.pop(); local_group.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]: visited[neighbor] = True; stack.append(neighbor)
                
                # Merge the group
                if len(local_group) > 0:
                    merged_box = merge_boxes_points(cls_boxes[local_group])
                    final_boxes.append(merged_box)
                    final_classes.append(cls)
                    # Take the max confidence of the group
                    final_confs.append(np.max(cls_confs[local_group]))
                    # Store original indices for visualization/debugging
                    merge_groups.append(global_idxs[local_group].tolist())
                    
    return np.array(final_boxes), np.array(final_classes), np.array(final_confs), merge_groups

# ============================================================================
# ERROR ANALYSIS LOGIC
# ============================================================================

def apply_single_ball_filter(result):
    """
    Ensures only one basketball exists per image (the one with highest confidence).
    
    Args:
        result: Ultralytics result object.
        
    Returns:
        tuple: (modified_result, bool_indicating_if_correction_occurred)
    """
    if result.obb is None or len(result.obb) == 0: return result, False
    classes = result.obb.cls
    is_ball = (classes == BASKETBALL_CLASS_ID)
    
    # If 0 or 1 ball, nothing to do
    if is_ball.sum() <= 1: return result, False
    
    keep_mask = ~is_ball # Keep everything that is NOT a ball
    
    # Find the best ball
    ball_indices = torch.nonzero(is_ball).flatten()
    best_ball_idx = ball_indices[torch.argmax(result.obb.conf[ball_indices])]
    
    keep_mask[best_ball_idx] = True
    
    # Filter detections
    result.obb = result.obb[keep_mask]
    return result, True

def calculate_iou_poly(pts1, pts2):
    """
    Calculates Intersection over Union (IoU) for two polygons using Shapely.
    
    Args:
        pts1, pts2: Flattened lists of coordinates [x1, y1, ... x4, y4].
        
    Returns:
        float: IoU value (0.0 to 1.0).
    """
    if not SHAPELY_AVAILABLE: return 0.0
    try:
        p1 = Polygon([(pts1[i], pts1[i+1]) for i in range(0, 8, 2)])
        p2 = Polygon([(pts2[i], pts2[i+1]) for i in range(0, 8, 2)])
        
        if not p1.is_valid or not p2.is_valid: return 0.0
        
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0.0
    except:
        return 0.0

def analyze_and_save_errors(model, img_path, gt_list, save_dir_fp, save_dir_fn, img_size, conf, iou_nms):
    """
    Runs inference, compares with Ground Truth, and saves images of 
    False Positives (FP) and False Negatives (FN).

    Args:
        model: Loaded YOLO model.
        img_path: Path to the image file.
        gt_list: List of ground truth objects for this image.
        save_dir_fp: Directory to save FP images.
        save_dir_fn: Directory to save FN images.
        img_size, conf, iou_nms: Inference parameters.

    Returns:
        tuple: (count_fp, count_fn)
    """
    # Run inference
    results = model.predict(source=str(img_path), imgsz=img_size, conf=conf, iou=iou_nms, verbose=False)
    result = results[0]
    
    # Apply business logic (single ball rule)
    result, _ = apply_single_ball_filter(result)
    
    preds = []
    if result.obb is not None:
        for i in range(len(result.obb)):
            preds.append({
                'class_id': int(result.obb.cls[i].item()),
                'points': result.obb.xyxyxyxy[i].view(-1).tolist(),
                'orig_idx': i,
                'conf': float(result.obb.conf[i].item())
            })
            # Add normalized points for IoU calculation
            w, h = result.orig_shape[1], result.orig_shape[0]
            preds[-1]['points_norm'] = [c / w if i % 2 == 0 else c / h for i, c in enumerate(preds[-1]['points'])]

    gt_matched = [False] * len(gt_list)
    pred_matched = [False] * len(preds)
    
    # Match Predictions to Ground Truth (Greedy Matching)
    for p_idx, p in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_list):
            if g['class_id'] == p['class_id'] and not gt_matched[g_idx]:
                iou = calculate_iou_poly(p['points_norm'], g['points'])
                if iou > best_iou: best_iou = iou; best_gt_idx = g_idx
        
        # Determine match based on IoU threshold (0.5 hardcoded here for analysis)
        if best_iou >= 0.5: pred_matched[p_idx] = True; gt_matched[best_gt_idx] = True

    # Handle False Positives
    fp_indices = [i for i, m in enumerate(pred_matched) if not m]
    if fp_indices:
        fp_result = result.new()
        fp_result.obb = result.obb[fp_indices]
        # Plot and save
        Image.fromarray(fp_result.plot(line_width=4, font_size=2, labels=True, conf=True)[..., ::-1]).save(save_dir_fp / f"FP_{img_path.name}")
        
    # Handle False Negatives
    fn_indices = [i for i, m in enumerate(gt_matched) if not m]
    if fn_indices:
        img_pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        w, h = img_pil.size
        for g_idx in fn_indices:
            gt_item = gt_list[g_idx]
            cls_name = CLASS_NAMES[gt_item['class_id']]
            pts = gt_item['points']
            # Scale normalized GT points back to absolute pixels
            abs_pts = [(pts[i]*w, pts[i+1]*h) for i in range(0, 8, 2)]
            draw.polygon(abs_pts, outline="lime", width=5)
            draw.text(abs_pts[0], f"MISSING: {cls_name}", fill="lime")
        img_pil.save(save_dir_fn / f"FN_{img_path.name}")

    return len(fp_indices), len(fn_indices)

# ============================================================================
# EIGEN-CAM (Interpretability)
# ============================================================================
class YOLOEigenCAM:
    """
    Generates Eigen-CAM heatmaps to visualize which parts of the image 
    the model focuses on.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        # Auto-detect the target layer (usually SPPF) if not provided
        if target_layer is None: self.target_layer = self.find_target_layer()
        else: self.target_layer = target_layer
        self.target_layer.register_forward_hook(self.forward_hook)

    def find_target_layer(self):
        """Finds the SPPF layer or the penultimate layer to hook into."""
        for module in self.model.model.modules():
            if isinstance(module, SPPF): return module
        return list(self.model.model.modules())[-2]

    def forward_hook(self, module, input, output): 
        """Callback to capture layer activations during forward pass."""
        self.activations = output

    def generate(self, img_path, img_size=(640, 640)):
        """
        Runs the image through the model and computes the Principal Component
        of the activations to create a heatmap.
        """
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, img_size)
        # Preprocess: normalize and permute dims (HWC -> CHW)
        img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        
        with torch.no_grad(): self.model.model(img_tensor)
        
        activations = self.activations[0] 
        c, h, w = activations.shape
        reshaped_activations = activations.reshape(c, -1).transpose(0, 1)
        
        # Compute Principal Component Analysis (PCA) approx via SVD
        try:
            reshaped_activations -= reshaped_activations.mean(dim=0)
            U, S, V = torch.linalg.svd(reshaped_activations, full_matrices=False)
            heatmap = torch.abs(U[:, 0].reshape(h, w))
        except: 
            # Fallback to mean activation if SVD fails
            heatmap = torch.mean(activations, dim=0)
            
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        # Normalize heatmap
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
        
        # Resize heatmap to original image size and colorize
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        return cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*60)
    print(f"EVALUATION PIPELINE: {VERSION}")
    print("="*60)

    # 0. Setup
    # Clean previous temp data and convert JSON GT to YOLO txt format
    if TEMP_DATASET_DIR.exists(): shutil.rmtree(TEMP_DATASET_DIR)
    gt_cache = prepare_dataset(GROUND_TRUTH_JSON, IMAGES_FOLDER, TEMP_DATASET_DIR, FIXED_CLASS_MAP)
    
    # Create dataset.yaml required by Ultralytics
    yaml_path = TEMP_DATASET_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump({'path': str(TEMP_DATASET_DIR.absolute()), 'train': 'images', 'val': 'images', 'test': 'images', 'names': {v: k for k, v in FIXED_CLASS_MAP.items()}}, f)

    # 1. Standard Metrics
    print("\n[STEP 1] Computing Standard Metrics...")
    metrics_dir = BASE_OUTPUT_DIR / 'metrics'
    model = YOLO(MODEL_WEIGHTS)
    
    # Run standard YOLO validation (mAP calculations)
    metrics = model.val(data=str(yaml_path), split='test', project=str(BASE_OUTPUT_DIR), name='metrics', batch=BATCH_SIZE, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, plots=True, save_json=True)
    
    # Write custom report
    report_path = metrics_dir / 'detailed_performance.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'Class':<25} {'mAP50':<12} {'mAP50-95':<12}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'all':<25} {metrics.box.map50:<12.3f} {metrics.box.map:<12.3f}\n")
        if len(metrics.ap_class_index) > 0:
            for i, c_id in enumerate(metrics.ap_class_index):
                f.write(f"{metrics.names[c_id]:<25} {metrics.box.ap50[i]:<12.3f} {metrics.box.maps[i]:<12.3f}\n")
    
    clear_gpu()
    # Cleanup auto-generated validation batch images
    if metrics_dir.exists():
        for f in metrics_dir.glob('val_batch*.jpg'): 
            try: f.unlink()
            except: pass

    # 2. Logic Check & 3. Error Analysis
    print("\n[STEP 2 & 3] Corrections (Hallucinations + Merges) & Error Analysis...")
    corrections_dir = BASE_OUTPUT_DIR / 'corrected_hallucinations'
    merged_dir = BASE_OUTPUT_DIR / 'merged_detections'
    errors_dir = BASE_OUTPUT_DIR / 'errors'
    fp_dir = errors_dir / 'FP'
    fn_dir = errors_dir / 'FN'
    
    # Reset output directories
    if corrections_dir.exists(): shutil.rmtree(corrections_dir)
    if merged_dir.exists(): shutil.rmtree(merged_dir)
    if errors_dir.exists(): shutil.rmtree(errors_dir)
    
    corrections_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(list((TEMP_DATASET_DIR / 'images').glob('*')))
    total_corrected = 0
    total_merged = 0
    total_fp = 0
    total_fn = 0
    
    model = YOLO(MODEL_WEIGHTS)

    for i, img_path in enumerate(test_files):
        # Run inference for analysis steps
        results = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        r = results[0]
        
        # A. VISUALISATION OF HALLUCINATIONS (MULTIPLE BALLS)
        # Checks if multiple basketballs are detected and visualizes which one is kept/rejected
        if r.obb is not None:
            ball_indices = torch.nonzero(r.obb.cls == BASKETBALL_CLASS_ID).flatten()
            if len(ball_indices) > 1:
                # Logic: Keep the one with highest confidence
                local_best_idx = torch.argmax(r.obb.conf[ball_indices])
                best_ball_idx = ball_indices[local_best_idx]
                rejected_indices = [idx.item() for idx in ball_indices if idx != best_ball_idx]
                
                img_pil = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.load_default()
                
                # Draw Rejected (Red)
                for idx in rejected_indices:
                    box = r.obb.xyxyxyxy[idx].view(-1).tolist()
                    points = [(box[j], box[j+1]) for j in range(0, 8, 2)]
                    conf = r.obb.conf[idx].item()
                    draw.polygon(points, outline="#FF0000", width=4)
                    draw.text((points[0][0], points[0][1] - 10), f"REJECTED: {conf:.2f}", fill="#FF0000", font=font)
                
                # Draw Kept (Green)
                if best_ball_idx != -1:
                    box = r.obb.xyxyxyxy[best_ball_idx].view(-1).tolist()
                    points = [(box[j], box[j+1]) for j in range(0, 8, 2)]
                    conf = r.obb.conf[best_ball_idx].item()
                    draw.polygon(points, outline="#00FF00", width=4)
                    draw.text((points[0][0], points[0][1] - 10), f"KEPT: {conf:.2f}", fill="#00FF00", font=font)
                
                img_pil.save(corrections_dir / img_path.name)
                total_corrected += 1
                print(f"  -> Hallucination Fixed: {img_path.name}")

        # --------------------------------------------------------------------
        # B. VISUALISATION OF MERGES - TWO IMAGES (BEFORE / AFTER)
        # Checks if boxes overlap and visualizes the merge process
        # --------------------------------------------------------------------
        if r.obb is not None and len(r.obb) > 1:
            raw_boxes = r.obb.xyxyxyxy.cpu().numpy().astype(int)
            raw_classes = r.obb.cls.cpu().numpy().astype(int)
            raw_confs = r.obb.conf.cpu().numpy()
            
            # Apply merge logic
            f_boxes, f_classes, f_confs, merge_groups = consolidate_detections(raw_boxes, raw_classes, raw_confs)
            
            # Check if any group resulted in a merge (size > 1)
            has_merge = any(len(grp) > 1 for grp in merge_groups)
            
            if has_merge:
                font = ImageFont.load_default()
                ORANGE = (255, 165, 0)
                CYAN = (0, 255, 255)
                
                # --- IMAGE 1: BEFORE (Shows raw boxes that will be merged) ---
                img_before = Image.open(img_path).convert("RGB")
                draw_before = ImageDraw.Draw(img_before)
                
                # Identify indices involved in merges
                merged_indices_flat = [idx for grp in merge_groups if len(grp) > 1 for idx in grp]
                
                for raw_idx in merged_indices_flat:
                    box = raw_boxes[raw_idx].flatten().tolist()
                    points = [(box[j], box[j+1]) for j in range(0, 8, 2)]
                    conf = raw_confs[raw_idx]
                    draw_before.polygon(points, outline=ORANGE, width=5)
                    draw_before.text((points[0][0], points[0][1] - 15), f"BEFORE: {conf:.2f}", fill=ORANGE, font=font)
                
                img_before.save(merged_dir / f"{img_path.stem}_BEFORE.jpg")

                # --- IMAGE 2: AFTER (Shows the resulting consolidated box) ---
                img_after = Image.open(img_path).convert("RGB")
                draw_after = ImageDraw.Draw(img_after)
                
                for idx_final, group in enumerate(merge_groups):
                    if len(group) > 1:
                        box_final = f_boxes[idx_final].flatten().tolist()
                        points_final = [(box_final[j], box_final[j+1]) for j in range(0, 8, 2)]
                        draw_after.polygon(points_final, outline=CYAN, width=8)
                        draw_after.text((points_final[0][0], points_final[0][1] - 15), "AFTER MERGED", fill=CYAN, font=font)
                
                img_after.save(merged_dir / f"{img_path.stem}_AFTER.jpg")
                
                total_merged += 1
                print(f"  -> Merge Visualized (Before/After): {img_path.name}")

        # C. Error Analysis
        # Calculates False Positives/Negatives and saves visual proof
        n_fp, n_fn = analyze_and_save_errors(model, img_path, gt_cache.get(img_path.name, []), fp_dir, fn_dir, IMG_SIZE, CONF_THRES, IOU_THRES)
        total_fp += n_fp
        total_fn += n_fn
        
        clear_gpu()
        print(f"Processing: {i+1}/{len(test_files)} | Corrected: {total_corrected} | Merged: {total_merged}", end='\r')

    # 4. Interpretability (Eigen-CAM)
    print("\n\n[STEP 4] Generating Eigen-CAM for Interpretability...")
    cam_dir = BASE_OUTPUT_DIR / 'interpretability'
    if cam_dir.exists(): shutil.rmtree(cam_dir)
    cam_dir.mkdir(parents=True, exist_ok=True)
    
    # Process a random subset of 30 images to save time
    cam_files = random.sample(test_files, min(30, len(test_files)))
    eigencam_model = YOLO(MODEL_WEIGHTS)
    eigencam = YOLOEigenCAM(eigencam_model)
    
    for idx, img_path in enumerate(cam_files):
        try:
            heatmap_img = eigencam.generate(img_path, img_size=(IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(cam_dir / f"eigencam_{img_path.name}"), heatmap_img)
        except: pass
        print(f"Eigen-CAM: {idx+1}/{len(cam_files)}", end='\r')
    clear_gpu()

    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE")
    print(f"1. Metrics Report: {metrics_dir / 'detailed_performance.txt'}")
    print(f"2. Hallucinations: {corrections_dir}")
    print(f"3. Merges:         {merged_dir} (Contains _BEFORE.jpg and _AFTER.jpg pairs)")
    print(f"4. Errors (FP/FN): {errors_dir}")
    print("="*60)

if __name__ == '__main__':
    main()