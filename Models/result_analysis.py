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

# Try importing shapely for precise OBB IOU
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    print("WARNING: 'shapely' library not found. Falling back to simple box matching.")
    SHAPELY_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "modelisation_v11"
SUB_MODEL = "yolo11s-obb_fine_tuned_v11_1280"

MODEL_WEIGHTS = f'Models/models_results/{VERSION}/{SUB_MODEL}/weights/best.pt'
IMAGES_FOLDER = 'Data/images/test_images_quality'
GROUND_TRUTH_JSON = 'Data/json_files/ground_truth.json'

BASE_OUTPUT_DIR = Path(f'Models/models_results/{VERSION}/evaluation')
TEMP_DATASET_DIR = Path('Data/temp_test_dataset')

IMG_SIZE = 1280
CONF_THRES = 0.6 
IOU_THRES = 0.7  
BATCH_SIZE = 4   

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
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w / 2, h / 2
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    poly_points = []
    rad = math.radians(r)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    for (px, py) in corners:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        final_x = max(0, min(1, (cx + rx) / 100.0))
        final_y = max(0, min(1, (cy + ry) / 100.0))
        poly_points.extend([final_x, final_y])
    return poly_points

def prepare_dataset(json_path, source_img_dir, dest_root, class_map):
    print(">> Preparing dataset...")
    images_dest = dest_root / 'images'
    labels_dest = dest_root / 'labels'
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f: data = json.load(f)
    processed_count = 0
    source_files = {p.name: p for p in Path(source_img_dir).glob('*')}
    gt_cache = {}

    for item in data:
        if not item.get('annotations'): continue
        orig_name = Path(item['data']['image']).name
        src_path = source_files.get(orig_name)
        if not src_path:
            stem = Path(orig_name).stem
            for f in source_files.values():
                if f.stem == stem: src_path = f; break
        if not src_path: continue

        txt_content = []
        gt_objects = []

        for ann in item['annotations']:
            for res in ann.get('result', []):
                val = res.get('value', {})
                label = val.get('rectanglelabels', [None])[0]
                if label not in class_map: continue
                points = []
                if 'points' in val:
                     for pt in val['points']: points.extend([pt[0]/100.0, pt[1]/100.0])
                elif 'x' in val:
                    points = get_box_corners_from_xywhr(val['x'], val['y'], val['width'], val['height'], val.get('rotation', 0))
                
                if len(points) == 8:
                    cid = class_map[label]
                    txt_content.append(f"{cid} " + " ".join(f"{p:.6f}" for p in points))
                    gt_objects.append({'class_id': cid, 'points': points})

        if txt_content:
            shutil.copy(src_path, images_dest / src_path.name)
            with open(labels_dest / (src_path.stem + '.txt'), 'w') as f:
                f.write('\n'.join(txt_content))
            gt_cache[src_path.name] = gt_objects
            processed_count += 1
            
    print(f">> Dataset ready: {processed_count} images.")
    return gt_cache

def apply_single_ball_filter(result):
    if result.obb is None or len(result.obb) == 0: return result, False
    classes = result.obb.cls
    is_ball = (classes == BASKETBALL_CLASS_ID)
    if is_ball.sum() <= 1: return result, False
    keep_mask = ~is_ball
    ball_indices = torch.nonzero(is_ball).flatten()
    best_ball_idx = ball_indices[torch.argmax(result.obb.conf[ball_indices])]
    keep_mask[best_ball_idx] = True
    result.obb = result.obb[keep_mask]
    return result, True

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ============================================================================
# ERROR ANALYSIS LOGIC
# ============================================================================

def calculate_iou_poly(pts1, pts2):
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
    results = model.predict(source=str(img_path), imgsz=img_size, conf=conf, iou=iou_nms, verbose=False)
    result = results[0]
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
            w, h = result.orig_shape[1], result.orig_shape[0]
            preds[-1]['points_norm'] = [c / w if i % 2 == 0 else c / h for i, c in enumerate(preds[-1]['points'])]

    gt_matched = [False] * len(gt_list)
    pred_matched = [False] * len(preds)
    
    for p_idx, p in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_list):
            if g['class_id'] == p['class_id'] and not gt_matched[g_idx]:
                iou = calculate_iou_poly(p['points_norm'], g['points'])
                if iou > best_iou: best_iou = iou; best_gt_idx = g_idx
        if best_iou >= 0.5: pred_matched[p_idx] = True; gt_matched[best_gt_idx] = True

    fp_indices = [i for i, m in enumerate(pred_matched) if not m]
    if fp_indices:
        fp_result = result.new()
        fp_result.obb = result.obb[fp_indices]
        Image.fromarray(fp_result.plot(line_width=4, font_size=2, labels=True, conf=True)[..., ::-1]).save(save_dir_fp / f"FP_{img_path.name}")
        
    fn_indices = [i for i, m in enumerate(gt_matched) if not m]
    if fn_indices:
        img_pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        w, h = img_pil.size
        for g_idx in fn_indices:
            gt_item = gt_list[g_idx]
            cls_name = CLASS_NAMES[gt_item['class_id']]
            pts = gt_item['points']
            abs_pts = [(pts[i]*w, pts[i+1]*h) for i in range(0, 8, 2)]
            draw.polygon(abs_pts, outline="lime", width=5)
            draw.text(abs_pts[0], f"MISSING: {cls_name}", fill="lime")
        img_pil.save(save_dir_fn / f"FN_{img_path.name}")

    return len(fp_indices), len(fn_indices)

# ============================================================================
# EIGEN-CAM IMPLEMENTATION (Better for Object Detection)
# ============================================================================

class YOLOEigenCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        
        # 1. Target the SPPF layer (End of Backbone)
        if target_layer is None:
            self.target_layer = self.find_target_layer()
        else:
            self.target_layer = target_layer
            
        print(f"  [Eigen-CAM] Hooking into layer: {type(self.target_layer).__name__}")

        # 2. Register Forward Hook ONLY (No backward needed for Eigen-CAM)
        self.target_layer.register_forward_hook(self.forward_hook)

    def find_target_layer(self):
        for module in self.model.model.modules():
            if isinstance(module, SPPF): return module
        return list(self.model.model.modules())[-2]

    def forward_hook(self, module, input, output):
        self.activations = output

    def generate(self, img_path, img_size=(640, 640)):
        # 1. Prepare Image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, img_size)
        img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        
        # 2. Forward Pass (No Grad needed)
        with torch.no_grad():
            self.model.model(img_tensor)
            
        # 3. Eigen-CAM Logic
        # Get activations: [Channels, H, W]
        activations = self.activations[0] 
        
        # Flatten spatial dimensions: [Channels, H*W]
        c, h, w = activations.shape
        reshaped_activations = activations.reshape(c, -1).transpose(0, 1) # [H*W, C]
        
        # Compute SVD (Principal Components)
        # We want the first principal component of the spatial features
        try:
            # Center the data
            reshaped_activations -= reshaped_activations.mean(dim=0)
            # SVD calculation
            U, S, V = torch.linalg.svd(reshaped_activations, full_matrices=False)
            # The first column of U corresponds to the first principal component (spatial projection)
            # U is [H*W, K], we take U[:, 0]
            heatmap = U[:, 0].reshape(h, w)
            
            # If the sign is flipped (Eigenvectors direction is arbitrary), we take abs or rely on ReLU later
            # In EigenCAM paper, they often just take the projection. 
            # Let's take the absolute value to capture "activity" regardless of direction
            heatmap = torch.abs(heatmap)
            
        except Exception as e:
            print(f"Eigen-CAM SVD failed, falling back to Mean-CAM: {e}")
            heatmap = torch.mean(activations, dim=0)

        # 4. Post-processing
        heatmap = heatmap.cpu().numpy()
        
        # Normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Colorize
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        return superimposed_img


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*60)
    print(f"EVALUATION PIPELINE: {VERSION}")
    print("="*60)

    # 0. Setup
    if TEMP_DATASET_DIR.exists(): shutil.rmtree(TEMP_DATASET_DIR)
    gt_cache = prepare_dataset(GROUND_TRUTH_JSON, IMAGES_FOLDER, TEMP_DATASET_DIR, FIXED_CLASS_MAP)
    
    yaml_path = TEMP_DATASET_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump({'path': str(TEMP_DATASET_DIR.absolute()), 'train': 'images', 'val': 'images', 'test': 'images', 'names': {v: k for k, v in FIXED_CLASS_MAP.items()}}, f)

    # ------------------------------------------------------------------------
    # 1. Standard Validation (With Report Saving)
    # ------------------------------------------------------------------------
    print("\n[STEP 1] Computing Standard Metrics...")
    metrics_dir = BASE_OUTPUT_DIR / 'metrics'
    model = YOLO(MODEL_WEIGHTS)
    metrics = model.val(data=str(yaml_path), split='test', project=str(BASE_OUTPUT_DIR), name='metrics', batch=BATCH_SIZE, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, plots=True, save_json=True)
    
    # --- SAVE DETAILED TEXT REPORT (METRICS) ---
    print(f"Saving detailed classification report to {metrics_dir}...")
    report_path = metrics_dir / 'detailed_performance.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'Class':<25} {'mAP50':<12} {'mAP50-95':<12}\n")
        f.write("-" * 55 + "\n")
        
        # Global Line
        f.write(f"{'all':<25} {metrics.box.map50:<12.3f} {metrics.box.map:<12.3f}\n")
        
        # Per Class Lines
        if len(metrics.ap_class_index) > 0:
            for i, c_id in enumerate(metrics.ap_class_index):
                class_name = metrics.names[c_id]
                map50 = metrics.box.ap50[i]
                map95 = metrics.box.maps[i]
                f.write(f"{class_name:<25} {map50:<12.3f} {map95:<12.3f}\n")
    # -------------------------------------------

    clear_gpu()
    if metrics_dir.exists():
        for f in metrics_dir.glob('val_batch*.jpg'): 
            try: f.unlink()
            except: pass

    # ------------------------------------------------------------------------
    # 2. Logic Check & 3. Error Analysis
    # ------------------------------------------------------------------------
    print("\n[STEP 2 & 3] Corrections & Error Analysis...")
    corrections_dir = BASE_OUTPUT_DIR / 'corrected_hallucinations'
    errors_dir = BASE_OUTPUT_DIR / 'errors'
    fp_dir = errors_dir / 'FP'
    fn_dir = errors_dir / 'FN'
    
    if corrections_dir.exists(): shutil.rmtree(corrections_dir)
    if errors_dir.exists(): shutil.rmtree(errors_dir)
    corrections_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(list((TEMP_DATASET_DIR / 'images').glob('*')))
    total_corrected = 0
    total_fp = 0
    total_fn = 0
    
    model = YOLO(MODEL_WEIGHTS)

    for i, img_path in enumerate(test_files):
        # A. Logic
        results = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        r = results[0]
        r_filtered, changed = apply_single_ball_filter(r)
        
        if changed and r_filtered.obb is not None:
            only_ball_mask = (r_filtered.obb.cls == BASKETBALL_CLASS_ID)
            r_filtered.obb = r_filtered.obb[only_ball_mask]
            Image.fromarray(r_filtered.plot(line_width=4, font_size=2, conf=True, labels=True)[..., ::-1]).save(corrections_dir / img_path.name)
            total_corrected += 1
            print(f"  -> Logic Correction: {img_path.name}")

        # B. Errors
        n_fp, n_fn = analyze_and_save_errors(model, img_path, gt_cache.get(img_path.name, []), fp_dir, fn_dir, IMG_SIZE, CONF_THRES, IOU_THRES)
        total_fp += n_fp
        total_fn += n_fn
        
        clear_gpu()
        print(f"Processing: {i+1}/{len(test_files)} | FP: {total_fp} | FN: {total_fn}", end='\r')

    # ------------------------------------------------------------------------
    # 4. Interpretability (Eigen-CAM)
    # ------------------------------------------------------------------------
    print("\n\n[STEP 4] Generating Eigen-CAM for Interpretability...")
    cam_dir = BASE_OUTPUT_DIR / 'interpretability'
    if cam_dir.exists(): shutil.rmtree(cam_dir)
    cam_dir.mkdir(parents=True, exist_ok=True)
    
    cam_files = random.sample(test_files, min(30, len(test_files)))
    eigencam_model = YOLO(MODEL_WEIGHTS)
    eigencam = YOLOEigenCAM(eigencam_model)
    
    print(f"Generating heatmaps for {len(cam_files)} images...")
    
    for idx, img_path in enumerate(cam_files):
        try:
            heatmap_img = eigencam.generate(img_path, img_size=(IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(cam_dir / f"eigencam_{img_path.name}"), heatmap_img)
        except Exception as e:
            print(f"Skipping {img_path.name} due to error: {e}")
        
        print(f"Eigen-CAM: {idx+1}/{len(cam_files)}", end='\r')
        
    clear_gpu()

    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE")
    print(f"1. Metrics Report: {metrics_dir / 'detailed_performance.txt'}")
    print(f"2. Corrections:    {corrections_dir}")
    print(f"3. Errors (FP/FN): {errors_dir}")
    print(f"4. Eigen-CAM:      {cam_dir}")
    print("="*60)

if __name__ == '__main__':
    main()