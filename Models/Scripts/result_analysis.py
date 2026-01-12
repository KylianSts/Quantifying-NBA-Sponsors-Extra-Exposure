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

VERSION = "modelisation_v10"
SUB_MODEL = "yolo11s-obb_fine_tuned_v10_1280"

MODEL_WEIGHTS = f'Models/models_results/{VERSION}/{SUB_MODEL}/weights/best.pt'
IMAGES_FOLDER = 'Data/images/test_images_quality'
GROUND_TRUTH_JSON = 'Data/json_files/ground_truth.json'

BASE_OUTPUT_DIR = Path(f'Models/models_results/{VERSION}/evaluation')
TEMP_DATASET_DIR = Path('Data/temp_test_dataset')

IMG_SIZE = 1280
CONF_THRES = 0.6 
IOU_THRES = 0.7  
BATCH_SIZE = 4   

# --- PARAMETRES DE FUSION ---
MERGE_IOU_THRESH = 0.10 
MERGE_CONTAINMENT_THRESH = 0.50

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

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ============================================================================
# LOGIQUE DE FUSION (MERGE LOGIC)
# ============================================================================

def should_merge(box1, box2):
    x_min1, y_min1 = box1.min(axis=0); x_max1, y_max1 = box1.max(axis=0)
    x_min2, y_min2 = box2.min(axis=0); x_max2, y_max2 = box2.max(axis=0)
    if (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1): return False
    area1 = cv2.contourArea(box1)
    area2 = cv2.contourArea(box2)
    if area1 == 0 or area2 == 0: return False 
    try: 
        inter_area, _ = cv2.intersectConvexConvex(box1.astype(np.float32), box2.astype(np.float32))
    except: return False
    if inter_area <= 0: return False
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return False
    iou = inter_area / union_area
    containment = inter_area / min(area1, area2)
    if iou > MERGE_IOU_THRESH or containment > MERGE_CONTAINMENT_THRESH: return True
    return False

def merge_boxes_points(boxes_list):
    if len(boxes_list) == 1: return boxes_list[0]
    all_points = np.vstack(boxes_list).astype(np.float32)
    rect = cv2.minAreaRect(all_points)
    return cv2.boxPoints(rect).astype(int)

def consolidate_detections(boxes, classes, confs):
    if len(boxes) == 0: return [], [], [], []
    final_boxes, final_classes, final_confs = [], [], []
    merge_groups = [] 
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        global_idxs = np.where(classes == cls)[0]
        cls_boxes = boxes[global_idxs]
        cls_confs = confs[global_idxs]
        n = len(global_idxs)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge(cls_boxes[i], cls_boxes[j]):
                    adj[i].append(j); adj[j].append(i)
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                stack = [i]; visited[i] = True; local_group = []
                while stack:
                    curr = stack.pop(); local_group.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]: visited[neighbor] = True; stack.append(neighbor)
                if len(local_group) > 0:
                    merged_box = merge_boxes_points(cls_boxes[local_group])
                    final_boxes.append(merged_box)
                    final_classes.append(cls)
                    final_confs.append(np.max(cls_confs[local_group]))
                    merge_groups.append(global_idxs[local_group].tolist())
    return np.array(final_boxes), np.array(final_classes), np.array(final_confs), merge_groups

# ============================================================================
# ERROR ANALYSIS LOGIC
# ============================================================================

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
# EIGEN-CAM
# ============================================================================
class YOLOEigenCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        if target_layer is None: self.target_layer = self.find_target_layer()
        else: self.target_layer = target_layer
        self.target_layer.register_forward_hook(self.forward_hook)

    def find_target_layer(self):
        for module in self.model.model.modules():
            if isinstance(module, SPPF): return module
        return list(self.model.model.modules())[-2]

    def forward_hook(self, module, input, output): self.activations = output

    def generate(self, img_path, img_size=(640, 640)):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, img_size)
        img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        with torch.no_grad(): self.model.model(img_tensor)
        activations = self.activations[0] 
        c, h, w = activations.shape
        reshaped_activations = activations.reshape(c, -1).transpose(0, 1)
        try:
            reshaped_activations -= reshaped_activations.mean(dim=0)
            U, S, V = torch.linalg.svd(reshaped_activations, full_matrices=False)
            heatmap = torch.abs(U[:, 0].reshape(h, w))
        except: heatmap = torch.mean(activations, dim=0)
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

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

    # 1. Standard Metrics
    print("\n[STEP 1] Computing Standard Metrics...")
    metrics_dir = BASE_OUTPUT_DIR / 'metrics'
    model = YOLO(MODEL_WEIGHTS)
    metrics = model.val(data=str(yaml_path), split='test', project=str(BASE_OUTPUT_DIR), name='metrics', batch=BATCH_SIZE, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, plots=True, save_json=True)
    report_path = metrics_dir / 'detailed_performance.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'Class':<25} {'mAP50':<12} {'mAP50-95':<12}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'all':<25} {metrics.box.map50:<12.3f} {metrics.box.map:<12.3f}\n")
        if len(metrics.ap_class_index) > 0:
            for i, c_id in enumerate(metrics.ap_class_index):
                f.write(f"{metrics.names[c_id]:<25} {metrics.box.ap50[i]:<12.3f} {metrics.box.maps[i]:<12.3f}\n")
    clear_gpu()
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
        results = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        r = results[0]
        
        # A. VISUALISATION DES HALLUCINATIONS (BALLONS MULTIPLES) - inchangé
        if r.obb is not None:
            ball_indices = torch.nonzero(r.obb.cls == BASKETBALL_CLASS_ID).flatten()
            if len(ball_indices) > 1:
                local_best_idx = torch.argmax(r.obb.conf[ball_indices])
                best_ball_idx = ball_indices[local_best_idx]
                rejected_indices = [idx.item() for idx in ball_indices if idx != best_ball_idx]
                img_pil = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.load_default()
                for idx in rejected_indices:
                    box = r.obb.xyxyxyxy[idx].view(-1).tolist()
                    points = [(box[j], box[j+1]) for j in range(0, 8, 2)]
                    conf = r.obb.conf[idx].item()
                    draw.polygon(points, outline="#FF0000", width=4)
                    draw.text((points[0][0], points[0][1] - 10), f"REJECTED: {conf:.2f}", fill="#FF0000", font=font)
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
        # B. VISUALISATION DES FUSIONS (MERGE) - DEUX IMAGES (BEFORE / AFTER)
        # --------------------------------------------------------------------
        if r.obb is not None and len(r.obb) > 1:
            raw_boxes = r.obb.xyxyxyxy.cpu().numpy().astype(int)
            raw_classes = r.obb.cls.cpu().numpy().astype(int)
            raw_confs = r.obb.conf.cpu().numpy()
            
            f_boxes, f_classes, f_confs, merge_groups = consolidate_detections(raw_boxes, raw_classes, raw_confs)
            
            has_merge = any(len(grp) > 1 for grp in merge_groups)
            
            if has_merge:
                font = ImageFont.load_default()
                # Couleurs opaques
                ORANGE = (255, 165, 0)
                CYAN = (0, 255, 255)
                
                # --- IMAGE 1: BEFORE (Montre les boîtes brutes qui vont être fusionnées) ---
                img_before = Image.open(img_path).convert("RGB")
                draw_before = ImageDraw.Draw(img_before)
                
                # On récupère tous les index bruts impliqués dans une fusion
                merged_indices_flat = [idx for grp in merge_groups if len(grp) > 1 for idx in grp]
                
                for raw_idx in merged_indices_flat:
                    box = raw_boxes[raw_idx].flatten().tolist()
                    points = [(box[j], box[j+1]) for j in range(0, 8, 2)]
                    conf = raw_confs[raw_idx]
                    # Épaisseur 5 (bien visible)
                    draw_before.polygon(points, outline=ORANGE, width=5)
                    draw_before.text((points[0][0], points[0][1] - 15), f"BEFORE: {conf:.2f}", fill=ORANGE, font=font)
                
                # Sauvegarde image AVANT
                img_before.save(merged_dir / f"{img_path.stem}_BEFORE.jpg")

                # --- IMAGE 2: AFTER (Montre le résultat de la fusion) ---
                img_after = Image.open(img_path).convert("RGB")
                draw_after = ImageDraw.Draw(img_after)
                
                for idx_final, group in enumerate(merge_groups):
                    if len(group) > 1:
                        box_final = f_boxes[idx_final].flatten().tolist()
                        points_final = [(box_final[j], box_final[j+1]) for j in range(0, 8, 2)]
                        # Épaisseur 8 (très épais)
                        draw_after.polygon(points_final, outline=CYAN, width=8)
                        draw_after.text((points_final[0][0], points_final[0][1] - 15), "AFTER MERGED", fill=CYAN, font=font)
                
                # Sauvegarde image APRÈS
                img_after.save(merged_dir / f"{img_path.stem}_AFTER.jpg")
                
                total_merged += 1
                print(f"  -> Merge Visualized (Before/After): {img_path.name}")

        # C. Error Analysis
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