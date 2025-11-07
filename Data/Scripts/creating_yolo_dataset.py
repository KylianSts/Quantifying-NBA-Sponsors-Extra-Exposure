import os
import json
import shutil
import math  # On peut utiliser math à la place de numpy pour cette fonction
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
JSON_SOURCE_FILE = "Data/yolo_label_studio.json"
SOURCE_IMAGES_DIR = "Data/train_images"
YOLO_DATASET_DIR = "Data/yolo_dataset" 
VALIDATION_SPLIT_RATIO = 0.2

CLASS_NAMES = [
    "back-court-logo",
    "basket-logo",
    "mid-court-logo",
    "side-court-led-logo",
    "side-court-logo"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_rotated_points(x: float, y: float, width: float, height: float, angle_deg: float, 
                       org_width: int, org_height: int) -> np.ndarray:
    """
    Calcule les 4 coins d'un rectangle pivoté à partir des données de Label Studio.
    VERSION FINALE VALIDÉE - Utilise la formule de la documentation officielle.
    """
    # Conversion des pourcentages en pixels
    x_px = x / 100.0 * org_width
    y_px = y / 100.0 * org_height
    w_px = width / 100.0 * org_width
    h_px = height / 100.0 * org_height

    # Convertir l'angle en radians (sens horaire est standard pour beaucoup d'outils graphiques)
    rotation_rad = math.radians(angle_deg)
    cos_r, sin_r = math.cos(rotation_rad), math.sin(rotation_rad)

    # Calcul des 4 coins en pixels, basé sur la construction vectorielle
    p1 = (x_px, y_px)
    p2 = (x_px + w_px * cos_r, y_px + w_px * sin_r)
    p4 = (x_px - h_px * sin_r, y_px + h_px * cos_r)
    p3 = (p2[0] + (p4[0] - p1[0]), p2[1] + (p4[1] - p1[1]))
    
    # Normalisation des coordonnées des points pour le format YOLO
    points_normalized = np.array([
        [p1[0] / org_width, p1[1] / org_height],
        [p2[0] / org_width, p2[1] / org_height],
        [p3[0] / org_width, p3[1] / org_height],
        [p4[0] / org_width, p4[1] / org_height]
    ])

    return points_normalized


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """Crée l'arborescence des dossiers YOLO."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    paths = {
        'images_train': os.path.join(base_dir, "images", "train"),
        'images_val': os.path.join(base_dir, "images", "val"),
        'labels_train': os.path.join(base_dir, "labels", "train"),
        'labels_val': os.path.join(base_dir, "labels", "val")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def load_and_split_annotations(json_path: str, val_split: float) -> Tuple[List[Dict], List[Dict]]:
    """Charge et divise les annotations JSON en ensembles train/val."""
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    validated_data = [
        item for item in all_data 
        if not item.get('annotations', [{}])[0].get('was_cancelled', False)
    ]
    
    train_data, val_data = train_test_split(
        validated_data, 
        test_size=val_split, 
        random_state=24
    )
    
    return train_data, val_data


def process_subset(subset_data: List[Dict], subset_name: str, paths: Dict[str, str], 
                   source_images_dir: str, class_names: List[str]) -> None:
    """Traite un sous-ensemble (train ou val) et crée les fichiers YOLO."""
    image_path_dest = paths[f'images_{subset_name}']
    label_path_dest = paths[f'labels_{subset_name}']
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for item in tqdm(subset_data, desc=f"Processing {subset_name} set"):
        image_path_source = item['data']['image']
        image_filename = os.path.basename(image_path_source)
        base_name = os.path.splitext(image_filename)[0]

        label_filepath = os.path.join(label_path_dest, f"{base_name}.txt")
        with open(label_filepath, 'w') as f_out:
            annotations = item['annotations'][0]['result']
            
            for ann in annotations:
                value = ann.get('value', {})
                if 'rectanglelabels' not in value:
                    continue
                
                class_name = value['rectanglelabels'][0]
                if class_name not in class_map:
                    continue
                
                class_id = class_map[class_name]

                # Récupérer les dimensions originales de l'image de l'annotation
                org_w, org_h = ann["original_width"], ann["original_height"]
                
                # Convertir le rectangle pivoté en 4 points normalisés
                rotated_points = get_rotated_points(
                    value['x'], value['y'], value['width'], 
                    value['height'], value['rotation'],
                    org_w, org_h
                )
                
                normalized_coords = [f"{coord:.6f}" for p in rotated_points for coord in p]
                f_out.write(f"{class_id} {' '.join(normalized_coords)}\n")

        source_image_filepath = os.path.join(source_images_dir, image_filename)
        dest_image_filepath = os.path.join(image_path_dest, image_filename)
        
        if os.path.exists(source_image_filepath):
            shutil.copy(source_image_filepath, dest_image_filepath)


def create_metadata_files(output_dir: str, class_names: List[str]) -> None:
    """Crée les fichiers classes.txt et dataset.yaml."""
    classes_filepath = os.path.join(output_dir, "classes.txt")
    with open(classes_filepath, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    yaml_filepath = os.path.join(output_dir, "nba_sponsor_dataset.yaml")
    absolute_path_to_dataset = os.path.abspath(output_dir).replace('\\', '/')
    
    yaml_content = f"""# YOLO OBB Dataset Configuration
path: {absolute_path_to_dataset}
train: images/train
val: images/val

# Class dictionary
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_filepath, 'w') as f:
        f.write(yaml_content)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    json_file: str = JSON_SOURCE_FILE,
    images_dir: str = SOURCE_IMAGES_DIR,
    output_dir: str = YOLO_DATASET_DIR,
    val_split: float = VALIDATION_SPLIT_RATIO,
    class_names: List[str] = CLASS_NAMES,
) -> None:
    
    paths = create_directory_structure(output_dir)    
    train_data, val_data = load_and_split_annotations(json_file, val_split)    
    process_subset(train_data, "train", paths, images_dir, class_names)
    process_subset(val_data, "val", paths, images_dir, class_names)
    
    create_metadata_files(output_dir, class_names)

if __name__ == "__main__":
    main()