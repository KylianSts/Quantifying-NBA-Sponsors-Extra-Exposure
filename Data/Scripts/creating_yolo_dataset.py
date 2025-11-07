import os
import json
import shutil
import math
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
JSON_SOURCE_FILE = "Data/yolo_label_studio.json"
SOURCE_IMAGES_DIR = "Data/images/train_images"
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

def get_rotated_points_from_rectangle(value: dict, org_width: int, org_height: int) -> np.ndarray:
    """
    Calculate 4 corners from 'rectanglelabels' format (x, y, w, h, rotation).
    
    This handles the standard Label Studio rectangle annotation with rotation.
    Uses vector-based calculation to compute the 4 corner points.
    
    Args:
        value: Dictionary containing x, y, width, height, and rotation
        org_width: Original image width in pixels
        org_height: Original image height in pixels
    
    Returns:
        Array of shape (4, 2) with normalized corner coordinates [0, 1]
    """
    x_rel, y_rel = value["x"], value["y"]
    w_rel, h_rel = value["width"], value["height"]
    rotation_deg = value.get("rotation", 0)

    # Convert percentages to pixels
    x_px = x_rel / 100.0 * org_width
    y_px = y_rel / 100.0 * org_height
    w_px = w_rel / 100.0 * org_width
    h_px = h_rel / 100.0 * org_height

    # Convert rotation to radians
    rotation_rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rotation_rad), math.sin(rotation_rad)

    # Calculate 4 corners using vector construction
    p1 = (x_px, y_px)
    p2 = (x_px + w_px * cos_r, y_px + w_px * sin_r)
    p4 = (x_px - h_px * sin_r, y_px + h_px * cos_r)
    p3 = (p2[0] + (p4[0] - p1[0]), p2[1] + (p4[1] - p1[1]))
    
    # Normalize to [0, 1] range
    points_normalized = np.array([
        [p1[0] / org_width, p1[1] / org_height],
        [p2[0] / org_width, p2[1] / org_height],
        [p3[0] / org_width, p3[1] / org_height],
        [p4[0] / org_width, p4[1] / org_height]
    ])
    return points_normalized


def get_points_from_polygon(value: dict, org_width: int, org_height: int) -> np.ndarray:
    """
    Read 4 corners directly from 'polygonlabels' format or 'rectanglelabels' with 'points'.
    
    This handles cases where Label Studio stores the annotation as pre-computed points
    rather than as x, y, width, height with rotation.
    
    Args:
        value: Dictionary containing a 'points' list
        org_width: Original image width in pixels (unused but kept for consistency)
        org_height: Original image height in pixels (unused but kept for consistency)
    
    Returns:
        Array of shape (4, 2) with normalized corner coordinates [0, 1]
    """
    points = value["points"]
    
    # Points are already in percentage format, just convert to [0, 1] range
    points_normalized = np.array([
        [p[0] / 100.0, p[1] / 100.0] for p in points
    ])
    return points_normalized


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """
    Create YOLO dataset directory structure.
    
    Args:
        base_dir: Base directory path for the dataset
    
    Returns:
        Dictionary with keys: 'images_train', 'images_val', 'labels_train', 'labels_val'
    """
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
    """
    Load and split JSON annotations into train/val sets.
    
    Args:
        json_path: Path to Label Studio JSON export file
        val_split: Proportion of data for validation (e.g., 0.2 for 20%)
    
    Returns:
        Tuple of (training_data, validation_data)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Filter out cancelled annotations
    validated_data = [
        item for item in all_data 
        if not item.get('annotations', [{}])[0].get('was_cancelled', False)
    ]
    
    # Split with fixed random seed for reproducibility
    train_data, val_data = train_test_split(
        validated_data, 
        test_size=val_split, 
        random_state=24
    )
    
    return train_data, val_data


def process_subset(subset_data: List[Dict], subset_name: str, paths: Dict[str, str], 
                   source_images_dir: str, class_names: List[str]) -> None:
    """
    Process a subset (train or val) and create YOLO files.
    
    This function handles two annotation formats:
    1. Rectangle with rotation (x, y, width, height, rotation)
    2. Pre-computed points (4 corner coordinates)
    
    Args:
        subset_data: List of annotation items to process
        subset_name: Name of subset ("train" or "val")
        paths: Dictionary with directory paths
        source_images_dir: Directory containing source images
        class_names: List of class names
    """
    image_path_dest = paths[f'images_{subset_name}']
    label_path_dest = paths[f'labels_{subset_name}']
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for item in tqdm(subset_data, desc=f"Processing {subset_name} set"):
        image_path_source = item['data']['image']
        image_filename = os.path.basename(image_path_source)
        base_name = os.path.splitext(image_filename)[0]

        # Create YOLO label file
        label_filepath = os.path.join(label_path_dest, f"{base_name}.txt")
        with open(label_filepath, 'w') as f_out:
            annotations = item['annotations'][0]['result']
            
            for ann in annotations:
                value = ann.get('value', {})
                org_w, org_h = ann["original_width"], ann["original_height"]
                
                rotated_points = None
                
                # Handle rectanglelabels annotations
                if 'rectanglelabels' in value:
                    class_name = value['rectanglelabels'][0]
                    if class_name not in class_map:
                        continue
                    
                    class_id = class_map[class_name]
                    
                    # Format 1: Rectangle with x, y, width, height, rotation
                    if 'x' in value:
                        rotated_points = get_rotated_points_from_rectangle(value, org_w, org_h)
                    
                    # Format 2: Pre-computed points
                    elif 'points' in value:
                        rotated_points = get_points_from_polygon(value, org_w, org_h)
                    
                    # Write to file if points were computed
                    if rotated_points is not None:
                        normalized_coords = [f"{coord:.6f}" for p in rotated_points for coord in p]
                        f_out.write(f"{class_id} {' '.join(normalized_coords)}\n")

        # Copy source image to destination
        source_image_filepath = os.path.join(source_images_dir, image_filename)
        dest_image_filepath = os.path.join(image_path_dest, image_filename)
        
        if os.path.exists(source_image_filepath):
            shutil.copy(source_image_filepath, dest_image_filepath)


def create_metadata_files(output_dir: str, class_names: List[str]) -> None:
    """
    Create classes.txt and dataset.yaml files.
    
    Args:
        output_dir: Directory where files will be created
        class_names: List of class names in order
    """
    # Create classes.txt
    classes_filepath = os.path.join(output_dir, "classes.txt")
    with open(classes_filepath, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Create YAML configuration
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
    """
    Create YOLO OBB dataset from Label Studio JSON export.
    
    This function supports two annotation formats:
    1. Standard rectangles with rotation (x, y, width, height, rotation)
    2. Pre-computed polygon points (4 corner coordinates)
    
    Args:
        json_file: Path to Label Studio JSON export
        images_dir: Directory with source images
        output_dir: Output directory for YOLO dataset
        val_split: Validation split ratio (default: 0.2)
        class_names: List of class names
    """
    paths = create_directory_structure(output_dir)    
    train_data, val_data = load_and_split_annotations(json_file, val_split)    
    process_subset(train_data, "train", paths, images_dir, class_names)
    process_subset(val_data, "val", paths, images_dir, class_names)
    create_metadata_files(output_dir, class_names)


if __name__ == "__main__":
    main()