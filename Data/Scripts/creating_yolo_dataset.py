"""
YOLO OBB Dataset Converter from Label Studio Annotations

This script converts Label Studio JSON exports into YOLO Oriented Bounding Box (OBB) format
for training object detection models with rotated bounding boxes. It handles multiple
annotation formats and creates a properly structured YOLO dataset.

Key features:
- Converts Label Studio JSON annotations to YOLO OBB format
- Supports two annotation formats: rectangles with rotation and pre-computed polygon points
- Automatically splits data into training and validation sets
- Creates YOLO-compatible directory structure (images/labels folders)
- Generates dataset configuration files (YAML and classes.txt)
- Filters out cancelled annotations
- Handles rotated bounding boxes using vector-based corner calculation
"""

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

# Path to Label Studio JSON export file containing annotations
JSON_SOURCE_FILE = "Data/json_files/yolo_label_studio.json"

# Directory containing the annotated images
SOURCE_IMAGES_DIR = "Data/images/train_images"

# Output directory for the YOLO-formatted dataset
YOLO_DATASET_DIR = "Data/yolo_dataset"

# Proportion of data to use for validation (0.2 = 20% validation, 80% training)
VALIDATION_SPLIT_RATIO = 0.2

# List of object classes in the dataset (order matters - index becomes class ID)
CLASS_NAMES = [
    "back-court-logo",       # Class 0: Logos on back court
    "basket-logo",           # Class 1: Logos near baskets
    "mid-court-logo",        # Class 2: Center court logos
    "side-court-led-logo",   # Class 3: LED display logos on sidelines
    "side-court-logo"        # Class 4: Static logos on sidelines
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_rotated_points_from_rectangle(value: dict, org_width: int, org_height: int) -> np.ndarray:
    """
    Calculate 4 corners from 'rectanglelabels' format (x, y, width, height, rotation).
    
    This handles the standard Label Studio rectangle annotation with rotation.
    Uses vector-based calculation to compute the 4 corner points of a rotated rectangle.
    The rotation is applied around the top-left corner (x, y) as the pivot point.
    
    Args:
        value: Dictionary containing x, y, width, height (in percentages), and rotation (in degrees)
        org_width: Original image width in pixels
        org_height: Original image height in pixels
    
    Returns:
        Array of shape (4, 2) with normalized corner coordinates in range [0, 1]
        Points are ordered: top-left, top-right, bottom-right, bottom-left
    """
    # Extract rectangle parameters (all in percentage format from Label Studio)
    x_rel, y_rel = value["x"], value["y"]  # Top-left corner position
    w_rel, h_rel = value["width"], value["height"]  # Rectangle dimensions
    rotation_deg = value.get("rotation", 0)  # Rotation angle (default: 0)

    # Convert percentages to pixel coordinates
    x_px = x_rel / 100.0 * org_width
    y_px = y_rel / 100.0 * org_height
    w_px = w_rel / 100.0 * org_width
    h_px = h_rel / 100.0 * org_height

    # Convert rotation from degrees to radians for trigonometric calculations
    rotation_rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rotation_rad), math.sin(rotation_rad)

    # Calculate 4 corners using vector construction
    # p1: top-left corner (starting point)
    p1 = (x_px, y_px)
    
    # p2: top-right corner (move along width vector)
    p2 = (x_px + w_px * cos_r, y_px + w_px * sin_r)
    
    # p4: bottom-left corner (move along height vector perpendicular to width)
    p4 = (x_px - h_px * sin_r, y_px + h_px * cos_r)
    
    # p3: bottom-right corner (vector sum of width and height movements)
    p3 = (p2[0] + (p4[0] - p1[0]), p2[1] + (p4[1] - p1[1]))
    
    # Normalize coordinates to [0, 1] range (YOLO format requirement)
    points_normalized = np.array([
        [p1[0] / org_width, p1[1] / org_height],  # Top-left
        [p2[0] / org_width, p2[1] / org_height],  # Top-right
        [p3[0] / org_width, p3[1] / org_height],  # Bottom-right
        [p4[0] / org_width, p4[1] / org_height]   # Bottom-left
    ])
    
    return points_normalized


def get_points_from_polygon(value: dict, org_width: int, org_height: int) -> np.ndarray:
    """
    Read 4 corners directly from 'polygonlabels' format or 'rectanglelabels' with 'points'.
    
    This handles cases where Label Studio stores the annotation as pre-computed points
    rather than as x, y, width, height with rotation. The points are already in the
    correct corner positions and just need to be normalized.
    
    Args:
        value: Dictionary containing a 'points' list with 4 [x, y] coordinate pairs
        org_width: Original image width in pixels (unused but kept for API consistency)
        org_height: Original image height in pixels (unused but kept for API consistency)
    
    Returns:
        Array of shape (4, 2) with normalized corner coordinates in range [0, 1]
    """
    # Extract pre-computed corner points (already in percentage format)
    points = value["points"]
    
    # Points are already in percentage format, just convert to [0, 1] range
    points_normalized = np.array([
        [p[0] / 100.0, p[1] / 100.0] for p in points
    ])
    
    return points_normalized


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """
    Create YOLO dataset directory structure.
    YOLO requires a specific folder organization: separate images and labels folders,
    each with train and val subdirectories.
    
    Args:
        base_dir: Base directory path for the dataset
    
    Returns:
        Dictionary with keys: 'images_train', 'images_val', 'labels_train', 'labels_val'
        mapping to their respective directory paths
    """
    # Remove existing dataset directory to start fresh (avoids conflicts from previous runs)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Define YOLO directory structure paths
    paths = {
        'images_train': os.path.join(base_dir, "images", "train"),
        'images_val': os.path.join(base_dir, "images", "val"),
        'labels_train': os.path.join(base_dir, "labels", "train"),
        'labels_val': os.path.join(base_dir, "labels", "val")
    }
    
    # Create all directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def load_and_split_annotations(json_path: str, val_split: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and split JSON annotations into train/val sets.
    Filters out cancelled annotations and uses stratified splitting for reproducibility.
    
    Args:
        json_path: Path to Label Studio JSON export file
        val_split: Proportion of data for validation (e.g., 0.2 for 20% validation)
    
    Returns:
        Tuple of (training_data, validation_data) where each is a list of annotation dictionaries
    """
    # Load Label Studio JSON export
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Filter out cancelled annotations (annotations that were marked as invalid/incorrect)
    validated_data = [
        item for item in all_data 
        if not item.get('annotations', [{}])[0].get('was_cancelled', False)
    ]
    
    print(f"Loaded {len(all_data)} total annotations")
    print(f"Using {len(validated_data)} validated annotations (filtered out {len(all_data) - len(validated_data)} cancelled)")
    
    # Split with fixed random seed for reproducibility (same split every run)
    train_data, val_data = train_test_split(
        validated_data, 
        test_size=val_split, 
        random_state=24  # Fixed seed ensures consistent train/val split
    )
    
    print(f"Split: {len(train_data)} training, {len(val_data)} validation")
    
    return train_data, val_data


def process_subset(subset_data: List[Dict], subset_name: str, paths: Dict[str, str], 
                   source_images_dir: str, class_names: List[str]) -> None:
    """
    Process a subset (train or val) and create YOLO annotation files.
    
    This function handles two annotation formats from Label Studio:
    1. Rectangle with rotation (x, y, width, height, rotation) - computed on-the-fly
    2. Pre-computed points (4 corner coordinates) - used directly
    
    For each image, creates a corresponding .txt file with YOLO OBB format:
    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    
    Args:
        subset_data: List of annotation items to process
        subset_name: Name of subset ("train" or "val") for logging
        paths: Dictionary with directory paths from create_directory_structure
        source_images_dir: Directory containing source images to copy
        class_names: List of class names (index in list = class ID)
    
    Returns:
        None
    """
    # Get destination paths for this subset
    image_path_dest = paths[f'images_{subset_name}']
    label_path_dest = paths[f'labels_{subset_name}']
    
    # Create class name to ID mapping (e.g., {'basket-logo': 1, ...})
    class_map = {name: i for i, name in enumerate(class_names)}
    
    # Process each annotated image
    for item in tqdm(subset_data, desc=f"Processing {subset_name} set"):
        # Extract image filename from annotation data
        image_path_source = item['data']['image']
        image_filename = os.path.basename(image_path_source)
        base_name = os.path.splitext(image_filename)[0]  # Filename without extension

        # Create YOLO label file (.txt with same name as image)
        label_filepath = os.path.join(label_path_dest, f"{base_name}.txt")
        
        with open(label_filepath, 'w') as f_out:
            # Get all annotations for this image
            annotations = item['annotations'][0]['result']
            
            # Process each bounding box annotation
            for ann in annotations:
                value = ann.get('value', {})
                org_w, org_h = ann["original_width"], ann["original_height"]
                
                rotated_points = None
                
                # Handle rectanglelabels annotations (both formats)
                if 'rectanglelabels' in value:
                    # Extract class name from annotation
                    class_name = value['rectanglelabels'][0]
                    
                    # Skip unknown classes
                    if class_name not in class_map:
                        continue
                    
                    class_id = class_map[class_name]
                    
                    # Format 1: Rectangle with x, y, width, height, rotation
                    if 'x' in value:
                        rotated_points = get_rotated_points_from_rectangle(value, org_w, org_h)
                    
                    # Format 2: Pre-computed points (4 corners already calculated)
                    elif 'points' in value:
                        rotated_points = get_points_from_polygon(value, org_w, org_h)
                    
                    # Write to YOLO OBB format if points were successfully computed
                    if rotated_points is not None:
                        # Flatten 4x2 array to 8 coordinates and format with 6 decimal places
                        normalized_coords = [f"{coord:.6f}" for p in rotated_points for coord in p]
                        
                        # Write line: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
                        f_out.write(f"{class_id} {' '.join(normalized_coords)}\n")

        # Copy source image to destination directory
        source_image_filepath = os.path.join(source_images_dir, image_filename)
        dest_image_filepath = os.path.join(image_path_dest, image_filename)
        
        # Only copy if source image exists (skip missing images)
        if os.path.exists(source_image_filepath):
            shutil.copy(source_image_filepath, dest_image_filepath)


def create_metadata_files(output_dir: str, class_names: List[str]) -> None:
    """
    Create classes.txt and dataset.yaml files for YOLO training.
    
    classes.txt: Simple list of class names (one per line)
    dataset.yaml: YOLO configuration file with paths and class mappings
    
    Args:
        output_dir: Directory where files will be created (dataset root)
        class_names: List of class names in order (index = class ID)
    
    Returns:
        None
    """
    # Create classes.txt (simple text file with one class name per line)
    classes_filepath = os.path.join(output_dir, "classes.txt")
    with open(classes_filepath, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Created classes file: {classes_filepath}")
    
    # Create YAML configuration for YOLO training
    yaml_filepath = os.path.join(output_dir, "nba_sponsor_dataset.yaml")
    
    # Use absolute path and normalize separators (YOLO prefers forward slashes)
    absolute_path_to_dataset = os.path.abspath(output_dir).replace('\\', '/')
    
    # Build YAML content with dataset paths and class mappings
    yaml_content = f"""# YOLO OBB Dataset Configuration
path: {absolute_path_to_dataset}
train: images/train
val: images/val

# Class dictionary (class_id: class_name)
names:
"""
    # Add each class with its ID
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    # Write YAML configuration file
    with open(yaml_filepath, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML configuration: {yaml_filepath}")


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
    
    Complete pipeline that converts Label Studio annotations to YOLO format:
    1. Creates proper directory structure (images/labels, train/val)
    2. Loads and filters annotations (removes cancelled items)
    3. Splits data into training and validation sets
    4. Converts annotations to YOLO OBB format
    5. Copies images to appropriate directories
    6. Creates configuration files (YAML and classes.txt)
    
    This function supports two annotation formats:
    1. Standard rectangles with rotation (x, y, width, height, rotation)
    2. Pre-computed polygon points (4 corner coordinates)
    
    Args:
        json_file: Path to Label Studio JSON export file
        images_dir: Directory with source images
        output_dir: Output directory for YOLO dataset
        val_split: Validation split ratio (default: 0.2 = 20% validation)
        class_names: List of class names in order (index = class ID)
    
    Returns:
        None
    """
    print("="*60)
    print("YOLO OBB DATASET CONVERSION")
    print("="*60)
    
    # Step 1: Create directory structure
    print("\n[1/5] Creating directory structure...")
    paths = create_directory_structure(output_dir)
    
    # Step 2: Load and split annotations
    print("\n[2/5] Loading and splitting annotations...")
    train_data, val_data = load_and_split_annotations(json_file, val_split)
    
    # Step 3: Process training set
    print("\n[3/5] Processing training set...")
    process_subset(train_data, "train", paths, images_dir, class_names)
    
    # Step 4: Process validation set
    print("\n[4/5] Processing validation set...")
    process_subset(val_data, "val", paths, images_dir, class_names)
    
    # Step 5: Create metadata files
    print("\n[5/5] Creating metadata files...")
    create_metadata_files(output_dir, class_names)
    
    # Display completion summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Dataset location: {output_dir}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Total classes: {len(class_names)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()