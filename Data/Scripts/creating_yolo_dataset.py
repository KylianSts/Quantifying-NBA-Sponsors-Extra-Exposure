import os
import json
import shutil
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

def get_rotated_points(x: float, y: float, width: float, height: float, angle_deg: float) -> np.ndarray:
    """
    Calculate the 4 corners of a rotated rectangle from Label Studio data.
    
    Label Studio stores rectangles with rotation as (x, y, width, height, rotation).
    This function converts that representation into 4 corner points in normalized
    coordinates [0, 1], which is the format required by YOLO OBB.
    
    Args:
        x: X coordinate of the rectangle top-left corner (percentage 0-100)
        y: Y coordinate of the rectangle top-left corner (percentage 0-100)
        width: Width of the rectangle (percentage 0-100)
        height: Height of the rectangle (percentage 0-100)
        angle_deg: Rotation angle in degrees (clockwise)
    
    Returns:
        Array of shape (4, 2) containing the 4 corner points in normalized coordinates
        Order: top-left, top-right, bottom-right, bottom-left
    """
    # Normalize coordinates from percentage to [0, 1] range
    x_norm, y_norm = x / 100.0, y / 100.0
    width_norm, height_norm = width / 100.0, height / 100.0
    
    # Convert angle to radians (negative because Label Studio uses clockwise rotation)
    angle_rad = -np.deg2rad(angle_deg)
    
    # Calculate center point of the rectangle
    center_x = x_norm + width_norm / 2
    center_y = y_norm + height_norm / 2
    
    # Calculate half-dimensions
    dx, dy = width_norm / 2, height_norm / 2
    
    # Define the 4 corners relative to center (before rotation)
    points_before = np.array([
        [-dx, -dy],  # Top-left
        [+dx, -dy],  # Top-right
        [+dx, +dy],  # Bottom-right
        [-dx, +dy]   # Bottom-left
    ])
    
    # Create rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Apply rotation and translate to final position
    rotated_points = np.dot(points_before, rotation_matrix.T)
    final_points = rotated_points + [center_x, center_y]
    
    return final_points


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """
    Create the YOLO dataset directory structure.
    
    Creates the standard YOLO directory layout:
    - images/train: Training images
    - images/val: Validation images
    - labels/train: Training labels (OBB format)
    - labels/val: Validation labels (OBB format)
    
    If the base directory already exists, it will be removed and recreated
    to ensure a clean dataset structure.
    
    Args:
        base_dir: Base directory path for the dataset
    
    Returns:
        Dictionary with keys: 'images_train', 'images_val', 'labels_train', 'labels_val'
        containing the full paths to each subdirectory
    """
    # Remove existing directory if present to start fresh
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Define all required subdirectories
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
    Load annotations from Label Studio JSON and split into train/val sets.
    
    This function performs three operations:
    1. Loads the JSON file containing all annotations
    2. Filters out cancelled/rejected annotations
    3. Splits the validated data into training and validation sets
    
    Args:
        json_path: Path to the Label Studio JSON export file
        val_split: Proportion of data to use for validation (e.g., 0.2 for 20%)
    
    Returns:
        Tuple of (training_data, validation_data)
    """
    # Load all annotations from JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Filter out cancelled annotations (those marked as invalid in Label Studio)
    # Only keep tasks where the annotation was not cancelled
    validated_data = [
        item for item in all_data 
        if not item.get('annotations', [{}])[0].get('was_cancelled', False)
    ]
    
    # Split into train and validation sets with fixed random seed for reproducibility
    train_data, val_data = train_test_split(
        validated_data, 
        test_size=val_split, 
        random_state=24
    )
    
    return train_data, val_data


def process_subset(subset_data: List[Dict], subset_name: str, paths: Dict[str, str], 
                   source_images_dir: str, class_names: List[str]) -> None:
    """
    Process a subset of data (train or val) and create corresponding YOLO files.
    
    For each annotation item, this function:
    1. Creates a YOLO OBB format label file with all annotations
    2. Copies the corresponding image file to the appropriate directory
    
    The YOLO OBB format is: class_id x1 y1 x2 y2 x3 y3 x4 y4
    where (x1,y1) through (x4,y4) are the 4 corners of the rotated bounding box
    in normalized coordinates [0, 1].
    
    Args:
        subset_data: List of annotation items to process
        subset_name: Name of the subset ("train" or "val")
        paths: Dictionary containing paths to all subdirectories
        source_images_dir: Directory containing the original source images
        class_names: List of class names (used to create class_id mapping)
    """
    # Get paths for this subset
    image_path_dest = paths[f'images_{subset_name}']
    label_path_dest = paths[f'labels_{subset_name}']
    
    # Create class name to ID mapping
    class_map = {name: i for i, name in enumerate(class_names)}
    
    # Process each annotation item with progress bar
    for item in tqdm(subset_data, desc=f"Processing {subset_name} set"):
        # Extract image filename from the annotation data
        image_path_source = item['data']['image']
        image_filename = os.path.basename(image_path_source)
        base_name = os.path.splitext(image_filename)[0]

        # Create YOLO format label file
        label_filepath = os.path.join(label_path_dest, f"{base_name}.txt")
        with open(label_filepath, 'w') as f_out:
            # Get all annotations for this image
            annotations = item['annotations'][0]['result']
            
            # Process each annotation (each logo in the image)
            for ann in annotations:
                value = ann.get('value', {})
                
                # Skip if not a rectangle annotation
                if 'rectanglelabels' not in value:
                    continue
                
                # Get class name and skip if not in our class list
                class_name = value['rectanglelabels'][0]
                if class_name not in class_map:
                    continue
                
                class_id = class_map[class_name]
                
                # Convert rotated rectangle to 4 corner points
                rotated_points = get_rotated_points(
                    value['x'], 
                    value['y'], 
                    value['width'], 
                    value['height'], 
                    value['rotation']
                )
                
                # Format coordinates as space-separated values with 6 decimal places
                normalized_coords = [f"{coord:.6f}" for p in rotated_points for coord in p]
                
                # Write in YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                f_out.write(f"{class_id} {' '.join(normalized_coords)}\n")

        # Copy source image to destination directory
        source_image_filepath = os.path.join(source_images_dir, image_filename)
        dest_image_filepath = os.path.join(image_path_dest, image_filename)
        
        if os.path.exists(source_image_filepath):
            shutil.copy(source_image_filepath, dest_image_filepath)


def create_metadata_files(output_dir: str, class_names: List[str]) -> None:
    """
    Create metadata files required for YOLO training.
    
    Creates two files:
    1. classes.txt: Simple text file with one class name per line
    2. dataset.yaml: YAML configuration file with dataset paths and class information
    
    Args:
        output_dir: Directory where the metadata files will be created
        class_names: List of class names in the correct order
    """
    # Create classes.txt file
    classes_filepath = os.path.join(output_dir, "classes.txt")
    with open(classes_filepath, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Create YAML configuration file
    yaml_filepath = os.path.join(output_dir, "nba_sponsor_dataset.yaml")
    
    # Get absolute path and convert to forward slashes (works on all platforms)
    absolute_path_to_dataset = os.path.abspath(output_dir).replace('\\', '/')
    
    # Build YAML content
    yaml_content = f"""# YOLO OBB Dataset Configuration
# Auto-generated configuration file for YOLOv8/YOLOv11 training

path: {absolute_path_to_dataset}
train: images/train
val: images/val

# Class dictionary (class_id: class_name)
names:
"""
    # Add each class with its ID
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_filepath, 'w') as f:
        f.write(yaml_content)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_yolo_obb_dataset(
    json_file: str = JSON_SOURCE_FILE,
    images_dir: str = SOURCE_IMAGES_DIR,
    output_dir: str = YOLO_DATASET_DIR,
    val_split: float = VALIDATION_SPLIT_RATIO,
    class_names: List[str] = CLASS_NAMES,
) -> None:
    """
    Create a YOLO OBB dataset from Label Studio JSON export.
    
    This is the main entry point that orchestrates the entire dataset creation process.
    It converts Label Studio annotations with rotated rectangles into the YOLO OBB
    format, which uses 4 corner points to represent oriented bounding boxes.
    
    The process involves:
    1. Creating the standard YOLO directory structure
    2. Loading annotations from Label Studio JSON and splitting into train/val
    3. Converting annotations to YOLO OBB format and copying images
    4. Creating metadata files (classes.txt and YAML config)
    
    Args:
        json_file: Path to Label Studio JSON export file
        images_dir: Directory containing the source images
        output_dir: Output directory for the YOLO dataset
        val_split: Proportion of data to use for validation (default: 0.2 = 20%)
        class_names: List of class names in order (index becomes class_id)
    """
    # Step 1: Create directory structure
    paths = create_directory_structure(output_dir)
    
    # Step 2: Load annotations and split into train/val sets
    train_data, val_data = load_and_split_annotations(json_file, val_split)
    
    # Step 3: Process both training and validation sets
    process_subset(train_data, "train", paths, images_dir, class_names)
    process_subset(val_data, "val", paths, images_dir, class_names)
    
    # Step 4: Create metadata files (classes.txt and YAML config)
    create_metadata_files(output_dir, class_names)


if __name__ == "__main__":
    create_yolo_obb_dataset()