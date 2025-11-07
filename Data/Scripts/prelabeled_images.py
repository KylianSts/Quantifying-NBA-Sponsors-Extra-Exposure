import os
import json
import random
from typing import List, Dict, Tuple
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Name of the YOLO model to use 
MODEL_NAME = "yolov8m-obb_fine_tuned_v1"

# Path to trained YOLO model weights 
MODEL_PATH = f"Models/NBA_Sponsor_Detection/{MODEL_NAME}/weights/best.pt"

# Directory containing images to auto-label
IMAGES_TO_LABEL_DIR = "Data/train_images"

# Output JSON file for Label Studio import
LABEL_STUDIO_JSON_OUTPUT = f"Data/prelabeled_tasks_{MODEL_NAME}.json"

# Number of random images to process from the directory
NUM_RANDOM_IMAGES = 3000

# Only detections above this threshold will be included
CONFIDENCE_THRESHOLD = 0

# Google Cloud Storage base URL for Label Studio image paths
GCS_BASE_URL = "gs://yolo_nba_sponsor/train_images"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path: str) -> YOLO:
    """
    Load a trained YOLO model from disk.
    
    Args:
        model_path: Path to the model weights file (.pt)
    
    Returns:
        Loaded YOLO model object
    
    Raises:
        Exception: If model cannot be loaded
    """
    return YOLO(model_path)


def get_image_files(directory: str, num_samples: int = None) -> List[str]:
    """
    Get list of image files from directory, optionally sampling randomly.
    
    Scans the directory for common image formats (PNG, JPG, JPEG) and
    returns either all images or a random sample of specified size.
    
    Args:
        directory: Path to directory containing images
        num_samples: Number of random images to sample (None = all images)
    
    Returns:
        List of image filenames to process
    """
    # Get all image files from directory
    all_images = [
        f for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Return random sample if requested and available
    if num_samples and len(all_images) > num_samples:
        return random.sample(all_images, num_samples)
    
    return all_images


def predict_on_image(model: YOLO, image_path: str) -> Tuple:
    """
    Run YOLO prediction on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Full path to the image file
    
    Returns:
        Tuple of (predictions, image_width, image_height)
    """
    # Run inference without verbose output
    results = model(image_path, verbose=False)
    result = results[0]
    
    # Get original image dimensions
    height, width = result.orig_shape
    
    return result, width, height


def convert_obb_to_label_studio(
    obb_predictions,
    class_names: Dict[int, str],
    image_width: int,
    image_height: int,
    confidence_threshold: float
) -> List[Dict]:
    """
    Convert YOLO OBB predictions to Label Studio annotation format.
    
    YOLO OBB format uses 4 corner points (xyxyxyxy) in pixel coordinates.
    Label Studio expects these points as percentages of image dimensions.
    
    Args:
        obb_predictions: YOLO OBB prediction results
        class_names: Dictionary mapping class IDs to class names
        image_width: Width of the original image in pixels
        image_height: Height of the original image in pixels
        confidence_threshold: Minimum confidence to include prediction
    
    Returns:
        List of Label Studio annotation dictionaries
    """
    annotations = []
    
    # Check if any OBB detections were made
    if obb_predictions is None:
        return annotations
    
    # Process each detection
    for i in range(len(obb_predictions.xyxyxyxy)):
        confidence = obb_predictions.conf[i].item()
        
        # Skip low-confidence detections
        if confidence < confidence_threshold:
            continue
        
        # Get class information
        class_id = int(obb_predictions.cls[i].item())
        class_name = class_names[class_id]
        
        # Get 4 corner points in pixel coordinates
        # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points_pixels = obb_predictions.xyxyxyxy[i].cpu().numpy()
        
        # Convert to percentages (Label Studio format)
        points_percent = (points_pixels / [image_width, image_height]) * 100
        points_list = points_percent.tolist()
        
        # Create Label Studio annotation format
        annotation = {
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "points": points_list,
                "rectanglelabels": [class_name]
            },
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image"
        }
        
        annotations.append(annotation)
    
    return annotations


def create_label_studio_task(
    image_filename: str,
    annotations: List[Dict],
    gcs_base_url: str,
    model_version: str = "yolov8m-obb-v1"
) -> Dict:
    """
    Create a Label Studio task dictionary for a single image.
    
    A task contains the image reference and pre-annotations (predictions)
    that will be loaded into Label Studio for manual review/correction.
    
    Args:
        image_filename: Name of the image file
        annotations: List of annotation dictionaries
        gcs_base_url: Base URL for cloud storage (e.g., gs://bucket/path)
        model_version: Version string for the model that generated predictions
    
    Returns:
        Label Studio task dictionary
    """
    task = {
        "data": {
            "image": f"{gcs_base_url}/{image_filename}"
        },
        "predictions": [{
            "model_version": model_version,
            "score": 0.0,
            "result": annotations
        }]
    }
    
    return task


def save_label_studio_json(tasks: List[Dict], output_path: str) -> None:
    """
    Save Label Studio tasks to JSON file.
    
    Args:
        tasks: List of Label Studio task dictionaries
        output_path: Path where JSON file will be saved
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def auto_label_images(
    model_path: str = MODEL_PATH,
    images_dir: str = IMAGES_TO_LABEL_DIR,
    output_json: str = LABEL_STUDIO_JSON_OUTPUT,
    num_samples: int = NUM_RANDOM_IMAGES,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    gcs_base_url: str = GCS_BASE_URL
) -> None:
    """
    Automatically pre-label images using a trained YOLO model.
    
    This function performs the complete auto-labeling pipeline:
    1. Loads a trained YOLO model
    2. Randomly samples images from a directory (or processes all)
    3. Runs predictions on each image
    4. Converts predictions to Label Studio format
    5. Saves results as JSON for import into Label Studio
    
    The output JSON can be directly imported into Label Studio, where the
    predictions will appear as pre-annotations that can be reviewed and
    corrected by human annotators.
    
    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        images_dir: Directory containing images to label
        output_json: Output path for Label Studio JSON file
        num_samples: Number of random images to process (None = all)
        confidence_threshold: Minimum confidence for including predictions
        gcs_base_url: Base URL for cloud storage image paths
    """
    # Step 1: Load the trained model
    model = load_model(model_path)
    
    # Step 2: Get list of images to process (random sample or all)
    image_files = get_image_files(images_dir, num_samples)
    
    # Step 3: Process each image and create Label Studio tasks
    label_studio_tasks = []
    
    for image_filename in tqdm(image_files, desc="Auto-labeling images"):
        # Get full path to image
        image_path = os.path.join(images_dir, image_filename)
        
        # Run prediction
        result, img_width, img_height = predict_on_image(model, image_path)
        
        # Convert predictions to Label Studio format
        annotations = convert_obb_to_label_studio(
            obb_predictions=result.obb,
            class_names=model.names,
            image_width=img_width,
            image_height=img_height,
            confidence_threshold=confidence_threshold
        )
        
        # Create task for this image
        task = create_label_studio_task(
            image_filename=image_filename,
            annotations=annotations,
            gcs_base_url=gcs_base_url
        )
        
        label_studio_tasks.append(task)
    
    # Step 4: Save all tasks to JSON file
    save_label_studio_json(label_studio_tasks, output_json)


if __name__ == "__main__":
    auto_label_images()