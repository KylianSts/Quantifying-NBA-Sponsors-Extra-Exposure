"""
YOLO Auto-Labeling for Label Studio (Incremental Processing)

This script automatically pre-labels unlabeled images using a trained YOLO model
and generates Label Studio-compatible JSON files. It intelligently skips images
that have already been labeled to avoid duplicate work.

Key features:
- Loads a trained YOLO OBB model for automatic predictions
- Identifies unlabeled images by comparing with existing annotations
- Randomly samples from unlabeled images for efficient active learning
- Converts YOLO OBB predictions to Label Studio format
- Generates cloud storage URLs for seamless Label Studio integration
- Supports incremental labeling workflow (only processes new images)
- Handles confidence thresholds for filtering low-quality predictions
"""

import os
import json
import random
from typing import List, Dict, Tuple, Set
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration (trained YOLO model to use for auto-labeling)
MODEL_NAME = "yolov8l-obb_fine_tuned_v4"
MODEL_PATH = f"Models/models_results/modelisation_v4/{MODEL_NAME}/weights/best.pt"

# Directory paths
IMAGES_TO_LABEL_DIR = "Data/images/train_images"  # Directory containing images to auto-label
EXISTING_LABELS_JSON = "Data/json_files/yolo_label_studio.json"  # Existing annotations (to skip)
LABEL_STUDIO_JSON_OUTPUT = f"Data/json_files/prelabeled_tasks_{MODEL_NAME}.json"  # Output file

# Auto-labeling parameters
NUM_RANDOM_IMAGES = 3000  # Maximum number of images to process (None = all unlabeled)
CONFIDENCE_THRESHOLD = 0.0  # Minimum confidence for predictions (0.0 = include all predictions)
GCS_BASE_URL = "gs://yolo_nba_sponsor/train_images"  # Cloud storage base URL for Label Studio


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path: str) -> YOLO:
    """
    Load a trained YOLO model from disk.
    
    Args:
        model_path: Path to the model weights file (.pt)
    
    Returns:
        Loaded YOLO model object ready for inference
    """
    print(f"Loading model from: {model_path}")
    return YOLO(model_path)


def get_all_image_files(directory: str) -> List[str]:
    """
    Get all image files from a directory.
    
    Scans the directory for common image formats (PNG, JPG, JPEG).
    Case-insensitive to handle various file extensions.
    
    Args:
        directory: Path to directory containing images
    
    Returns:
        List of all image filenames (just filenames, not full paths)
    """
    return [
        f for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def get_already_labeled_images(json_path: str) -> Set[str]:
    """
    Extract the set of image filenames that have already been labeled.
    
    Reads an existing Label Studio JSON file and extracts all image filenames
    that have valid (non-cancelled) annotations. This prevents re-labeling
    images that have already been reviewed or annotated.
    
    Args:
        json_path: Path to existing Label Studio JSON file
    
    Returns:
        Set of image filenames that are already labeled (empty set if file doesn't exist)
    """
    # Return empty set if no existing labels file
    if not os.path.exists(json_path):
        print(f"No existing labels found at: {json_path}")
        return set()

    try:
        # Load existing Label Studio annotations
        with open(json_path, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        
        # Extract filenames from valid (non-cancelled) annotations
        # Cancelled annotations are ones marked as invalid/incorrect during review
        labeled_filenames = {
            os.path.basename(item['data']['image']) 
            for item in labeled_data 
            if not item.get('annotations', [{}])[0].get('was_cancelled', False)
        }
        
        print(f"Found {len(labeled_filenames)} already labeled images")
        return labeled_filenames
        
    except (json.JSONDecodeError, KeyError) as e:
        # Return empty set if JSON is malformed
        print(f"Error reading existing labels: {e}")
        return set()


def filter_unlabeled_images(
    all_images: List[str],
    already_labeled: Set[str]
) -> List[str]:
    """
    Filter image list to keep only unlabeled images.
    
    Compares all available images against the set of already-labeled images
    to identify which images still need annotations.
    
    Args:
        all_images: List of all available image filenames
        already_labeled: Set of filenames that are already labeled
    
    Returns:
        List of image filenames that need to be labeled (not in already_labeled set)
    """
    unlabeled = [img for img in all_images if img not in already_labeled]
    print(f"Found {len(unlabeled)} unlabeled images (out of {len(all_images)} total)")
    return unlabeled


def predict_on_image(model: YOLO, image_path: str) -> Tuple:
    """
    Run YOLO prediction on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Full path to the image file
    
    Returns:
        Tuple of (prediction_result, image_width, image_height)
        - prediction_result: YOLO result object containing predictions
        - image_width: Original image width in pixels
        - image_height: Original image height in pixels
    """
    # Run inference with verbose=False to suppress per-image output
    results = model(image_path, verbose=False)
    result = results[0]  # Get first (and only) result
    
    # Extract original image dimensions
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
        obb_predictions: YOLO OBB prediction results (contains xyxyxyxy, conf, cls)
        class_names: Dictionary mapping class IDs to class names (e.g., {0: 'basket-logo'})
        image_width: Width of the original image in pixels
        image_height: Height of the original image in pixels
        confidence_threshold: Minimum confidence to include prediction (0.0 to 1.0)
    
    Returns:
        List of Label Studio annotation dictionaries (one per detected object)
    """
    annotations = []
    
    # Return empty list if no OBB predictions
    if obb_predictions is None:
        return annotations
    
    # Process each detected object
    for i in range(len(obb_predictions.xyxyxyxy)):
        # Extract confidence score
        confidence = obb_predictions.conf[i].item()
        
        # Skip predictions below confidence threshold
        if confidence < confidence_threshold:
            continue
        
        # Extract class information
        class_id = int(obb_predictions.cls[i].item())
        class_name = class_names[class_id]
        
        # Get 4 corner points in pixel coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
        points_pixels = obb_predictions.xyxyxyxy[i].cpu().numpy()
        
        # Convert to percentages (Label Studio format: 0-100 scale)
        points_percent = (points_pixels / [image_width, image_height]) * 100
        points_list = points_percent.tolist()
        
        # Create Label Studio annotation format
        annotation = {
            "original_width": image_width,  # Used for coordinate conversion
            "original_height": image_height,
            "image_rotation": 0,  # No rotation applied
            "value": {
                "points": points_list,  # 4 corner points as percentages
                "rectanglelabels": [class_name]  # Class label for this detection
            },
            "type": "rectanglelabels",  # Annotation type
            "from_name": "label",  # Label Studio field name
            "to_name": "image"  # Target field name
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
    
    A task represents one image with its predicted annotations (pre-labels).
    The task structure includes the image URL and predictions for review.
    
    Args:
        image_filename: Name of the image file (e.g., 'image_001.png')
        annotations: List of annotation dictionaries (predictions from YOLO)
        gcs_base_url: Base URL for cloud storage (e.g., 'gs://bucket/folder')
        model_version: Version string for the model (for tracking which model generated predictions)
    
    Returns:
        Label Studio task dictionary ready for JSON export
    """
    return {
        "data": {
            "image": f"{gcs_base_url}/{image_filename}"  # Full cloud storage URL
        },
        "predictions": [{
            "model_version": model_version,  # Track which model made these predictions
            "score": 0.0,  # Overall task score (not used in this workflow)
            "result": annotations  # List of predicted bounding boxes
        }]
    }


def save_label_studio_json(tasks: List[Dict], output_path: str) -> None:
    """
    Save Label Studio tasks to JSON file.
    
    Writes the tasks in JSON format with proper indentation for readability.
    This file can be directly imported into Label Studio.
    
    Args:
        tasks: List of Label Studio task dictionaries
        output_path: Path where JSON file will be saved
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write JSON with indentation for human readability
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Saved {len(tasks)} tasks to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def auto_label_new_images(
    model_path: str = MODEL_PATH,
    images_dir: str = IMAGES_TO_LABEL_DIR,
    output_json: str = LABEL_STUDIO_JSON_OUTPUT,
    num_samples: int = NUM_RANDOM_IMAGES,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    gcs_base_url: str = GCS_BASE_URL,
    existing_labels_json: str = EXISTING_LABELS_JSON
) -> None:
    """
    Automatically pre-label only NEW images that haven't been labeled yet.
    
    This function performs incremental auto-labeling by:
    1. Loading a trained YOLO model
    2. Finding all images in the directory
    3. Filtering out images that are already labeled (in existing JSON)
    4. Randomly sampling from unlabeled images (for efficient active learning)
    5. Running predictions on each new image
    6. Converting predictions to Label Studio format
    7. Saving results as JSON for import into Label Studio
    
    This prevents duplicate work by skipping images that have already been
    annotated or reviewed in previous labeling sessions. Random sampling
    enables active learning by selecting diverse images for review.
    
    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        images_dir: Directory containing images to label
        output_json: Output path for Label Studio JSON file
        num_samples: Maximum number of images to process (None = process all unlabeled)
        confidence_threshold: Minimum confidence for including predictions (0.0 to 1.0)
        gcs_base_url: Base URL for cloud storage image paths (for Label Studio)
        existing_labels_json: Path to existing Label Studio JSON to check for duplicates
    
    Returns:
        None
    """
    print("="*60)
    print("AUTO-LABELING NEW IMAGES")
    print("="*60)
    
    # Step 1: Load the trained model
    print("\n[1/7] Loading trained YOLO model...")
    model = load_model(model_path)
    
    # Step 2: Get all available images in the directory
    print("\n[2/7] Scanning for images...")
    all_image_files = get_all_image_files(images_dir)
    print(f"Total images found: {len(all_image_files)}")
    
    # Step 3: Find images that are already labeled
    print("\n[3/7] Checking for already labeled images...")
    already_labeled = get_already_labeled_images(existing_labels_json)
    
    # Step 4: Filter to get only unlabeled images
    print("\n[4/7] Filtering unlabeled images...")
    unlabeled_images = filter_unlabeled_images(all_image_files, already_labeled)
    
    # Step 5: Sample from unlabeled images if needed (for active learning)
    print("\n[5/7] Selecting images to process...")
    if num_samples and len(unlabeled_images) > num_samples:
        images_to_process = random.sample(unlabeled_images, num_samples)
        print(f"Randomly sampled {num_samples} images from {len(unlabeled_images)} unlabeled images")
    else:
        images_to_process = unlabeled_images
        print(f"Processing all {len(unlabeled_images)} unlabeled images")
        
    # Exit early if no new images to process
    if not images_to_process:
        print("\nNo new images to label. All images have already been processed.")
        return

    # Step 6: Process each image and create Label Studio tasks
    print(f"\n[6/7] Running predictions on {len(images_to_process)} images...")
    label_studio_tasks = []
    
    for image_filename in tqdm(images_to_process, desc="Auto-labeling images"):
        # Build full path to image
        image_path = os.path.join(images_dir, image_filename)
        
        # Run YOLO prediction
        result, img_width, img_height = predict_on_image(model, image_path)
        
        # Convert YOLO predictions to Label Studio format
        annotations = convert_obb_to_label_studio(
            obb_predictions=result.obb,  # OBB predictions (oriented bounding boxes)
            class_names=model.names,  # Class ID to name mapping
            image_width=img_width,
            image_height=img_height,
            confidence_threshold=confidence_threshold
        )
        
        # Create Label Studio task with predictions
        task = create_label_studio_task(
            image_filename=image_filename,
            annotations=annotations,
            gcs_base_url=gcs_base_url,
            model_version=MODEL_NAME
        )
        
        label_studio_tasks.append(task)
    
    # Step 7: Save all tasks to JSON file for Label Studio import
    print("\n[7/7] Saving Label Studio tasks...")
    save_label_studio_json(label_studio_tasks, output_json)
    
    # Display completion summary
    print("\n" + "="*60)
    print("AUTO-LABELING COMPLETE")
    print("="*60)
    print(f"Processed images: {len(images_to_process)}")
    print(f"Skipped (already labeled): {len(already_labeled)}")
    print(f"Output file: {output_json}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*60 + "\n")


if __name__ == "__main__":
    auto_label_new_images()