import os
import json
import random
from typing import List, Dict, Tuple, Set
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = "yolov8m-obb_fine_tuned_v2"
MODEL_PATH = f"Models/models_results/{MODEL_NAME}/weights/best.pt"

# Directory paths
IMAGES_TO_LABEL_DIR = "Data/images/train_images"
EXISTING_LABELS_JSON = "Data/json_files/yolo_label_studio.json"
LABEL_STUDIO_JSON_OUTPUT = f"Data/json_files/prelabeled_tasks_{MODEL_NAME}.json"

# Auto-labeling parameters
NUM_RANDOM_IMAGES = 3000          # Maximum number of images to process
CONFIDENCE_THRESHOLD = 0.0        # Minimum confidence for predictions (0.0 = all predictions)
GCS_BASE_URL = "gs://yolo_nba_sponsor/train_images"  # Cloud storage base URL


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
    """
    return YOLO(model_path)


def get_all_image_files(directory: str) -> List[str]:
    """
    Get all image files from a directory.
    
    Scans the directory for common image formats (PNG, JPG, JPEG).
    
    Args:
        directory: Path to directory containing images
    
    Returns:
        List of all image filenames
    """
    return [
        f for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def get_already_labeled_images(json_path: str) -> Set[str]:
    """
    Extract the set of image filenames that have already been labeled.
    
    Reads an existing Label Studio JSON file and extracts all image filenames
    that have valid (non-cancelled) annotations.
    
    Args:
        json_path: Path to existing Label Studio JSON file
    
    Returns:
        Set of image filenames that are already labeled
    """
    if not os.path.exists(json_path):
        return set()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        
        # Extract filenames from valid (non-cancelled) annotations
        labeled_filenames = {
            os.path.basename(item['data']['image']) 
            for item in labeled_data 
            if not item.get('annotations', [{}])[0].get('was_cancelled', False)
        }
        
        return labeled_filenames
        
    except (json.JSONDecodeError, KeyError) as e:
        return set()


def filter_unlabeled_images(
    all_images: List[str],
    already_labeled: Set[str]
) -> List[str]:
    """
    Filter image list to keep only unlabeled images.
    
    Args:
        all_images: List of all available image filenames
        already_labeled: Set of filenames that are already labeled
    
    Returns:
        List of image filenames that need to be labeled
    """
    return [img for img in all_images if img not in already_labeled]


def predict_on_image(model: YOLO, image_path: str) -> Tuple:
    """
    Run YOLO prediction on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Full path to the image file
    
    Returns:
        Tuple of (predictions, image_width, image_height)
    """
    results = model(image_path, verbose=False)
    result = results[0]
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
    
    if obb_predictions is None:
        return annotations
    
    for i in range(len(obb_predictions.xyxyxyxy)):
        confidence = obb_predictions.conf[i].item()
        
        if confidence < confidence_threshold:
            continue
        
        class_id = int(obb_predictions.cls[i].item())
        class_name = class_names[class_id]
        
        # Get 4 corner points in pixel coordinates
        points_pixels = obb_predictions.xyxyxyxy[i].cpu().numpy()
        
        # Convert to percentages (Label Studio format)
        points_percent = (points_pixels / [image_width, image_height]) * 100
        points_list = points_percent.tolist()
        
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
    
    Args:
        image_filename: Name of the image file
        annotations: List of annotation dictionaries
        gcs_base_url: Base URL for cloud storage
        model_version: Version string for the model
    
    Returns:
        Label Studio task dictionary
    """
    return {
        "data": {
            "image": f"{gcs_base_url}/{image_filename}"
        },
        "predictions": [{
            "model_version": model_version,
            "score": 0.0,
            "result": annotations
        }]
    }


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
    4. Randomly sampling from unlabeled images (or processing all)
    5. Running predictions on each new image
    6. Converting predictions to Label Studio format
    7. Saving results as JSON for import into Label Studio
    
    This prevents duplicate work by skipping images that have already been
    annotated or reviewed in previous labeling sessions.
    
    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        images_dir: Directory containing images to label
        output_json: Output path for Label Studio JSON file
        num_samples: Maximum number of images to process (None = all)
        confidence_threshold: Minimum confidence for including predictions
        gcs_base_url: Base URL for cloud storage image paths
        existing_labels_json: Path to existing Label Studio JSON to check for duplicates
    """
    # Step 1: Load the trained model
    model = load_model(model_path)
    
    # Step 2: Get all available images
    all_image_files = get_all_image_files(images_dir)
    
    # Step 3: Find images that are already labeled
    already_labeled = get_already_labeled_images(existing_labels_json)
    
    # Step 4: Filter to get only unlabeled images
    unlabeled_images = filter_unlabeled_images(all_image_files, already_labeled)
    
    # Step 5: Sample from unlabeled images if needed
    if num_samples and len(unlabeled_images) > num_samples:
        images_to_process = random.sample(unlabeled_images, num_samples)
    else:
        images_to_process = unlabeled_images
        
    # Exit early if no new images to process
    if not images_to_process:
        return

    # Step 6: Process each image and create Label Studio tasks
    label_studio_tasks = []
    
    for image_filename in tqdm(images_to_process, desc="Auto-labeling new images"):
        image_path = os.path.join(images_dir, image_filename)
        
        result, img_width, img_height = predict_on_image(model, image_path)
        
        annotations = convert_obb_to_label_studio(
            obb_predictions=result.obb,
            class_names=model.names,
            image_width=img_width,
            image_height=img_height,
            confidence_threshold=confidence_threshold
        )
        
        task = create_label_studio_task(
            image_filename=image_filename,
            annotations=annotations,
            gcs_base_url=gcs_base_url,
            model_version=MODEL_NAME
        )
        
        label_studio_tasks.append(task)
    
    # Step 7: Save all tasks to JSON file
    save_label_studio_json(label_studio_tasks, output_json)


if __name__ == "__main__":
    auto_label_new_images()