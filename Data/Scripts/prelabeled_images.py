"""
YOLO Auto-Labeling for Label Studio

Automatically pre-labels unlabeled images using a trained YOLO model
and generates Label Studio-compatible JSON files.

Key features:
- Loads a trained YOLO OBB model for automatic predictions
- Identifies unlabeled images by comparing with existing annotations
- Randomly samples from unlabeled images for efficient active learning
- Converts YOLO OBB predictions to Label Studio format
- Generates cloud storage URLs for seamless Label Studio integration
- Supports incremental labeling workflow (only processes new images)
"""

import os
import json
import random
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "yolo11s-obb_fine_tuned_v10_1280"
MODEL_PATH = f"Models/models_results/modelisation_v10/{MODEL_NAME}/weights/best.pt"

IMAGES_TO_LABEL_DIR = "Data/images/train_images_quality_2"
EXISTING_LABELS_JSON = "Data/json_files/yolo_label_studio.json"
LABEL_STUDIO_JSON_OUTPUT = f"Data/json_files/prelabeled_tasks_{MODEL_NAME}_quality_10.json"

NUM_RANDOM_IMAGES = None  # None = process all unlabeled images
CONFIDENCE_THRESHOLD = 0.0  # Minimum confidence for predictions
GCS_BASE_URL = "gs://yolo_nba_sponsor/train_images_quality"


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
    
    Performs incremental auto-labeling by filtering out already-labeled images,
    running predictions, and converting to Label Studio format.
    """
    print("=" * 60)
    print("AUTO-LABELING NEW IMAGES")
    print("=" * 60)
    
    # Load the trained YOLO model
    print("\n[1/5] Loading trained YOLO model...")
    print(f"Model path: {model_path}")
    model = YOLO(model_path)
    
    # Get all image files from directory
    print("\n[2/5] Scanning for images...")
    all_image_files = [
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Total images found: {len(all_image_files)}")
    
    # Load existing labels to identify already-labeled images
    print("\n[3/5] Checking for already labeled images...")
    already_labeled = set()
    if os.path.exists(existing_labels_json):
        try:
            with open(existing_labels_json, 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
            
            already_labeled = {
                os.path.basename(item['data']['image']) 
                for item in labeled_data 
                if not item.get('annotations', [{}])[0].get('was_cancelled', False)
            }
            print(f"Found {len(already_labeled)} already labeled images")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading existing labels: {e}")
    else:
        print(f"No existing labels found at: {existing_labels_json}")
    
    # Filter to get only unlabeled images
    unlabeled_images = [img for img in all_image_files if img not in already_labeled]
    print(f"Found {len(unlabeled_images)} unlabeled images (out of {len(all_image_files)} total)")
    
    # Sample from unlabeled images if needed
    if num_samples and len(unlabeled_images) > num_samples:
        images_to_process = random.sample(unlabeled_images, num_samples)
        print(f"Randomly sampled {num_samples} images from {len(unlabeled_images)} unlabeled images")
    else:
        images_to_process = unlabeled_images
        print(f"Processing all {len(unlabeled_images)} unlabeled images")
    
    if not images_to_process:
        print("\nNo new images to label. All images have already been processed.")
        return
    
    # Process each image and create Label Studio tasks
    print(f"\n[4/5] Running predictions on {len(images_to_process)} images...")
    label_studio_tasks = []
    
    for image_filename in tqdm(images_to_process, desc="Auto-labeling images"):
        image_path = os.path.join(images_dir, image_filename)
        
        # Run YOLO prediction
        results = model(image_path, verbose=False)
        result = results[0]
        height, width = result.orig_shape
        
        # Convert YOLO OBB predictions to Label Studio format
        annotations = []
        if result.obb is not None:
            for i in range(len(result.obb.xyxyxyxy)):
                confidence = result.obb.conf[i].item()
                
                if confidence < confidence_threshold:
                    continue
                
                class_id = int(result.obb.cls[i].item())
                class_name = model.names[class_id]
                
                # Get 4 corner points and convert to percentages
                points_pixels = result.obb.xyxyxyxy[i].cpu().numpy()
                points_percent = (points_pixels / [width, height]) * 100
                points_list = points_percent.tolist()
                
                annotations.append({
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "points": points_list,
                        "rectanglelabels": [class_name]
                    },
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image"
                })
        
        # Create Label Studio task
        task = {
            "data": {
                "image": f"{gcs_base_url}/{image_filename}"
            },
            "predictions": [{
                "model_version": MODEL_NAME,
                "score": 0.0,
                "result": annotations
            }]
        }
        
        label_studio_tasks.append(task)
    
    # Save all tasks to JSON file
    print("\n[5/5] Saving Label Studio tasks...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(label_studio_tasks, f, indent=2)
    
    print(f"Saved {len(label_studio_tasks)} tasks to: {output_json}")
    
    # Display completion summary
    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE")
    print("=" * 60)
    print(f"Processed images: {len(images_to_process)}")
    print(f"Skipped (already labeled): {len(already_labeled)}")
    print(f"Output file: {output_json}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    auto_label_new_images()