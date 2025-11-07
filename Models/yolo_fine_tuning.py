import torch
from ultralytics import YOLO

# Model selection
MODEL_TO_USE = 'Models/yolo/yolov8m-obb.pt'

# Path to dataset configuration file (created by create_yolo_dataset.py)
DATASET_YAML_PATH = 'Data/yolo_dataset/nba_sponsor_dataset.yaml'

# Training hyperparameters
EPOCHS = 1           
IMAGE_SIZE = 640       
BATCH_SIZE = 8  

# Output directories for training results
PROJECT_NAME = 'Models/NBA_Sponsor_Detection'
EXPERIMENT_NAME = 'yolov8m-obb_fine_tuned_v1'

def train_yolo_model(
    model_path: str,
    dataset_yaml: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    project_name: str,
    experiment_name: str
) -> YOLO:
    """
    Train a YOLO OBB model on the NBA sponsor detection dataset.
    
    This function handles the complete training process:
    1. Loads the pre-trained YOLO model
    2. Configures training parameters
    3. Trains the model on the custom dataset
    4. Saves training results and model checkpoints
    
    The training process will:
    - Fine-tune the model on oriented bounding boxes (OBB)
    - Save best and last weights automatically
    - Generate training metrics and visualizations
    - Create validation predictions for inspection
    
    Args:
        model_path: Path to pre-trained YOLO model weights (.pt file)
        dataset_yaml: Path to dataset configuration YAML file
        epochs: Number of training epochs to run
        image_size: Size to resize images (e.g., 640 for 640x640)
        batch_size: Number of images per batch (adjust based on GPU memory)
        project_name: Name of the project folder for organizing results
        experiment_name: Name of this specific experiment/run
    
    Returns:
        Trained YOLO model object
    """
    # Load pre-trained model
    model = YOLO(model_path)
    
    model.train(
        data=dataset_yaml,           # Dataset configuration
        epochs=epochs,               # Training duration
        imgsz=image_size,            # Input image size
        batch=batch_size,            # Batch size
        project=project_name,        # Project folder
        name=experiment_name,        # Experiment name
        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    return model


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main training pipeline for YOLO OBB model.
    
    Orchestrates the training process:
    1. Checks GPU availability
    2. Trains the model with configured parameters
    3. Saves results to the specified project directory
    """
    # Check GPU availability
    gpu_available =  torch.cuda.is_available()
    
    # Train the model
    train_yolo_model(
        model_path=MODEL_TO_USE,
        dataset_yaml=DATASET_YAML_PATH,
        epochs=EPOCHS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME
    )


if __name__ == '__main__':
    main()