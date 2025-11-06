import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def create_yolo_dataset(
    yolo_base_dir: str = "Data/yolo_dataset",
    source_images_dir: str = "Data/train_images",
    validation_split_ratio: float = 0.2,
    image_ext: str = ".png"
) -> Tuple[List[str], List[str]]:
    """
    Create a YOLO dataset directory structure and distribute files into
    training and validation subsets based on label files.

    Args:
        yolo_base_dir: Base directory for YOLO dataset (default: 'Data/yolo_dataset')
        source_images_dir: Directory containing all source images (default: 'Data/train_images')
        validation_split_ratio: Fraction of data used for validation (default: 0.2)
        image_ext: Image file extension to match (default: '.png')

    Returns:
        Tuple of two lists: (train_label_files, val_label_files)
    """
    # Define paths for images and labels ---
    labels_dir = os.path.join(yolo_base_dir, "labels")
    images_train_dir = os.path.join(yolo_base_dir, "images", "train")
    images_val_dir = os.path.join(yolo_base_dir, "images", "val")
    labels_train_dir = os.path.join(yolo_base_dir, "labels", "train")
    labels_val_dir = os.path.join(yolo_base_dir, "labels", "val")

    # Create necessary directories ---
    for path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(path, exist_ok=True)

    # Retrieve all label files (.txt) ---
    all_label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    all_label_names = [os.path.basename(p) for p in all_label_files]

    if not all_label_names:
        return [], []  # No label files found

    # Split into training and validation subsets ---
    train_labels, val_labels = train_test_split(
        all_label_names,
        test_size=validation_split_ratio,
        random_state=24
    )

    # Move label files and copy corresponding images ---
    def process_files(file_list: List[str], dest_subset: str):
        for label_file in file_list:
            base_name = os.path.splitext(label_file)[0]
            src_label = os.path.join(labels_dir, label_file)
            src_image = os.path.join(source_images_dir, f"{base_name}{image_ext}")
            dst_label = os.path.join(yolo_base_dir, "labels", dest_subset, label_file)
            dst_image = os.path.join(yolo_base_dir, "images", dest_subset, f"{base_name}{image_ext}")

            if os.path.exists(src_image):
                shutil.move(src_label, dst_label)
                shutil.copy(src_image, dst_image)

    process_files(train_labels, "train")
    process_files(val_labels, "val")

    return train_labels, val_labels


if __name__ == "__main__":

    train_files, val_files = create_yolo_dataset()
    print(f"Train labels moved: {len(train_files)}")
    print(f"Validation labels moved: {len(val_files)}")
