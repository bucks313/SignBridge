import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dir = r"C:\Users\Bilal\Downloads\fyp_appfinal\fyp_app\useful_frames"  # Original dataset for training
augmented_dir = r"C:\Users\Bilal\Downloads\fyp_appfinal\fyp_app\augmented_frames"  # Augmented dataset for validation and testing
output_dir = r"C:\Users\Bilal\Downloads\fyp_appfinal\fyp_app\dataset"  # Final dataset directory

# Ratios
val_ratio = 0.5  # Proportion of augmented data for validation (rest for testing)

def ensure_dir_exists(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_original_to_train(original_dir, train_dir):
    """
    Copy the original dataset to the training directory.
    Args:
        original_dir (str): Path to the original dataset.
        train_dir (str): Path to the training directory.
    """
    for root, dirs, files in os.walk(original_dir):
        for class_folder in dirs:
            class_path = os.path.join(root, class_folder)
            train_class_dir = os.path.join(train_dir, os.path.relpath(class_path, original_dir))
            ensure_dir_exists(train_class_dir)

            for frame in os.listdir(class_path):
                frame_path = os.path.join(class_path, frame)
                if frame.endswith((".jpg", ".png")):
                    shutil.copy(frame_path, train_class_dir)

            print(f"Copied original frames for class '{os.path.relpath(class_path, original_dir)}' to training.")

def split_augmented(augmented_dir, val_dir, test_dir, val_ratio):
    """
    Split the augmented dataset into validation and testing.
    Args:
        augmented_dir (str): Path to the augmented dataset.
        val_dir (str): Path to the validation directory.
        test_dir (str): Path to the testing directory.
        val_ratio (float): Proportion of data for validation (rest for testing).
    """
    for root, dirs, files in os.walk(augmented_dir):
        for class_folder in dirs:
            class_path = os.path.join(root, class_folder)
            rel_path = os.path.relpath(class_path, augmented_dir)

            # Get all frames in the class folder
            frames = [os.path.join(class_path, frame) for frame in os.listdir(class_path) if frame.endswith((".jpg", ".png"))]

            if len(frames) < 2:  # Log inconsistencies
                print(f"Warning: '{rel_path}' has only {len(frames)} frames. Augmentation may have failed.")
                continue

            if len(frames) == 2:  # Special case: only two frames
                val_files = [frames[0]]
                test_files = [frames[1]]
            else:
                # Split frames into validation and testing
                val_files, test_files = train_test_split(frames, test_size=(1 - val_ratio), random_state=42)

            val_class_dir = os.path.join(val_dir, rel_path)
            test_class_dir = os.path.join(test_dir, rel_path)

            ensure_dir_exists(val_class_dir)
            ensure_dir_exists(test_class_dir)

            for file in val_files:
                shutil.copy(file, val_class_dir)
            for file in test_files:
                shutil.copy(file, test_class_dir)

            print(f"Class '{rel_path}' split into:")
            print(f"  Validation: {len(val_files)} frames")
            print(f"  Test: {len(test_files)} frames")

def split_combined_dataset(original_dir, augmented_dir, output_dir, val_ratio):
    """
    Split dataset with original frames for training and augmented frames for validation/testing.
    Args:
        original_dir (str): Path to the original frames directory.
        augmented_dir (str): Path to the augmented frames directory.
        output_dir (str): Path to save the split dataset.
        val_ratio (float): Proportion of augmented data for validation (rest for testing).
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    ensure_dir_exists(train_dir)
    ensure_dir_exists(val_dir)
    ensure_dir_exists(test_dir)

    print("Copying original frames to training...")
    copy_original_to_train(original_dir, train_dir)

    print("Splitting augmented frames into validation and testing...")
    split_augmented(augmented_dir, val_dir, test_dir, val_ratio)

# Run the splitting process
split_combined_dataset(original_dir, augmented_dir, output_dir, val_ratio)
