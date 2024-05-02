import os
import shutil
import numpy as np

def train_val_split(src_images, src_masks, dest_train_img, dest_train_mask, dest_val_img, dest_val_mask, val_size=0.2):
    # Get all filenames
    images = os.listdir(src_images)
    masks = os.listdir(src_masks)

    # Shuffle with the same order
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = np.array(images)[indices]
    masks = np.array(masks)[indices]

    # Split indices
    val_indices = int(len(images) * val_size)

    # Validation set
    val_images = images[:val_indices]
    val_masks = masks[:val_indices]

    # Training set
    train_images = images[val_indices:]
    train_masks = masks[val_indices:]

    # Function to copy files
    def copy_files(files, src_dir, dest_dir):
        for file in files:
            shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    # Copy files to their new directories
    copy_files(train_images, src_images, dest_train_img)
    copy_files(train_masks, src_masks, dest_train_mask)
    copy_files(val_images, src_images, dest_val_img)
    copy_files(val_masks, src_masks, dest_val_mask)

    print(f"Moved {len(train_images)} to training set and {len(val_images)} to validation set.")

# Usage example
train_val_split('oil-spill-detection-dataset/Oil Spill Detection Dataset/train/images', 'oil-spill-detection-dataset/Oil Spill Detection Dataset/train/labels',
                'data/input/train/images', 'data/input/train/masks',
                'data/input/val/images', 'data/input/val/masks')
