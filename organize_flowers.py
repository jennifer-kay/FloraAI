import os
import shutil
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

# Load the labels from the imagelabels.mat file using scipy.io.loadmat
mat = loadmat('imagelabels.mat')
labels = mat['labels'][0]

# Source directory where images are stored
src_dir = 'flowers'  # Replace with the actual directory of images

# Define the directories for train, validation, and test sets
train_dir = os.path.join('flower_data', 'train')
valid_dir = os.path.join('flower_data', 'valid')
test_dir = os.path.join('flower_data', 'test')

# Create directories if they do not exist
for directory in [train_dir, valid_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)
    for i in range(1, 103):  # 102 classes
        os.makedirs(os.path.join(directory, str(i)), exist_ok=True)

# Generate random train, validation, and test splits
indices = np.arange(len(labels))
train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, labels, test_size=0.3, stratify=labels)
valid_idx, test_idx, valid_labels, test_labels = train_test_split(temp_idx, temp_labels, test_size=0.5, stratify=temp_labels)

# Function to move files to their respective directories
def move_files(indices, dest_dir):
    for idx in indices:
        label = labels[idx]
        img_filename = f'image_{idx + 1:05d}.jpg'  # Zero-padded image filename
        src_path = os.path.join(src_dir, img_filename)
        dest_path = os.path.join(dest_dir, str(label), img_filename)
        if os.path.exists(src_path):  # Check if the source file exists
            shutil.move(src_path, dest_path)
        else:
            print(f"File {img_filename} does not exist in the source directory.")

# Organize images into train, valid, and test directories
move_files(train_idx, train_dir)
move_files(valid_idx, valid_dir)
move_files(test_idx, test_dir)

print("Dataset organized successfully!")
