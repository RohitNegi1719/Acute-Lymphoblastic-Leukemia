import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
original_data_dir = r'C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Original'
base_dir = r'C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Split_Data'

# Define paths for training, validation, and test directories
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Create directories if they don't exist
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)
    for subdir in ['Benign', 'Early', 'Pre', 'Pro']:
        os.makedirs(os.path.join(directory, subdir), exist_ok=True)

# Set the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


# Function to split data and copy files
def split_data(class_name):
    # List all files in the class directory
    class_dir = os.path.join(original_data_dir, class_name)
    files = os.listdir(class_dir)

    # Split files into train, validation, and test sets
    train_files, temp_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (test_ratio + val_ratio)),
                                             random_state=42)

    # Function to copy files
    def copy_files(file_list, target_dir):
        for file_name in file_list:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(target_dir, class_name, file_name)
            shutil.copyfile(src, dst)

    # Copy files to respective directories
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)


# Split and copy data for each class
for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
    split_data(class_name)

print("Data split and copied successfully.")

"""
C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Split_Data\test: 416 files
C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Split_Data\train: 2277 files
C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Split_Data\validation: 416 files
"""
