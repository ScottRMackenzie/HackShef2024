import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Function to parse the label file
def parse_label_file(label_file):
    """
    Parses a YOLO format label file and converts bounding boxes to (xmin, ymin, xmax, ymax) format.

    Args:
        label_file (str): Path to the label file.
    Returns:
        np.ndarray: Array of bounding boxes with shape (N, 5) where each box is [xmin, ymin, xmax, ymax, class_id].
    """
    boxes = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Unexpected label format in {label_file}: {line.strip()}")
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to (xmin, ymin, xmax, ymax)
                xmin = center_x - width / 2
                ymin = center_y - height / 2
                xmax = center_x + width / 2
                ymax = center_y + height / 2
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        
        return np.array(boxes)
    except Exception as e:
        raise RuntimeError(f"Error parsing label file {label_file}: {e}")

# Function to load and preprocess images and labels
def load_and_preprocess(image_and_label_tuple, target_size=(224, 224), max_boxes=100):
    """
    Loads and preprocesses an image and its corresponding labels.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        target_size (tuple): Desired image size (height, width).
        max_boxes (int): Maximum number of bounding boxes.
    
    Returns:
        tf.Tensor, tf.Tensor: Preprocessed image tensor and padded bounding boxes tensor.
    """
    try:
        # Load image
        image = tf.io.read_file(image_and_label_tuple[0])
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, target_size)  # Resize image
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
        image.set_shape([*target_size, 3])  # Set explicit shape

        # Load labels
        boxes = parse_label_file(image_and_label_tuple[1])
        
        # Pad or truncate boxes to fixed size
        num_boxes = len(boxes)
        if num_boxes > max_boxes:
            boxes = boxes[:max_boxes]
        elif num_boxes < max_boxes:
            padding = np.zeros((max_boxes - num_boxes, 5))
            boxes = np.vstack([boxes, padding])
        
        return image, tf.constant(boxes, dtype=tf.float32)
    except Exception as e:
        raise RuntimeError(f"Error processing file {image_and_label_tuple[0]} or {image_and_label_tuple[1]}: {e}")

# Function to create a dataset
def create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32, max_boxes=100):
    """
    Creates a TensorFlow dataset from image and label directories.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing label files.
        target_size (tuple): Desired image size (height, width).
        batch_size (int): Number of samples per batch.
        max_boxes (int): Maximum number of bounding boxes per image.

    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    """
    image_paths = []
    label_paths = []
    
    try:
        # Collect image and label file paths
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(image_dir, filename)
                label_name = filename.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(label_dir, label_name)
                
                # Only add if both image and label files exist
                if os.path.exists(label_path):
                    image_paths.append(image_path)
                    label_paths.append(label_path)
    except OSError as e:
        raise RuntimeError(f"Error reading directory: {e}")
    
    if not image_paths:
        raise ValueError("No valid image-label pairs found")
    
    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    
    # Load and preprocess images and labels
    dataset = dataset.map(lambda img, lbl: load_and_preprocess((img, lbl), target_size, max_boxes), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example usage
if __name__ == "__main__":
    image_dir = './data/train/images'
    label_dir = './data/train/labels'
    try:
        dataset = create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32)
        print("Dataset created successfully.")
    except Exception as e:
        print(f"Error: {e}")
