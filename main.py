import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to parse the label file
# def parse_label_file(label_file):
#     boxes = []
#     try:
#         label_path = label_file.numpy().decode('utf-8')
#         with open(label_path, 'r') as f:
#             for line in f:
#                 try:
#                     parts = line.strip().split()
#                     if len(parts) != 5:
#                         print(f"Warning: Skipping invalid line in {label_path}: {line}")
#                         continue
                    
#                     class_id = int(parts[0])
#                     center_x = float(parts[1])
#                     center_y = float(parts[2])
#                     width = float(parts[3])
#                     height = float(parts[4])
                    
#                     # Ensure values are within valid range
#                     if not all(0 <= x <= 1 for x in [center_x, center_y, width, height]):
#                         print(f"Warning: Invalid coordinates in {label_path}: {line}")
#                         continue
                        
#                     xmin = max(0, center_x - width / 2)
#                     ymin = max(0, center_y - height / 2)
#                     xmax = min(1, center_x + width / 2)
#                     ymax = min(1, center_y + height / 2)
                    
#                     # boxes.append([xmin, ymin, xmax, ymax, class_id])
#                     boxes.append([class_id])
#                 except ValueError as e:
#                     print(f"Warning: Could not parse line in {label_path}: {line} - Error: {e}")
#                     continue
#     except Exception as e:
#         print(f"Error processing label file {label_path}: {e}")
#         return np.zeros((1, 5), dtype=np.float32)  # Return empty box on error
        
#     if not boxes:
#         # print(f"Warning: No valid boxes found in {label_path}")
#         return np.zeros((1, 5), dtype=np.float32)  # Return empty box if no valid boxes
        
#     return np.array(boxes)

def parse_label_file(label_file):
    box = []
    max_class_id = -1
    try:
        label_path = label_file.numpy().decode('utf-8')
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Ensure values are within valid range
                    if not all(0 <= x <= 1 for x in [center_x, center_y, width, height]):
                        print(f"Warning: Invalid coordinates in {label_path}: {line}")
                        continue
                        
                    xmin = max(0, center_x - width / 2)
                    ymin = max(0, center_y - height / 2)
                    xmax = min(1, center_x + width / 2)
                    ymax = min(1, center_y + height / 2)
                    
                    if (parts[0] > max_class_id):
                        max_class_id = parts[0]
                        box = [xmin, ymin, xmax, ymax, class_id]
                    
                except ValueError as e:
                    print(f"Warning: Could not parse line in {label_path}: {line} - Error: {e}")
                    continue
    except Exception as e:
        print(f"Error processing label file {label_path}: {e}")
        return [-1, np.zeros((1, 4), dtype=np.float32)]
        
    return np.array(box)

# Function to load and preprocess images and labels
def load_and_preprocess(image_obj, label_obj, target_size=(224, 224), max_boxes=100):
    # Add explicit output shapes
    image, boxes = tf.py_function(
        lambda x, y: _load_and_preprocess_impl(x, y, target_size, max_boxes),
        [image_obj, label_obj],
        [tf.float32, tf.float32]
    )
    # Set static shapes
    image.set_shape([*target_size, 3])
    boxes.set_shape([5])
    return image, boxes

def _load_and_preprocess_impl(image_obj, label_obj, target_size=(224, 224), max_boxes=100):
    img_path = image_obj.numpy().decode('utf-8')
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    boxes = parse_label_file(label_obj)
    
    # num_boxes = len(boxes)
    # if num_boxes > max_boxes:
    #     boxes = boxes[:max_boxes]
    # elif num_boxes < max_boxes:
    #     padding = np.zeros((max_boxes - num_boxes, 5))
    #     boxes = np.vstack([boxes, padding])
    
    return image, np.array(boxes, dtype=np.float32)

# Function to create a dataset
def create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32, max_boxes=100):
    image_paths = []
    label_paths = []
    
    try:
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.abspath(os.path.join(image_dir, filename))  # Use absolute paths
                label_name = filename.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.abspath(os.path.join(label_dir, label_name))
                
                # Verify both files exist and are readable
                if os.path.isfile(label_path) and os.path.isfile(image_path):
                    try:
                        # Verify the image can be opened
                        with Image.open(image_path) as img:
                            pass
                        # Verify the label file can be opened
                        with open(label_path, 'r') as f:
                            if f.read().strip():  # Check if file is not empty
                                image_paths.append(image_path)
                                label_paths.append(label_path)
                    except Exception as e:
                        print(f"Skipping corrupted files: {image_path} - Error: {e}")
                
                # if len(image_paths) >= 20:  # Debug with smaller dataset
                #     break
    except OSError as e:
        raise RuntimeError(f"Error reading directory: {e}")
    
    if not image_paths:
        raise ValueError("No valid image-label pairs found")
    
    print(f"Found {len(image_paths)} valid image-label pairs")  # Debug print
    
    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    
    # Add prefetch and cache for better performance
    dataset = dataset.map(
        lambda x, y: load_and_preprocess(x, y, target_size, max_boxes),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)
    
    # Add batch after preprocessing
    dataset = dataset.batch(batch_size)
    
    return dataset

def create_dataset_no_labels(image_paths, target_size=(224, 224), batch_size=32):
    # Create a dummy label path of empty strings with the same length as image_paths
    dummy_labels = [''] * len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, dummy_labels))
    dataset = dataset.map(
        lambda x, y: load_and_preprocess(x, y, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size)
    # plot the images
    for image in dataset.take(1):
        plt.imshow(image[0].numpy().astype('uint8')[0, :, :])
        plt.show()
    return dataset

# if model is not already trained
if not os.path.exists('model.keras'):
    # Example usage
    image_dir = './data/train/images'
    label_dir = './data/train/labels'
    dataset = create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32)

    # print the shape of the dataset
    print(dataset.take(1).element_spec[0].shape[1])

    # train the model
    model = tf.keras.Sequential([
        # Input layer and feature extraction
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        # Output layer: reshape to match expected dimensions [batch_size, 100, 5]
        tf.keras.layers.Dense(100 * 5),  # matches max_boxes * 5 values per box
        tf.keras.layers.Reshape((100, 5))  # Reshape to match target shape
    ])

    # compile the model with appropriate loss function
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error
    )

    # train the model
    history = model.fit(dataset, epochs=2)

    # save the model
    model.save('model.keras')
else:
    # load the model
    model = tf.keras.models.load_model('model.keras') # Switch to .keras for compatibility

# # Load the testing dataset
# image_dir = './data/test/images'
# label_dir = './data/test/labels'
# test_dataset = create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32)

# # Make predictions
# predictions = model.predict(test_dataset)

# # Get ground truth labels from test dataset
# ground_truth = np.vstack([y for x, y in test_dataset])

# # # Calculate mAP
# # mae_loss = tf.keras.losses.MeanAbsoluteError()
# # print(mae_loss(ground_truth, predictions))

# print(predictions)
# print(ground_truth)


def is_image_waiting():
    # return true if there is an image in /predictions
    return len(os.listdir('./predict')) > 0

def get_all_paths_of_images_waiting():
    return [os.path.join('./predict', filename) for filename in os.listdir('./predict') if filename.endswith(('.jpg', '.png'))]

def delete_all_images_in_directory(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


def plot_predictions(predictions, image_paths):
    # Load images using PIL and convert to numpy arrays
    images = [np.array(Image.open(path)) for path in image_paths]
    
    # For each prediction, plot the image and the predicted bounding box
    for i, (image, pred_boxes) in enumerate(zip(images, predictions)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # For each predicted box in the image
        for box in pred_boxes:
            # Skip boxes that are all zeros (padding)
            if not np.any(box):
                continue
                
            # Extract coordinates (xmin, ymin, xmax, ymax, class_id)
            xmin, ymin, width, height, _ = box
            
            # Plot the rectangle
            plt.gca().add_patch(plt.Rectangle(
                (xmin * image.shape[1], ymin * image.shape[0]),
                width * image.shape[1],
                height * image.shape[0],
                fill=False,
                color='red',
                linewidth=2
            ))
        
        plt.axis('off')
        plt.show()
        plt.close()

print("Waiting for images...")
predicting = False
while True:
    if not predicting and is_image_waiting():
        predicting = True
        print("Images found, processing...")
        image_paths = get_all_paths_of_images_waiting()
        dataset = create_dataset_no_labels(image_paths, target_size=(224, 224), batch_size=32)
        predictions = model.predict(dataset)
        plot_predictions(predictions, image_paths)
        print(predictions)
        # delete_all_images_in_directory('./predict')
        # predicting = False
    time.sleep(10)

