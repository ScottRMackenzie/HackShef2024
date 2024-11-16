import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def parse_label_file(label_file):
    box = None
    max_class_id = 0
    try:
        label_path = label_file.numpy().decode('utf-8')
        if os.path.getsize(label_path) == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
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
                    
                    if class_id > max_class_id:
                        max_class_id = class_id
                        box = [xmin, ymin, xmax, ymax, class_id]
                    
                except ValueError as e:
                    print(f"Warning: Could not parse line in {label_path}: {line} - Error: {e}")
                    continue
                    
        # Always return a numpy array with shape (5,)
        if box is None:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        return np.array(box, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing label file {label_path}: {e}")
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

# Function to load and preprocess images and labels
def load_and_preprocess(image_obj, label_obj, target_size=(224, 224)):
    # Add explicit output shapes
    image, boxes = tf.py_function(
        lambda x, y: _load_and_preprocess_impl(x, y, target_size),
        [image_obj, label_obj],
        [tf.float32, tf.float32]
    )
    # Set static shapes
    image.set_shape([*target_size, 3])
    boxes.set_shape([5])
    return image, boxes

def _load_and_preprocess_impl(image_obj, label_obj, target_size=(224, 224)):
    img_path = image_obj.numpy().decode('utf-8')
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    boxes = parse_label_file(label_obj)
    
    return image, boxes

# Function to create a dataset
def create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32):
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
                            img.size
                        # Verify the label file can be opened
                        with open(label_path, 'r') as f:
                            f.read()
                        image_paths.append(image_path)
                        label_paths.append(label_path)
                    except Exception as e:
                        print(f"Skipping corrupted files: {image_path} - Error: {e}")
                
                # if len(image_paths) >= 2000:  # Debug with smaller dataset
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
        lambda x, y: load_and_preprocess(x, y, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # .cache().prefetch(tf.data.AUTOTUNE)
    
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
    return dataset

# if model is not already trained
if not os.path.exists('model.keras'):
    print("Training model...")
    # Example usage
    image_dir = './data/train/images'
    label_dir = './data/train/labels'
    dataset = create_dataset(image_dir, label_dir, target_size=(224, 224), batch_size=32)

    # print the shape of the dataset
    print(dataset.take(1).element_spec[0].shape[1])
    
    list(dataset)

    # train the model
    model = tf.keras.Sequential([
        # Replace the first Conv2D layer with an Input layer followed by Conv2D
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1 * 5),
        tf.keras.layers.Reshape((1, 5))
    ])

    # compile the model with appropriate loss function
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error
    )

    # print the shape of the dataset
    print(dataset.shape)
    # train the model
    history = model.fit(dataset, epochs=2)

    # save the model
    model.save('model.keras')
else:
    # load the model
    model = tf.keras.models.load_model('model.keras') # Switch to .keras for compatibility

# Load the testing dataset
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
    images = [np.array(Image.open(path)) for path in image_paths]
    
    for i, (image, pred_box) in enumerate(zip(images, predictions)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Handle single box prediction
        box = pred_box[0]  # Get first (and only) box since model outputs shape [batch, 1, 5]
        
        # Extract coordinates and convert to absolute pixels
        xmin, ymin, xmax, ymax, class_id = box
        
        # Plot the rectangle using absolute coordinates
        width = (xmax - xmin) * image.shape[1]
        height = (ymax - ymin) * image.shape[0]
        
        plt.gca().add_patch(plt.Rectangle(
            (xmin * image.shape[1], ymin * image.shape[0]),
            width,
            height,
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

