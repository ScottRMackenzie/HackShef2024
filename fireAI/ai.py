import random
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Ask the user whether to train the model or load an existing one
train_model_choice = input("Do you want to train the model? (y/n): ").strip().lower()

# Define paths to the image and label directories
image_dir = "./data/train/images"
label_dir = "./data/train/labels"
max_boxes = 5  # Maximum number of bounding boxes per image

### Step 1: Load Data Function with Normalization
def load_data(image_dir, label_dir, max_images=2000, max_boxes=5):
    images, bboxes, labels = [], [], []
    
    all_images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    random.shuffle(all_images)
    selected_images = all_images[:max_images]

    for image_name in selected_images:
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        img = cv2.resize(img, (224, 224))
        images.append(img)

        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
        bbox_list = []
        try:
            with open(label_path, 'r') as f:
                labels = f.readlines()
                for line in labels:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, center_x, center_y, width, height = map(float, parts)
                        bbox = [
                            (center_x - width / 2),  # xmin
                            (center_y - height / 2), # ymin
                            (center_x + width / 2),  # xmax
                            (center_y + height / 2)  # ymax
                        ]
                        bbox_list.append(bbox)
        except FileNotFoundError:
            print(f"Label file not found for image: {image_name}")

        while len(bbox_list) < max_boxes:
            bbox_list.append([-244, -244, -244, -244])
        bboxes.append(bbox_list[:max_boxes])

    return np.array(images) / 255.0, np.array(bboxes)

if train_model_choice == "y":
    images, labels = load_data(image_dir, label_dir, max_boxes=max_boxes)

### Step 2: Define an Improved Model Architecture
def create_model(max_boxes=5):
    model = Sequential([
        Input(shape=(224, 224, 3)),  # Define the input shape as (224, 224, 3)
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(max_boxes * 4, activation='sigmoid')  # Output normalized coordinates for max_boxes
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])
    return model

### Step 3: Custom Loss Function (if needed)
# Not required in this example, but can be customized further
def custom_loss(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

if train_model_choice == "y":
    model = create_model(max_boxes=max_boxes)
    model.fit(images, labels.reshape(-1, max_boxes * 4), epochs=10, batch_size=8)
    model.save('improved_bbox_model.keras')
    print("Model saved to 'improved_bbox_model.keras'")

### Step 4: Load the Model
try:
    model = tf.keras.models.load_model('improved_bbox_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

### Step 5: Enhanced Prediction Function with Thresholding
def predict_bounding_boxes(image_path, model, max_boxes=5, threshold=0.1):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    predictions = model.predict(img_resized)[0].reshape(max_boxes, 4)

    height, width, _ = img.shape

    for bbox in predictions:
        xmin, ymin, xmax, ymax = bbox

        # Skip boxes with very small size (near zero) and confidence threshold
        if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
            continue  # No box detected
        
        # Apply thresholding to filter out weak predictions
        if (xmax - xmin) < threshold or (ymax - ymin) < threshold:
            continue  # Skip small boxes (likely false positives)

        # Scale the bounding box coordinates to original image size
        xmin, xmax = int(xmin * width), int(xmax * width)
        ymin, ymax = int(ymin * height), int(ymax * height)

        # Ensure bounding box is within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("Bounding Box Predictions", img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

### Step 6: Predict on Random Images
def predict_random_images(image_dir, model, num_images=10):
    all_images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
    random_images = random.sample(all_images, num_images)
    for image_name in random_images:
        image_path = os.path.join(image_dir, image_name)
        predict_bounding_boxes(image_path, model)

if model:
    predict_random_images('./data/train/images', model, num_images=100)
