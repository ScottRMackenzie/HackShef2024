import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to the image and label directories
image_dir = "./data/train/images"
label_dir = "./data/train/labels"

# Helper function to load data
def load_data(image_dir, label_dir):
    images = []
    bboxes = []
    
    # Loop through each image
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))  # Resize to standard size (224x224)
            images.append(img)
            
            # Load the corresponding label file
            label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
            with open(label_path, 'r') as f:
                label = f.readlines()
                
            # Initialize bounding boxes list
            bbox_list = []
            
            if len(label) == 0:  # No bounding boxes detected in the image
                bbox_list.append([0, 0, 0, 0])  # All zeros if no object
            else:
                for line in label:
                    parts = line.strip().split()
                    if len(parts) == 5:  # We have a valid bounding box
                        class_id, center_x, center_y, width, height = map(float, parts)
                        # Convert normalized coordinates to pixel values
                        bbox = [
                            int((center_x - width / 2) * 224),  # xmin
                            int((center_y - height / 2) * 224),  # ymin
                            int((center_x + width / 2) * 224),   # xmax
                            int((center_y + height / 2) * 224)   # ymax
                        ]
                        bbox_list.append(bbox)
            
            # If no bounding boxes, append the all-zero bbox list
            if not bbox_list:
                bbox_list = [[0, 0, 0, 0]]  # Ensure there's at least a "no object" label
            
            # Append the list of bounding boxes for this image
            bboxes.append(bbox_list)
    
    # Convert images and bounding boxes into numpy arrays
    return np.array(images), np.array(bboxes)

# Load data
images, labels = load_data(image_dir, label_dir)

# Normalize images
images = images / 255.0  # Normalize pixel values between 0 and 1

# Model to predict bounding boxes
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4)  # Predict 4 values (xmin, ymin, xmax, ymax) for bounding box
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Train the model
model.fit(images, labels, epochs=10, batch_size=4)

# Save the model
model.save('bounding_box_model.h5')
print("Model saved to 'bounding_box_model.h5'")

# Example of using the trained model to predict on a new image
def predict_bounding_box(image_path):
    # Load the saved model
    model = tf.keras.models.load_model('bounding_box_model.h5')
    
    # Preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # Get the predicted bounding box
    prediction = model.predict(img_resized)
    
    # Denormalize back to image size
    pred_bbox = prediction[0]
    xmin, ymin, xmax, ymax = pred_bbox
    xmin = int(xmin * 224)
    ymin = int(ymin * 224)
    xmax = int(xmax * 224)
    ymax = int(ymax * 224)
    
    # Draw the bounding box on the image
    img = cv2.imread(image_path)
    
    # Check if the prediction is all zeros (indicating no object)
    if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
        print("No object detected in the image.")
    else:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow("Predicted Bounding Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example of predicting on an image
predict_bounding_box('.\data\test\images\AoF06723.jpg')
