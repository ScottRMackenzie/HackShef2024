import tensorflow as tf
import os

# Function to load and preprocess an image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    return image

# Function to load text data
def load_text(text_path):
    text = tf.io.read_file(text_path)
    return tf.strings.to_string(text)

# Paths to your image and text files
image_dir = '.'  # Directory containing images
text_dir = '.'     # Directory containing text files

# Create a list of image and text file paths
image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
text_files = [os.path.join(text_dir, fname) for fname in os.listdir(text_dir) if fname.endswith('.txt')]

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((image_files, text_files))

# Map the loading functions to the dataset
dataset = dataset.map(lambda img, txt: (load_image(img), load_text(txt)))

# Shuffle and batch the dataset
batch_size = 32
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

# Iterate through the dataset
for images, texts in dataset:
    print("Batch of images shape:", images.shape)
    print("Batch of texts:", texts.numpy())
