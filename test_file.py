import tensorflow as tf

# Read and process the image
image_path = './WEB08119.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.image.resize(image, [224, 224])

# Read the text file
text_path = './WEB08119.txt'
text = tf.io.read_file(text_path)

# Print the results
print("Image shape:", image.shape)
print("Text content:", text.numpy().decode('utf-8'))  # Decode bytes to string
