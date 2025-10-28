# Import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom function to display images
def show_data(dataset, num_images=3):
    """Display images from dataset in grayscale without labels or axes"""
    # Create figure with specified dimensions
    plt.figure(figsize=(12, 4))
    
    # Iterate through dataset and display images
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        # Display image in grayscale format
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        # Remove axes for clean display
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load Fashion-MNIST validation data
(_, _), (val_images, val_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0,1] range
val_images = val_images.astype('float32') / 255.0
# Add channel dimension for TensorFlow
val_images = val_images[..., tf.newaxis]

# Create TensorFlow dataset from arrays
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

# Display first three images
show_data(val_dataset)