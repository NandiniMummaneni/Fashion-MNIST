import tensorflow as tf
import matplotlib.pyplot as plt

def show_data(dataset, num_images=3):
    """Display images from dataset in grayscale without labels or axes"""
    plt.figure(figsize=(12, 4))
    
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load Fashion-MNIST dataset
(_, _), (val_images, val_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize and add channel dimension
val_images = val_images.astype('float32') / 255.0
val_images = val_images[..., tf.newaxis]

# Create TensorFlow dataset from validation data
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

# Display first three images
show_data(val_dataset)