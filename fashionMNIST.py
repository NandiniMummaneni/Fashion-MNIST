# fashion_mnist_cnn.py
# Train a CNN to classify Fashion MNIST images (aim: ≥85% validation accuracy)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1️⃣ Load and preprocess the dataset
# -----------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add a channel dimension (28,28,1)
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# -----------------------------------------------------------
# 2️⃣ Create Dataset objects
# -----------------------------------------------------------
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# -----------------------------------------------------------
# 3️⃣ Build the CNN model
# -----------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.summary()

# -----------------------------------------------------------
# 4️⃣ Compile the model
# -----------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------------------------------------
# 5️⃣ Train the model
# -----------------------------------------------------------
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10)

# -----------------------------------------------------------
# 6️⃣ Evaluate and display accuracy
# -----------------------------------------------------------
test_loss, test_acc = model.evaluate(val_ds)
print(f"\n✅ Validation accuracy: {test_acc * 100:.2f}%")

# -----------------------------------------------------------
# 7️⃣ Plot training & validation accuracy
# -----------------------------------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
