# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Create training data arrays
epochs = np.arange(1, 11)
# Loss values (decreasing over epochs)
loss = [0.8, 0.6, 0.45, 0.35, 0.28, 0.23, 0.19, 0.16, 0.14, 0.12]
# Accuracy values (increasing over epochs)
accuracy = [0.72, 0.82, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97]

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss curve on primary y-axis
ax1.plot(epochs, loss, 'r-', label='Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# Create secondary y-axis for accuracy
ax2 = ax1.twinx()
# Plot accuracy curve on secondary y-axis
ax2.plot(epochs, accuracy, 'b-', label='Accuracy', linewidth=2)
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add title and formatting
plt.title('Model Training: Loss and Accuracy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save plot as image file
plt.savefig('single_training_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("Single training curve saved as 'single_training_curve.png'")