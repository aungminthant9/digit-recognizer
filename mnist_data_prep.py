import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print dataset shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data for CNN input (add channel dimension)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"X_train shape after reshaping: {X_train.shape}")
print(f"X_test shape after reshaping: {X_test.shape}")
print(f"y_train shape after one-hot encoding: {y_train.shape}")
print(f"y_test shape after one-hot encoding: {y_test.shape}")

# Visualize some examples
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.show()

# Explore data distribution
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist([np.argmax(y) for y in y_train], bins=10, rwidth=0.8)
plt.title('Training Data Distribution')
plt.xlabel('Digit')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist([np.argmax(y) for y in y_test], bins=10, rwidth=0.8)
plt.title('Test Data Distribution')
plt.xlabel('Digit')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('mnist_distribution.png')
plt.show()

# Save processed data for later use
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("Data preparation complete. Files saved.")