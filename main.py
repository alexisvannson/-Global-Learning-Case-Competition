import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import ViTFeatureExtractor
from some_custom_ndf_library import NDF  # Make sure to import your NDF model correctly

# Assume Kaggle dataset is already downloaded and extracted to 'dataset_path'
dataset_path = "path/to/breast-histopathology-images"  

# Placeholder for images and labels
images = []
labels = []

# Efficiently load images and labels
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            image = Image.open(image_path).resize((224, 224))
            images.append(np.array(image) / 255.0)  # Normalize pixel values
            labels.append(int(folder_name))  # Assuming folder name is the label
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Convert to numpy arrays and split the dataset
images = np.array(images)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Initialize ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the NDF model (assuming NDF is properly defined or imported)
model = NDF(num_layers=3, num_trees=5, depth=3, num_classes=2)

# Create a TensorFlow dataset for more efficient loading
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Model compilation and training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
