import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import NDF, ViTFeatureExtractor
from google.colab import files

#upload on google colab
files.upload() #json kaggle key
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d paultimothymooney/breast-histopathology-images

!unzip -q breast-histopathology-images.zip

# Load the breast cancer image dataset
dataset_path = "/content"  

images = []
labels = []

count = 0

# Load images and labels from the dataset path
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        count += 1
        if count > 10:
            break 
        else: 
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for image_name in os.listdir(subfolder_path):
                        image_path = os.path.join(subfolder_path, image_name)
                        if os.path.isfile(image_path):
                            try:
                                image = Image.open(image_path)
                                image = image.resize((128, 128))  # Resize to a fixed size
                                image = np.array(image) / 255.0  # Normalize pixel values
                                images.append(image)
                                labels.append(int(subfolder))  # '1' for cancer, '0' for healthy
                            except Exception as e:
                                print(f"Error processing image {image_path}: {e}")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Load ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k-finetuned-imagenet')

# Extract features from the images
X_train_features = feature_extractor(images=X_train.tolist(), return_tensors='np', padding=True, truncation=True)
X_val_features = feature_extractor(images=X_val.tolist(), return_tensors='np', padding=True, truncation=True)
X_test_features = feature_extractor(images=X_test.tolist(), return_tensors='np', padding=True, truncation=True)

# Define the NDF model
model = NDF(num_layers=3, num_trees=5, depth=3, num_classes=2)

# Create a sequential model
sequential_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=X_train_features['pixel_values'].shape[1:]),
    model
])

# Compile the model
sequential_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

# Train the model
# sequential_model.fit(X_train_features['pixel_values'], y_train, epochs=10, validation_data=(X_val_features['pixel_values'], y_val))

# Evaluate the model
# loss, accuracy = sequential_model.evaluate(X_test_features['pixel_values'], y_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
