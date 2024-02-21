# Breast Cancer Image Classification with Hugging Face's NDF

This script performs breast cancer image classification using Hugging Face's Neural Decision Forests (NDF) framework. It leverages a pre-trained Vision Transformer (ViT) for feature extraction and the NDF model for classification.

## Overview

Breast cancer is a critical health issue, and early detection plays a crucial role in effective treatment. Image classification techniques, especially deep learning models, have shown promising results in detecting breast cancer from histopathological images.

This script aims to:

- **Load Dataset**: It loads a dataset of breast histopathology images. The dataset should be organized into directories, with subdirectories representing different classes (e.g., '0' for healthy and '1' for cancerous).

- **Preprocess Images**: The script preprocesses the images, resizing them to a fixed size and normalizing pixel values to ensure consistency across the dataset.

- **Feature Extraction**: It utilizes a pre-trained Vision Transformer (ViT) model to extract features from the preprocessed images. These features are then used as input for the Neural Decision Forests (NDF) model.

- **Model Sequencing**: The script constructs a sequential model in TensorFlow, combining the ViT feature extractor and the NDF model. This sequential model is compiled for classification with appropriate loss and optimization functions.

- **Model Training**: It trains the NDF model on the preprocessed image features, using a portion of the dataset designated for training. The training process involves optimizing the model parameters to minimize the classification loss.

- **Model Evaluation**: After training, the script evaluates the performance of the trained model on a separate portion of the dataset reserved for testing. It computes metrics such as test loss and accuracy to assess the model's effectiveness in classifying breast cancer images.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Hugging Face Transformers

You can install the required Python packages using pip:

```bash
pip install tensorflow transformers
```

## Usage

1. **Dataset Preparation**: Place your breast histopathology image dataset in a directory. Update the `dataset_path` variable in the script to point to your dataset directory.

2. **Running the Script**: Execute the script in your preferred Python environment:

```bash
python breast_cancer_classification.py
```

3. **Training and Evaluation**: The script will load the dataset, preprocess the images, train the NDF model, and evaluate its performance on a test dataset.

4. **Results**: Upon completion, the script will display the test loss and accuracy metrics.

## Customization

- You can customize the script by adjusting parameters such as image size, model architecture, and training epochs to better suit your dataset and requirements.
- Experiment with different pre-trained models available in the Hugging Face Transformers library for feature extraction.

## Credits

- This script utilizes Hugging Face's NDF framework and Transformers library for image classification.
- Dataset used in this script: [Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images).

## License

This script is provided under the [MIT License](LICENSE).

---

Feel free to adjust the README file as per your project's specific details and requirements.
