# traffic-light-classification
# Traffic Sign Recognition with Convolutional Neural Network

This repository contains code for a Convolutional Neural Network (CNN) trained to recognize German traffic signs. The model is implemented using the Keras framework and is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Getting Started

To use this code, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. Run the Jupyter notebook `Untitled5.ipynb`:

    ```bash
    jupyter notebook Untitled5.ipynb
    ```

    Note: Make sure you have the necessary dependencies installed.

3. Execute the notebook cell by cell, following the instructions and comments provided.

## Requirements

- Python 3
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Matplotlib
- pandas
- OpenCV

You can install the required dependencies using the following:

```bash
pip install jupyter tensorflow keras numpy matplotlib pandas opencv-python
```

## Dataset

The GTSRB dataset is used for training and testing the model. It includes images of 43 different traffic sign classes.

- Training data: 'german-traffic-signs/train.p'
- Validation data: 'german-traffic-signs/valid.p'
- Test data: 'german-traffic-signs/test.p'

## Data Preprocessing

The notebook includes various data preprocessing steps such as loading, visualizing, and augmenting the dataset. It also performs image normalization and converts labels to one-hot encoded format.

## Model Architecture

The implemented CNN model consists of convolutional layers, max-pooling layers, a flatten layer, fully connected layers, and a softmax output layer. The architecture is summarized using Keras's `summary()` function.

## Model Training

The model is trained using the Adam optimizer and categorical crossentropy loss. Data augmentation is applied during training using the `ImageDataGenerator` from Keras. Training progress is visualized with loss and accuracy plots.

## Model Evaluation

The trained model is evaluated on the test dataset, and the test accuracy is displayed.

## Prediction Example

The notebook concludes with an example of predicting a traffic sign from an internet image using the trained model.

Feel free to explore, modify, and use the code for your own projects related to image classification or traffic sign recognition. 
