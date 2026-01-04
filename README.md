🫁 Pneumonia Detection from Chest X-Ray Images using CNN
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to automatically detect Pneumonia from Chest X-ray images.

The model classifies X-ray images into two categories:

PNEUMONIA

NORMAL

The system is trained and evaluated using the Chest X-Ray Pneumonia dataset and achieves strong performance on unseen test data.

📂 Dataset

Dataset Name: Chest X-Ray Images (Pneumonia)

Source: Kaggle

Link:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

⚙️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Google Colab

KaggleHub

🧠 Model Architecture

The CNN architecture consists of:

Multiple Conv2D layers with ReLU activation

Batch Normalization for training stability

MaxPooling layers for downsampling

Dropout layers to reduce overfitting

Fully Connected Dense layers

Sigmoid output layer for binary classification

Training Details:

Loss Function: Binary Cross-Entropy

Optimizer: RMSprop

Evaluation Metric: Accuracy

🔄 Data Preprocessing & Augmentation
Preprocessing

Grayscale image conversion

Image resizing to 150 × 150

Normalization (pixel values scaled to [0, 1])

Data Augmentation

Rotation

Width & height shifts

Zoom

Horizontal flipping

📊 Results

Model evaluated on unseen test data

Performance metrics include:

Accuracy

Precision

Recall

F1-Score

Confusion matrix visualization used for evaluation
