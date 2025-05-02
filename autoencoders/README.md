# Autoencoder Exploration

## Overview and Notebook
This notebook presents an exploration of autoencoder architectures implemented using TensorFlow and Keras. Autoencoders are neural networks designed to compress data into a low-dimensional representation and then reconstruct it, making them valuable for dimensionality reduction, denoising, and feature learning. The notebook progresses through three increasingly sophisticated implementations: a shallow autoencoder built with the Keras functional API, a shallow autoencoder using model subclassing, and finally a deep convolutional autoencoder. Detailed explanations of each architecture and implementation approach are included in the notebook ("autoencoders.ipynb").

## Project Goals
The goals of this exploration were two-fold:
1. To gain familiarity with different methods of implementing autoencoders in TensorFlow/Keras (functional API vs. model subclassing)
2. To explore various autoencoder architectures, from simple dense networks to more sophisticated convolutional structures

## Implementation Details
The primary dataset used in this project is the MNIST handwritten digit dataset, which consists of 28x28 pixel grayscale images of digits 0-9. Each implementation approach tackles the same fundamental task: compressing these images into a lower-dimensional representation and then reconstructing the original images from this compressed format.

### Shallow Autoencoder (Functional API)
The first implementation uses the Keras functional API to construct a simple autoencoder with the following characteristics:
* Input: Flattened 784-dimensional vectors (28×28 pixel images)
* Encoder: A single dense layer reducing the dimension to 36 neurons with ReLU activation
* Decoder: A single dense layer expanding back to 784 dimensions with sigmoid activation
* Loss function: Binary cross-entropy
* Optimization: Adam optimizer with a learning rate of 0.001

### Shallow Autoencoder (Model Subclassing)
The second implementation achieves the same architecture but leverages Keras model subclassing to provide a more object-oriented approach:
* The same architecture is implemented by defining a custom `Autoencoder` class
* This approach demonstrates the flexibility of Keras for more complex model designs
* Performance metrics are comparable to the functional API implementation

### Deep Convolutional Autoencoder
The final and most sophisticated implementation uses convolutional layers rather than dense connections:
* Input: 28×28×1 image tensors (2D structure preserved)
* Encoder: Two convolutional layers with 16 and 8 filters respectively
* Decoder: Two transposed convolutional layers followed by a final convolutional layer
* Loss function: Mean squared error
* Optimization: Adam optimizer with a learning rate of 0.001

## Results
Each model's performance was evaluated based on two primary metrics:
1. Loss curves showing training and validation performance over epochs
2. Visual comparison of original test images with their reconstructions

The notebook provides a framework for comparing different autoencoder implementations, though no explicit comparative analysis between the models is presented. The primary focus is on exploring different architectural approaches and implementation methods rather than optimizing for performance. This exploration serves as a valuable foundation for understanding how different autoencoder structures can be implemented and visualized.

## Technical Implementation
The project implements several important technical components:
* Custom visualization functions for loss curves and image reconstruction comparison
* Checkpointing to save model weights during training
* JSON serialization to store training history for later analysis
* Proper normalization of input data (scaling pixel values to [0,1])
* Reshape operations to transform data between formats required by different architectures

This exploration was conducted as part of the IBM Course on Deep Learning and Reinforcement Learning, itself a component of the IBM Machine Learning Professional Certification. The primary focus was on understanding different approaches for building model architectures in Keras, rather than hyperparameter optimization or achieving state-of-the-art performance.
