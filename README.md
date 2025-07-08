
# NeuralNets from scratch

This repository presents a complete implementation of deep neural networks built entirely from scratch using Python and NumPy, without any high-level machine learning or deep learning libraries. The goal is to understand and demonstrate the fundamental building blocks of neural networks by coding all components manually.

## Project Overview

### General L-Layer Neural Network Framework

* Implemented a flexible **L-layer neural network** supporting any number of layers, with:

  * He initialization for parameters
  * Forward propagation using ReLU (hidden layers) and Sigmoid (output layer)
  * Numerically stable binary cross-entropy cost function
  * Backward propagation for gradient computation
  * Gradient descent parameter updates

### Application 1: Tabular Data Classification — Introvert vs Extrovert

* Applied the model to classify personality traits using tabular data.
* Used a **2-layer neural network** (input → hidden → output).
* Achieved steady learning with the cost reducing from approximately **1.38** to **0.30** over training.

### Application 2: Image Classification — Pandas vs Bears

* Worked on image classification with RGB images resized to **64x64**, flattened for input.
* Employed a **4-layer neural network** with three hidden layers of 8 neurons each to handle image complexity.
* To improve training on image data, we incorporated **He initialization** for weights and added **clipping in the cost function** to avoid numerical issues during logarithm computation.
* These improvements resulted in a significant cost reduction from around **0.76** to below **0.005** after training, demonstrating better model convergence on image data.

### Technical Highlights

* He initialization improved training stability and convergence speed, especially for the deeper 4-layer network applied to images.
* Clipping predicted probabilities in the cost function prevented NaN values and ensured stable gradient updates.
* Gradient checking confirmed the correctness of backpropagation.

### Limitations and Considerations

* Flattening images into vectors removes spatial context, limiting performance compared to convolutional networks, but the exercise remains valuable for learning core neural network mechanics.
* The fully connected network was sufficient for tabular data but required depth and careful initialization for image classification.

## How to Use

* Clone the repository.
* Prepare your dataset following preprocessing steps (resize, flatten, normalize).
* Train either the 2-layer or 4-layer model as appropriate.
* Monitor cost and accuracy metrics for performance evaluation.

## Acknowledgements

This project is inspired by and built upon the foundational teachings of **Andrew Ng’s Deep Learning Specialization**. His clear explanations and approach to neural networks have been instrumental in guiding this work.



