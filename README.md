# Neural Networks From Scratch (Python)

A complete implementation of feed-forward neural networks built from scratch in Python, following the concepts from **Neural Networks from Scratch in Python (NNFS)** while deeply exploring the mathematics and implementation details behind modern deep learning.

This repository is not built on TensorFlow or PyTorch. Every component—from forward propagation to backpropagation and optimization—is implemented manually using **NumPy**.

The goal of this project was to understand how neural networks actually work under the hood rather than relying on high-level frameworks.

---

## Project Goals

- Build neural networks from first principles
- Understand the mathematics behind deep learning
- Implement every training component manually
- Learn how modern optimizers update parameters
- Understand backpropagation by deriving gradients step-by-step
- Gain experience debugging numerical computations
- Learn how models are trained, evaluated, saved, loaded, and used for inference

---

## Features

### Neural Network Architecture

- Fully connected (Dense) layers
- Modular layer design
- Object-oriented implementation
- Configurable network architectures

### Activation Functions

- ReLU
- Softmax
- Sigmoid
- Linear

Implemented with both:

- Forward propagation
- Analytical backward propagation

---

## Loss Functions

Implemented from scratch:

- Categorical Cross-Entropy
- Binary Cross-Entropy
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Including numerical stability techniques such as clipping to avoid logarithm and division errors.

---

## Optimizers

Implemented gradient-based optimization algorithms:

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- AdaGrad
- RMSProp
- Adam

Including:

- Learning rate decay
- Momentum
- Cache updates
- Bias correction

---

## Backpropagation

The entire backward pass is implemented manually.

This includes gradient computation for:

- Dense layers
- ReLU
- Softmax
- Sigmoid
- Linear activation
- All supported loss functions

Understanding and implementing backpropagation was one of the primary learning objectives of this project.

---

## Training Features

- Mini-batch training
- Dataset shuffling
- Epoch-based training loop
- Validation support
- Accuracy tracking
- Loss tracking
- Regularization support
- Parameter updates

---

## Model Management

Implemented support for:

- Saving model parameters
- Loading parameters
- Saving complete models
- Loading trained models
- Running inference on new data

---

## Dataset

The project uses the **Fashion-MNIST** dataset.

The repository includes utilities to:

- Download the dataset
- Load images
- Preprocess data
- Normalize inputs
- Prepare training and validation sets

---

## Topics Covered

This project covers many of the core topics behind modern deep learning:

- Forward propagation
- Backpropagation
- Chain rule
- Gradient descent
- Weight initialization
- Bias initialization
- Activation functions
- Loss functions
- Softmax probabilities
- Cross-entropy
- Regression
- Classification
- Binary classification
- Multi-class classification
- Model evaluation
- Numerical stability
- Batch processing
- Learning rate scheduling
- Parameter optimization
- Saving and loading neural networks

---

## Repository Structure

```
accuracy.py
activations.py
dataset.py
layers.py
losses.py
model.py
optimizers.py

main_multiclass_classification.py
main_regression.py
new_data.py
```

Each module focuses on a specific component of a neural network to keep the implementation clean, modular, and easy to understand.

---

## Technologies

- Python
- NumPy

No deep learning frameworks were used.

---

## What I Learned

This project significantly improved my understanding of how neural networks work internally.

Rather than treating machine learning models as black boxes, I learned how to implement:

- matrix-based forward propagation
- analytical gradient computation
- optimization algorithms
- model training loops
- inference pipelines
- numerical stability techniques
- modular neural network architectures

I also gained a much deeper understanding of the mathematical foundations behind modern deep learning.

---

## References

- *Neural Networks from Scratch in Python* — Harrison Kinsley & Daniel Kukieła

---

## Future Improvements

- Dropout
- Batch Normalization
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- GPU acceleration
- Additional datasets
- Hyperparameter search
- Visualization tools
- Unit tests

---

## License

This repository is intended for educational purposes and personal learning.
