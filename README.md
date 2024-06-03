#Racket Deep Learning Library
This is a simple deep learning library implemented in Racket, designed to provide a basic framework for building and training feedforward neural networks. The library includes essential components such as tensor operations, activation functions, loss functions, and backpropagation for gradient calculation and weight updates.
Features

Tensor creation and manipulation functions
Element-wise tensor operations (addition, subtraction, multiplication)
Matrix multiplication for tensors
Activation functions (ReLU and its derivative)
Loss function (mean squared error)
Dense layer forward propagation
Dense layer backward propagation (gradient calculation and weight updates)
Random tensor initialization
Feedforward neural network initialization

Usage
To use the library, follow these steps:

Clone the repository or download the source files.
Install Racket if you haven't already.
Include the necessary library files in your Racket project:

(require "tensor.rkt")
(require "deep_learn_library.rkt")
