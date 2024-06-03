# MIND A Racket Deep Learning Library

MIND is a simple deep learning library implemented in Racket, designed to provide a basic framework for building and training feedforward neural networks. The library includes essential components such as tensor operations, activation functions, loss functions, and backpropagation for gradient calculation and weight updates. This was written so I can learn more about neural networks for personal projects, to get better at racket and LISP, and of course to provide back to the community!

# Features

Tensor creation and manipulation functions
Element-wise tensor operations (addition, subtraction, multiplication)
Matrix multiplication for tensors
Activation functions (ReLU and its derivative)
Loss function (mean squared error)
Dense layer forward propagation
Dense layer backward propagation (gradient calculation and weight updates)
Random tensor initialization
Feedforward neural network initialization

# Usage
To use the library, follow these steps:

Clone the repository or download the source files.
Install Racket if you haven't already.
Include the necessary library files in your Racket project:
```
(require "tensor.rkt")
(require "deep_learn_library.rkt")
```

# Tensor creation
```
(define tensor1 (create-tensor '(2 2) '(1 2 3 4)))
```
To check the tensor you can run 

```
(print-tensor tensor1)
```

Initialize your neural network architecture using the provided functions. For example, to create a network with an input dimension of 3, a hidden layer with 4 neurons, and an output dimension of 2, you can use the initialize-fnn function:

```
(let-values ([(input-tensor weights biases) (initialize-fnn batch-size 3 4 2)])
```
This will create the necessary tensors for the input, weights, and biases of the network.

The library provides functions for performing forward propagation, loss calculation, and backpropagation. Here's an example of how to use them:

```
(let* ([hidden-output (dense-forward input-tensor hidden-weights hidden-biases)]
       [output (dense-forward hidden-output output-weights output-biases)])
```

This performs forward propagation through the network, calculating the hidden layer output and the final output.

# Loss Calculation: 

```
(let ([loss (mean-squared-error output-tensor output)])
```

This calculates the mean squared error between the predicted output and the target output.

# Backpropagation 

```
(let-values ([(output-grad-weights output-grad-biases output-grad-input)
              (dense-backward hidden-output output-weights output-biases output
                              (tensor-subtract output-tensor output) learning-rate)]
             [(hidden-grad-weights hidden-grad-biases _)
              (dense-backward input-tensor hidden-weights hidden-biases hidden-output
                              output-grad-input learning-rate)])
```
