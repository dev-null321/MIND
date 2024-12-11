# RacoGrad

RacoGrad is a simple deep learning library implemented in Racket, a version of scheme-lisp, designed to provide a basic framework for building and training feedforward neural networks. The library includes essential components such as tensor operations, activation functions, loss functions, and backpropagation for gradient calculation and weight updates. This is a hobby project, with updates coming sporadically. 
## Features

- Tensor creation and manipulation functions
- Element-wise tensor operations (addition, subtraction, multiplication)
- Matrix multiplication for tensors
- Activation functions (ReLU, Sigmoid, Tanh, and their derivatives)
- Loss functions (Mean Squared Error and Cross Entropy)
- Dense layer forward propagation
- Dense layer backward propagation (gradient calculation and weight updates)
- Random tensor initialization
- Feedforward neural network initialization
- One-hot encoding for classification
- Softmax function for multi-class classification

## Usage

To use the library, follow these steps:

1. Clone the repository or download the source files.
2. Install Racket if you haven't already.
3. Include the necessary library files in your Racket project:
   ```racket
   (require "tensor.rkt")
   (require "deep_learn_library.rkt")
   ```

### Tensor Creation

Create a tensor using the following:
```racket
(define tensor1 (t:create '(2 2) '(1 2 3 4)))
```
To check the tensor you can run:
```racket
(t:print tensor1)
```

### Neural Network Initialization

Initialize your neural network architecture using the provided functions. For example, to create a network with an input dimension of 3, a hidden layer with 4 neurons, and an output dimension of 2, you can use the `initialize-fnn` function:
```racket
(let-values ([(input-tensor weights biases) (initialize-fnn 32 3 4)])
  ...)
```
This will create the necessary tensors for the input, weights, and biases of the network.

### Forward Propagation

Perform forward propagation through a dense layer:
```racket
(let* ([hidden-output (dense-forward input-tensor weights biases relu)]
       [output (dense-forward hidden-output output-weights output-biases softmax)])
  ...)
```
This calculates the hidden layer output and the final output.

### Loss Calculation

Calculate the loss using the provided loss functions:
```racket
(let ([loss (mean-squared-error output-tensor output)])
  ...)
```
Or for classification tasks:
```racket
(let ([loss (cross-entropy y-pred y-true)])
  ...)
```

### Backpropagation

Perform backpropagation to compute gradients and update weights:
```racket
(let-values ([(grad-weights grad-biases grad-input)
              (dense-backward input-tensor weights biases output grad-output relu-derivative 0.01)])
  ...)
```

### Training Loop

Here's an example of a training loop for the MNIST dataset:
```racket
(for ([epoch (in-range epochs)])
  (let* ([indices (shuffle (range (car (t:shape X-train))))] ; Shuffle indices
         [num-batches (quotient (length indices) batch-size)])

    ;; Iterate through batches
    (for ([batch (in-range num-batches)])
      (let* ([batch-indices (take (drop indices (* batch batch-size)) batch-size)]
             [X-batch (get-batch X-train batch-indices)] ; Get batch data
             [y-batch (get-batch y-train batch-indices)]
             [loss (train-batch X-batch y-batch)])       ; Train on batch

        ;; Log batch results every 50 batches
        (when (= (modulo batch 50) 0)
          (let ([batch-accuracy (get-test-accuracy X-batch y-batch)])
            (printf "Epoch ~a, Batch ~a, Loss: ~a, Accuracy: ~a%~n"
                    epoch batch loss batch-accuracy)))))

    ;; Evaluate on test set after each epoch
    (let ([test-accuracy (get-test-accuracy X-test y-test)])
      (printf "Epoch ~a complete. Test accuracy: ~a%~n"
              epoch test-accuracy))))
```

### Example Output
After training your model, you might see output like this:
```
Epoch 0, Batch 50, Loss: 0.7881, Accuracy: 72.5%
Epoch 1, Batch 100, Loss: 0.5803, Accuracy: 88.2%
...
Epoch 9 complete. Test accuracy: 91.87%
```

## TO DO

- **Planned Tensor Operations:**
  - Broadcasting
  - Sparse tensor support
- **Advanced Features:**
  - Convolutional layers
  - GPU acceleration (CUDA/Metal)
- **Learning Algorithms:**
  - Optimizers: SGD, Adam, RMSProp
  - Regularization: L1, L2, Dropout
- **Autograd Enhancements:**
  - Custom gradient functions
  - Better debugging for gradients (e.g., vanishing/exploding gradient detection)


## Why RacoGrad?

RacoGrad is similar in spirit to libraries like Micrograd and Tinygrad, but it’s implemented in Racket to:

- Provide a learning tool for understanding the fundamentals of autograd.
- Leverage the power of Lisp-based languages for AI development.
- Offer a minimalist, educational framework for neural networks in Racket.

While not as feature-rich as PyTorch or TensorFlow, RacoGrad is perfect for learning, experimentation, and lightweight use cases.
