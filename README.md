# CIFAR-10 Image Classification with a Simple Two-Layer Neural Network

This project implements a simple two-layer neural network from scratch to classify images from the CIFAR-10 dataset. The main objective is to gain a deeper understanding of deep neural networks, backpropagation, and gradient descent by implementing these concepts manually before using a deep learning framework like PyTorch. The assignment is divided into four questions, each focusing on a different aspect of neural network implementation and training.
Project Structure

    ex2_FCnet.py: Contains the implementation of a feedforward neural network and the backpropagation algorithm using only basic matrix operations. This code is used for questions 1-3.
    ex2_pytorch.py: Contains the implementation of a network using PyTorch, allowing us to experiment with deeper architectures and take advantage of PyTorch's optimizations (Question 4).

Implementation Steps

    Feedforward Model:
        Implemented a basic two-layer neural network using matrix operations for forward propagation.

    Backpropagation:
        Derived and coded the backpropagation algorithm to compute gradients and update weights.

    Training with Stochastic Gradient Descent:
        Used stochastic gradient descent (SGD) to train the model, experimenting with different hyperparameters to improve performance.

    PyTorch Implementation:
        Re-implemented the model using PyTorch for easier experimentation with deeper networks and more advanced features.

Results

    After training, the PyTorch the two-layer network achieved around 58% test accuracy, which is a reasonable result given the simplicity of the model. I did not increase the depth of the network because the performance didn't improve at all.


Conclusion

This project provided a hands-on understanding of neural network fundamentals, backpropagation, and how to train models from scratch. The results are decent for such a simple architecture, and using PyTorch allowed for a smoother experience in exploring the model.
