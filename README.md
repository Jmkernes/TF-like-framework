# TF-like-framework
A reconstruction of bare bones TensorFlow 1

My implementation of a computational graph to train neural networks. Similar to TensorFlow V1. The main guts are in the module Framework.py.

There I define the main pieces of the computational graph, as well as implement a number of operators on layers, including convolution, pooling, dropout, batchnorm, etc. 
Numerical gradient checking on networks can be performed by importing the module gradient_checking.py

There are two example notebooks. The first is using the framework to train a simple two-three layer neural network on a toy spiral dataset, where the goal is to classify different arms of a spiral.
In the notebook conv_main.py, we load part of the CIFAR-10 dataset and train both a three layer neural net and a three layer convolutional net on the training set.

The framework is paired with the graphviz module to automatically generate and display the computational graph associated with a net.
