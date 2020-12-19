# Schrodinger Equation Deep Learning Project

## Abstract

Computational models that use machine learning principles to solve quantum mechanics problems promise to deliver accurate as possible information about a quantum system at high speeds. These two fields are related to each other by linear algebra, and it is imperative we examine the redundancy in solving Schrodinger's equations throughout different types of potential situations: step, piecewise linear, and random Fourier series. This prjoect uses the TensorFlow package in Python to initially generate one-dimensional potentials and then solve them using a gradient descent method. These potentials and their respective solutions are partitioned into sets of training, validation, and test data. The training data is inputted into a simple neural network with two hidden layers. The mean square distance between the "correct"s olutions and the output of the neural network is the cost function and the gradient descent on the network "solves" the problem. 

## Data Generation

* `genpotential.py`: We require training data for the neural network. We generate thousands of random potential functions and solve them with conventional methods and solve them individually. We currently use seed 0 to create the initial set of data, but we can modify this with a for loop to create more data.

## Neural Network

* `schrodinger_nn.py`: We set up a simple neural network and solve the one-dimensional Schrodinger equation. We currently use seed 0 to read in the initial set of data, but we can modify this with a for loop to read in more data. There is additional code that can be modified for generating plots of the potentials and for saving the information about the weights and biases.

## Visualization

* `visualize.py`: Outputs bitmaps of the weights and biases from `schrodinger_nn.py`. Sorts them using the Guassian kernel to increase spatial correlation between weights and nodes. Doubles the size of the bitmap before generating the output. 