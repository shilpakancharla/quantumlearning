"""
    Reads training data from genpotential.py and then initalizes a neural network with two layers.
"""

import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

bins = 128
seedmax = 20 # Open seed files 0 to 19
train_x = []
train_y = []
valid_x = []
valid_y = []

# Reading in data
for i in range(seedmax):
    with open('test_potentials' + str(i) + '.csv', 'r') as doc:
        reader = csv.reader(doc)
        for row in reader:
            train_x.append([float(num) for num in row])
    with open('test_out' + str(i) + '.csv', 'r') as doc:
        reader = csv.reader(doc)
        for row in reader:
            train_y.append([float(num) for num in row])
    with open('valid_potentials' + str(i) + '.csv', 'r') as doc:
        reader = csv.reader(doc)
        for row in reader:
            valid_x.append([float(num) for num in row])
    with open('valid_out' + str(i) + '.csv', 'r'):
        reader = csv.reader(doc)
        for row in reader:
            valid_y.append([float(num) for num in row])

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

# Have a decaying learning rate so that a convergence is faster at first and fit is better at the end.
# By trial and error, the simplest exponential decay doesn't work well.
# Trying a method by which the decay occurs at hand-specified intervals
start_rate = 0.125
gs = 0
gs_list = [1, 1, 2, 3, 10, 20, 40, 100, 200, 10000]
ic = 0
learn_rate = tf.Variable(start_rate, trainable=False)
update_learn_rate = tf.assign(learn_rate, tf.multiply(learn_rate, 0.75))

# Set up neural network layers. 
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# First hidden layer
W1 = tf.Variable(tf.random_uniform([bins - 1, bins - 1], -1. / bins, 1. / bins))
B1 = tf.Variable(tf.random_uniform([bins - 1], -1., 1.))
L1 = tf.nn.softplus(tf.matmul(X, W1) + B1)

# Second hidden layer
