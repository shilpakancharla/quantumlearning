"""
    Reads training data from genpotential.py and then initalizes a neural network with two layers.
"""

import os
import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

bins = 128
seedmax = 20 # Open seed files 0 to 19
current_seed = 0
train_x = []
train_y = []
valid_x = []
valid_y = []

# Reading in data
os.chdir('data')
with open('test_potentials' + str(current_seed) + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_x.append([float(num) for num in row])
    csvfile.close()
with open('test_out' + str(current_seed) + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_y.append([float(num) for num in row])
    csvfile.close()
with open('valid_potentials' + str(current_seed) + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        valid_x.append([float(num) for num in row])
    csvfile.close()
with open('valid_out' + str(current_seed) + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        valid_y.append([float(num) for num in row])
    csvfile.close()

os.chdir('../')
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
W2 = tf.Variable(tf.random_uniform([bins - 1, bins - 1], -1. / bins, 1. / bins))
B2 = tf.Variable(tf.random_uniform([bins - 1], -1., 1.))
L2 = tf.nn.softplus(tf.matmul(L1, W2) + B2)

# Output layer
W3 = tf.Variable(tf.random_uniform([bins - 1, bins - 1], -1. / bins, 1. / bins))
B3 = tf.Variable(tf.random_uniform([bins - 1], -1., 1.))
L3 = tf.nn.softplus(tf.matmul(L2, W3) + B3)

# Cost function
cost_function = tf.reduce_mean(tf.square(tf.subtract(L3, Y)))
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
train_step = optimizer.minimize(cost_function)

# Initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100000):
    if step % 150 == 0:
        if ic == gs_list[gs]:
            gs = gs + 1
            ic = 1
            sess.run(update_learn_rate)
        else:
            ic = ic + 1
    if step % 100 == 0:
        print(step, 'Train loss: ', sess.run(cost_function, feed_dict={X: train_x, Y: train_y}),
            'Valid loss: ', sess.run(cost_function, feed_dict={X: valid_x, Y: valid_y}))
    sess.run(train_step, feed_dict={X: train_x, Y: train_y})