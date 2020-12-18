"""
    Generates random potentials of 3 different types:
    1. Step functions
    2. Piecewise linear functions
    3. Random Fourier series
    Each of these potential types becomes more "jagged" as generation progresses.
    The ground state wavefunction of each potential is found using the gradient
    descent provided by TensorFlow and is applied on each energy function given by
    Schrodinger's equation. These potentials are partitioned into training and 
    validation data sets, and saved with a random seed appended to a filename.
"""

import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
    Subexponential function.
"""
def subexp(exp):
    return np.power(abs(np.log(np.random.uniform())), exp)

"""
    Function for generating potentials.
    1. 0 = step
    2. 1 = linear
    3. 2 = Fourier

    0-1 represents "jaggedness"
"""
def generate_potential(style,param): 
    mu = 1. + bins * param # Mean number of jump points for styles 0 + 1
    forxp = 2.5 - 2 * param # Fourier exponent for style 2
    scale = 5.0 * (np.pi * np.pi * 0.5) # Energy scale
    if style < 2:
        dx = bins / mu
        xlist = [-dx / 2]
        while xlist[-1] < bins:
            xlist.append(xlist[-1] + dx * subexp(1.))
        vlist = [scale * subexp(2.) for k in range(len(xlist))]
        k = 0
        poten = []
        for l in range(1, bins):
            while xlist[k + 1] < l:
                k = k + 1
            if style == 0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k] + (vlist[k + 1] - vlist[k]) *(l - xlist[k]) / (xlist[k + 1] - xlist[k]))
    else:
        sincoef = [(2 * np.random.randint(2) - 1.) * scale * subexp(2.) / np.power(k, forxp) for k in range(1, bins // 2)]
        coscoef = [(2 * np.random.randint(2) - 1.) * scale * subexp(2.) / np.power(k, forxp) for k in range(1, bins // 2)]
        zercoef = scale * subexp(2.)
        poten = np.maximum(np.add(np.add(np.matmul(sincoef, sinval), np.matmul(coscoef, cosval)), zercoef), 0).tolist()
    return poten

seed = 0
np.random.seed(seed)
bins = 128 
npots = 200 
validnth = 5 
sinval = np.sin([[np.pi * i * j / bins for i in range(1, bins)] for j in range(1, bins // 2)])
cosval = np.cos([[np.pi * i * j / bins for i in range(1, bins)] for j in range(1, bins // 2)])
sqrt2 = np.sqrt(2)

defgrdstate = tf.constant([sqrt2 * np.sin(i * np.pi / bins) for i in range(1, bins)])
psi = tf.Variable(defgrdstate)
zerotens = tf.zeros([1])
psil = tf.concat([psi[1:], zerotens], 0)
psir = tf.concat([zerotens, psi[:-1]], 0)
renorm = tf.assign(psi, tf.divide(psi, tf.sqrt(tf.reduce_mean(tf.square(psi)))))
optim = tf.train.GradientDescentOptimizer(0.0625 / bins)
reinit = tf.assign(psi, defgrdstate)
init = tf.global_variables_initializer()

potentials = []
valid_potentials = []
wave_functions = []
valid_functions = []

sess = tf.Session()
sess.run(init)
for i in range(npots):
    if i % 10 == 0:
        print(str((100. * i) / npots) + '% complete')
    for j in range(3):
        vofx = generate_potential(j, (1. * i) / npots)
        energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),
                                            tf.add(vofx, 1. * bins * bins)),
                                            tf.multiply(tf.multiply(tf.add(psil, psir), psi),
                                            0.5 * bins * bins)))
        training = optim.minimize(energy)
        sess.run(reinit)
        for t in range(20000):
            sess.run(training)
            sess.run(renorm)
        if i%validnth == 0:
            valid_potentials.append(vofx)
            valid_functions.append(sess.run(psi).tolist())
        else:
            potentials.append(vofx)
            wave_functions.append(sess.run(psi).tolist())

with open('test_potentials' + str(seed) + '.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(potentials)

with open('valid_potentials' + str(seed) + '.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(valid_potentials)

with open('test_out' + str(seed) + '.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(wave_functions)

with open('valid_out' + str(seed) + '.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(valid_functions)

print('Output complete')