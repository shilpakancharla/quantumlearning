# quantumlearning
Applications of Machine Learning in Quantum Mechanics. 

## Abstract

Computational models that use machine learning principles to solve quantum mechanics problems promise to deliver accurate as possible information about a quantum system at high speeds. These two fields are related to each other by linear algebra, and it is imperative we examine the redundancy in solving Schrödinger’s equations throughout different types of potential situations: step, piecewise linear and random Fourier series. This project uses TensorFlow package in Python to initially generate 1-D potentials and then solve them using a gradient descent method. These potentials and their respective solutions are partitioned into sets of training data and test data. The training data is inputted into a simple neural network with two hidden layers. The mean square distance between the “correct” solutions and the output of the neural network is the cost function and the gradient descent on the network “solves” the problem. 


## Main ML Files in Repository

- Schrodinger Neural Network.ipynb

- New Potentials.ipynb

- gradient_descent.py

- 1-D graphs.ipynb


## Future Directions

Please contact me if you are interested in extending/improving this project in any of the further points! 😄

- extend study to three dimension solutions  😳

- increase layers in neural network 🧠

- using neural network for other aspects of Schrödinger’s equations (tell us other information about quantum state) 🔮

- cross-validation techniques 📈

- feature engineering to further reduce errors 📉
