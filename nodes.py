import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5], 
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]
#input data to the neural network
#normalize and scale the data? 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): #n_inputs is the size of an input
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  #should be n_neurons, n_inputs to get the weights for the input in a row, but this helps us avoid the .T later 
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

