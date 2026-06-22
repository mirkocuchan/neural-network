import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): #n_inputs is the size of an input
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  #should be n_neurons, n_inputs to get the weights for the input in a row, but this helps us avoid the .T later 
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
#create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

#perform a forward pass of our training data through this layer
layer1.forward(X)
activation1.forward(layer1.output)

