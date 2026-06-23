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

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
#create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 3)
#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3, 3) #lo tomamos como output layer y decimos 3 por las 3 clases 
activation2 = Activation_Softmax() 

#perform a forward pass of our training data through this layer
layer1.forward(X)
#forward pass through activation func. #takes in output from previous layer
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#ahora tenemos el output de la softmax, probabilities

