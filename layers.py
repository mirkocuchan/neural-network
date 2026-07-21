import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0): #n_inputs is the size of an input
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  #should be n_neurons, n_inputs to get the weights for the input in a row, but this helps us avoid the .T later 
        self.biases = np.zeros((1, n_neurons))

        #set regularization strength, son los lambda
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        #L1 mata exactamente los pesos que no aportan nada a la predicción, y deja vivos los que sí aportan. L2 en ese caso ayuda 
        #a que los que sobreviven no sean enormes. L1 decide quién vive y quién muere, L2 controla el tamaño de los que viven.

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    #backward pass
    def backward(self, dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        #gradients on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    #retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    #set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


#dropout
class Layer_Dropout:
    #init
    def __init__(self, rate):
        #store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    #forward pass
    def forward(self, inputs, training):
        #save input values, we need them for the backward pass
        self.inputs = inputs
        #if not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        #generate and save scaled mask, we divide by self.rate to scale the values up to not change the expected value of the neurons
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate #para que no quede distinto el training con el validation
        #apply mask to output values, 1 means keep neuron output; 0 means drop neuron output. 
        self.output = inputs * self.binary_mask
    #backward pass
    def backward(self, dvalues):
        #gradient on values, we need to apply the mask to the values as well, otherwise the gradient would be wrong.
        self.dinputs = dvalues * self.binary_mask
        #la derivada de output = input * mask respecto a input es simplemente mask