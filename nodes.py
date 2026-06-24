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
        #unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalized them
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#common loss class
class Loss:
    #calculates the data and regularization losses given model output and ground truth values (ground truth viene de la muestra)
    def calculate(self, output, y):
        #calculate sample losses
        sample_losses = self.forward(output, y)
        #calculate mean loss
        data_loss = np.mean(sample_losses)
        #return loss
        return data_loss

#cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    #forward pass, toma las predicciones y la ground truth de la muestra
    def forward(self, y_pred, y_true):
        #number of samples in a batch (cuantos batches tengo en la prediccion)
        samples = len(y_pred)
        #clip data to prevent division by 0, we clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values - only if categorical labels (sparse)
        if len(y_true.shape) == 1: #se fija si es un vector tipo (3,) esto es 1, sparse
            correct_confidences = y_pred_clipped[range(samples), y_true]
            #armamos confidences seleccionando de cada sample, que confianzas nos dieron y truth de la muestra
            #probabilidad que la red le asignó a la clase correcta.
            #[
            #[0.7 , 0.1 , 0.2 ],
            #[0.1 , 0.5 , 0.4 ],
            #[0.02, 0.9 , 0.08]
            #]
            #correct = [0,1,1] 
            #correct_confidences = [0.7, 0.5, 0.9]
        
        #mask values - only for one-hot encoded labels, tienen forma (3,3), 2 valores en la shape
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        #y_true = [
        #[1, 0, 0],  # clase 0 correcta
        #[0, 1, 0],  # clase 1 correcta
        #[0, 0, 1]   # clase 2 correcta
        #]
        #y_pred_clipped * y_true = [
        #[0.7, 0.0, 0.0],
        #[0.0, 0.8, 0.0],
        #[0.0, 0.0, 0.4]
        #]
        #correct_confidences = [0.7, 0.8, 0.4], el axis=1 suma filas por separado

        #losses
        negative_log_likelihoods = -np.log(correct_confidences) #evitando losses negativas con el -
        return negative_log_likelihoods

#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
#create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 3)
#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
layer2 = Layer_Dense(3, 3) #lo tomamos como output layer y decimos 3 por las 3 clases 
activation2 = Activation_Softmax() 

#create loss function
loss_function = Loss_CategoricalCrossentropy()

#perform a forward pass of our training data through this layer
layer1.forward(X)
#forward pass through activation func. #takes in output from previous layer
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#ahora tenemos el output de la softmax, probabilities

#perform a forward pass through loss function, it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y) #y son los correct results

#calculate accuracy from output of activation2 and targets, calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
#miramos cada fila (cada muestra) y elegimos la clase con mayor probabilidad
if len(y.shape) == 2: #one hot case, para los y de truth, class targets diferente con varias filas
    y = np.argmax(y, axis=1) #lo aplana y lo convierte de [[1,0,0],[0,1,0],[0,1,0]] a [0,1,1] para sacar la accuracy
accuracy = np.mean(predictions==y)
#comparamos predicciones con el truth value y ponele queda algo así: [0,0,1] == [0,1,1] eso es [True, False, True], promedio de esto