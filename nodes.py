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
        self.inputs = inputs

    #backward pass
    def backward(self, dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    
        #backward pass
    def backward(self, dvalues):
        #since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        #zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        #unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalized them
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    #∂L/∂z0 = ∂L/∂S0 * ∂S0/∂z0 (Con la Jacobiana estamos haciendo aS/az)
    #despues de la backpropagation de la loss, le toca a softmax. recibe la matriz con los gradientes por sample ((samples, clases))
    #cuanto afectó cada sample al loss
    #cada fila es el gradiente de la loss respecto a las probabilidades de ese sample
    def backward(self, dvalues):
        #create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        #iteramos sample por sample porque cada uno necesita su propia jacobiana
        #single_output: probabilidades que produjo softmax para este sample [Sclase0, S_clase1, S_clase2]
        #single_dvalues: gradiente de la loss para este sample [∂L/∂S0clase0, ∂L/∂S0clase1, ∂L/∂S0clase2]
        for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, dvalues)):
            #flatten output array para poder hacer el dot product
            single_output = single_output.reshape(-1, 1)
            #la jacobiana captura como cada z afecta a cada probabilidad dentro del S sample
            #tiene forma (clases x clases) — celda [i][j] = ∂Si/∂zj
            #diagflat: pone cada S en la diagonal → representa el término δij * Si
            #dot: todas las combinacio nes Si * Sj → se resta porque softmax, normaliza todo junto, si un S sube los otros bajan
            #la derivada de softmax es S_i*(δij - S_j) = diagflat - dot
            #Tenés 3 clases, entonces tenés 3 valores de z y 3 probabilidades en un sample. 
            #"Si cambio z_j, cuánto cambia S_i?" 3 z y 1 Sample con 3 clases, hay 9 combinaciones posibles. 
            # ∂S_clase0/∂z_clase0   ∂S_clase0/∂z_clase1   ∂S_clase0/∂z_clase2 Tenés que mirar cómo z0 afectó a todas las probabilidades — S(clases0, 1 y 2) — porque Softmax las mezcla todas.
            # ∂S_clase1/∂z_clase0   ∂S_clase1/∂z_clase1   ∂S_clase1/∂z_clase2
            # ∂S_clase2/∂z_clase0   ∂S_clase2/∂z_clase1   ∂S_clase2/∂z_clase2
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            #calculate sample-wise gradient and add it to the array of sample gradients
            #regla de la cadena: jacobiana (3x3) · dvalues (3,) = dinputs (3,)
            #multiplica cada fila de la jacobiana por el gradiente de la loss y los suma → convierte 
            #el gradiente respecto a S en gradiente respecto a z
            #un grad por clase, que le pasamos a la anterior layer
            #cuánto afectó cada z al error.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            #dvalues que vienen de la Loss son la derivada de la Loss respecto a cada probabilidad S:
            #dvalues = [∂L/∂S_clase0, ∂L/∂S_clase1, ∂L/∂S_clase2]
            #"Si cambio la probabilidad de clase0 un poquito, cuánto cambia el error total."

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
    #backward pass, donde el dvalues es el resultado de la red, array of predictions
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of labels in every sample, we'll use the first sample to count them
        labels = len(dvalues[0])
        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        #calculate gradient
        self.dinputs = -y_true / dvalues
        #normalize gradient
        self.dinputs = self.dinputs / samples

#softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    #creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        #output layer's activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    #probabilidades que salieron del Softmax, true results: ​∂L/∂zi​= predicción ​− real
    def backward(self, dvalues, y_true):
        #number of samples en el batch
        samples = len(dvalues)
        #if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient, restando 1 en la posición correcta: ŷ - y
        #we’re taking advantage of the fact that the y being y_true in the code consists of one-hot encoded vectors, 
        #and for each sample, there is only a singular value of 1 in these vectors and the remaining positions are filled with zeros.
        self.dinputs[range(samples), y_true] -= 1
        #normalize gradient
        self.dinputs = self.dinputs / samples

#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
#create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 3)
#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
layer2 = Layer_Dense(3, 3) #lo tomamos como output layer y decimos 3 por las 3 clases 

#create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    #for the slower version
    #activation2 = Activation_Softmax() 
    #create loss function
    #loss_function = Loss_CategoricalCrossentropy()

#perform a forward pass of our training data through this layer
layer1.forward(X)
#forward pass through activation func. #takes in output from previous layer
activation1.forward(layer1.output)

layer2.forward(activation1.output)

    #slower version
    #activation2.forward(layer2.output)
    #ahora tenemos el output de la softmax, probabilities
    #perform a forward pass through loss function, it takes the output of second dense layer here and returns loss
    #loss = loss_function.calculate(activation2.output, y) #y son los correct results

#perform a forward pass through the activation/loss function, it takes the output of second dense layer here and returns loss
loss = loss_activation.forward(layer2.output, y)

#calculate accuracy from output of activation2 and targets, calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
#miramos cada fila (cada muestra) y elegimos la clase con mayor probabilidad
if len(y.shape) == 2: #one hot case, para los y de truth, class targets diferente con varias filas
    y = np.argmax(y, axis=1) #lo aplana y lo convierte de [[1,0,0],[0,1,0],[0,1,0]] a [0,1,1] para sacar la accuracy
accuracy = np.mean(predictions==y)
#comparamos predicciones con el truth value y ponele queda algo así: [0,0,1] == [0,1,1] eso es [True, False, True], promedio de esto

#backward pass
loss_activation.backward(loss_activation.output, y)
layer2.backward(loss_activation.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)