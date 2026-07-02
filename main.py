import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Layer_Dense
from activations import Activation_ReLU
from losses import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers import Optimizer_Adam
 
nnfs.init()

#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
#create Dense layer with 2 input features and 64 output values
layer1 = Layer_Dense(2, 64)
#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#create second Dense layer with 64 input features (as we take output of previous layer here) and 3 output values
layer2 = Layer_Dense(64, 3) #3 output values (output values)

#create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    #for the slower version
    #activation2 = Activation_Softmax() 
    #create loss function
    #loss_function = Loss_CategoricalCrossentropy()

#create optimizer
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
#adagard optimizer option
#optimizer = Optimizer_Adagrad(decay=1e-4)
#rms prop optimizer opotion
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
#optimizer adam option
#optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7) (best one so far)

#train in loop
for epoch in range(10001):
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

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}' + f'lr: {optimizer.current_learning_rate}')

    #backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    #update weights and biases
    optimizer.pre_update_params()

    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()
