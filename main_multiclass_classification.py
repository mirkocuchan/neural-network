import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Sigmoid
from losses import Activation_Softmax_Loss_CategoricalCrossentropy, Loss_BinaryCrossentropy
from optimizers import Optimizer_Adam, Optimizer_SGD
 
nnfs.init()
#this also contains binary logistic regression, but we are going to use it for multiclass classification. 
#there is code in comments that covers blr and the code is tabbed

#training
#create dataset, 100 feature sets and 3 classes and each feature set has 2 fetures, like (a, b) = featureSet1 (we have 300)
X, y = spiral_data(samples=100, classes=3)
    #create dataset for Binary Logistic Regression and Binary Cross-Entropy Loss
    #X, y = spiral_data(samples=100, classes=2)
    #reshape labels to be a list of lists, inner list contains one output (either 0 or 1) per each output neuron, 1 in this case
    #y = y.reshape(-1, 1)

#create Dense layer with 2 input features and 64 output values
layer1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#create dropout layer
dropout1 = Layer_Dropout(0.1)
    #create second Dense layer with 64 input features (as we take output of previous layer here) and 1 output value
    #if we are doing binary classification, we dont use dropout in the last layer.
    #dense2 = Layer_Dense(64, 1) to do binary classification, we need 1 output neuron, not 3.
    #create Sigmoid activation:
    #activation2 = Activation_Sigmoid()
    #create loss function for binary classification
    #loss_function = Loss_BinaryCrossentropy()
    #create optimizer, we are using Adam optimizer, which is a combination of RMSprop and Momentum. It adapts the learning rate for each parameter. Same as before
    #optimizer = Optimizer_Adam(decay=5e-7)
#create second Dense layer with 64 input features (as we take output of previous layer here) and 3 output values
layer2 = Layer_Dense(64, 3) #3 output values (output values)

#create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    #for the slower version
    #activation2 = Activation_Softmax() 
    #create loss function
    #loss_function = Loss_CategoricalCrossentropy()

    #create optimizer
    #optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
    #adagard optimizer option
    #optimizer = Optimizer_Adagrad(decay=1e-4)
    #rms prop optimizer opotion
    #optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)

#optimizer adam option
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5) #(best one so far)

#train in loop
for epoch in range(10001):
    #perform a forward pass of our training data through this layer
    layer1.forward(X)
    #forward pass through activation func. #takes in output from previous layer
    activation1.forward(layer1.output)
    #perform a forward pass through Dropout layer
    dropout1.forward(activation1.output)
    
    #perform a forward pass through second Dense layer, takes outputs of activation function of first layer as inputs
    layer2.forward(dropout1.output)

        #slower version
        #activation2.forward(layer2.output)
        #ahora tenemos el output de la softmax, probabilities
        #perform a forward pass through loss function, it takes the output of second dense layer here and returns loss
        #loss = loss_function.calculate(activation2.output, y) #y son los correct results

    #perform a forward pass through the activation/loss function, it takes the output of second dense layer here and returns loss
    data_loss = loss_activation.forward(layer2.output, y)

    #calculate regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(layer1) + loss_activation.loss.regularization_loss(layer2)
    #calculate overall loss
    loss = data_loss + regularization_loss
    
        #calculate accuracy from output of activation2 and targets
        #part in the brackets returns a binary mask - array consisting of True/False values, multiplying it by 1 changes it into array of 1s and 0s
        #predictions = (activation2.output > 0.5) * 1, clasifico las probabilidades de la softmax en 0 o 1, si es mayor a 0.5 es 1, sino es 0. Esto es para binary classification, no para multiclass.
        #accuracy = np.mean(predictions==y), es lo correcto o no, devuelve un array de True/False, np.mean lo convierte en un porcentaje de aciertos, cuantos hizo bien de todos los samples. np.mean(True) = 1, np.mean(False) = 0, np.mean([True, False, True]) = 0.6667
    
    #calculate accuracy from output of activation2 and targets, calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    #miramos cada fila (cada muestra) y elegimos la clase con mayor probabilidad
    if len(y.shape) == 2: #one hot case, para los y de truth, class targets diferente con varias filas
        y = np.argmax(y, axis=1) #lo aplana y lo convierte de [[1,0,0],[0,1,0],[0,1,0]] a [0,1,1] para sacar la accuracy
    accuracy = np.mean(predictions==y)
    #comparamos predicciones con el truth value y ponele queda algo así: [0,0,1] == [0,1,1] eso es [True, False, True], promedio de esto

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}' + f'reg_loss: {regularization_loss:.3f}), ' +
         f'lr: {optimizer.current_learning_rate}')

    #backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    dropout1.backward(layer2.dinputs)
    activation1.backward(dropout1.dinputs)
    layer1.backward(activation1.dinputs)
    
    #update weights and biases
    optimizer.pre_update_params()

    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()


#validating the model
#create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
#perform a forward pass of our testing data through this layer
layer1.forward(X_test)
#perform a forward pass through activation function, takes the output of first dense layer here
activation1.forward(layer1.output)
#perform a forward pass through second Dense layer, takes outputs of activation function of first layer as inputs
layer2.forward(activation1.output)
#perform a forward pass through the activation/loss function, takes the output of second dense layer here and returns loss
loss = loss_activation.forward(layer2.output, y_test)
#calculate accuracy from output of activation2 and targets, calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    #validate the model but for binary classification
    #create test dataset
    #X_test, y_test = spiral_data(samples=100, classes=2)
    #reshape labels to be a list of lists, inner list contains one output (either 0 or 1) per each output neuron, 1 in this case
    #y_test = y_test.reshape(-1, 1)

    #perform a forward pass of our testing data through this layer
    #dense1.forward(X_test)
    #perform a forward pass through activation function, takes the output of first dense layer here
    #activation1.forward(dense1.output)
    #perform a forward pass through second Dense layer, takes outputs of activation function of first layer as inputs
    #dense2.forward(activation1.output)
    #perform a forward pass through activation function, takes the output of second dense layer here
    #activation2.forward(dense2.output)
    #calculate the data loss
    #loss = loss_function.calculate(activation2.output, y_test)
    #calculate accuracy from output of activation2 and targets part in the brackets returns a binary mask - array consisting of True/False values, multiplying it by 1 changes it into array of 1s and 0s
    #predictions = (activation2.output > 0.5) * 1
    #accuracy = np.mean(predictions==y_test)
    #print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
