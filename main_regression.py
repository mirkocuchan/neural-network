import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data

from layers import Layer_Dense 
from activations import Activation_ReLU, Activation_Linear
from losses import Loss_MeanSquaredError
from optimizers import Optimizer_Adam 
 
nnfs.init()
#create dataset
X, y = sine_data()
#create Dense layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

#create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#create second Dense layer with 64 input features (as we take output of previous layer here) and 64 output values
dense2 = Layer_Dense(64, 64)
#create ReLU activation (to be used with Dense layer):
activation2 = Activation_ReLU()
#create third Dense layer with 64 input features (as we take output of previous layer here) and 1 output value
dense3 = Layer_Dense(64, 1)

#create Linear activation for the output of the third layer (to be used with Dense layer):
activation3 = Activation_Linear()

#create loss function
loss_function = Loss_MeanSquaredError()

#create optimizer
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
#accuracy precision for accuracy calculation
#there are no really accuracy factor for regression problem, but we can simulate/approximate it. We'll calculate it by checking how many values have a difference to their ground truth equivalent less than given precision
#we'll calculate this precision as a fraction of standard deviation of al the ground truth values
#why standard deviation? because it gives us an idea of how spread out the values are. If the values are very spread out, we can allow a larger difference between predicted and true values. If they are very close together, we want a smaller difference.
#si estan proximos los valores, queremos una diferencia menor. si estan muy dispersos, podemos permitir una diferencia mayor, ya que la prediccion puede estar mas lejos del valor real y seguir siendo buena.
accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    #perform a forward pass of our training data through this layer
    dense1.forward(X)
    #perform a forward pass through activation function, takes the output of first dense layer here
    activation1.forward(dense1.output)
    #perform a forward pass through second Dense layer, takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    #perform a forward pass through activation function, takes the output of second dense layer here
    activation2.forward(dense2.output)
    #perform a forward pass through third Dense layer, takes outputs of activation function of second layer as inputs
    dense3.forward(activation2.output)
    #perform a forward pass through activation function takes the output of third dense layer here
    activation3.forward(dense3.output)
    #calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y)

    #calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)
    #calculate overall loss
    loss = data_loss + regularization_loss
    #calculate accuracy from output of activation3 and targets.
    #to calculate it we're taking absolute difference between predictions and ground truth values and compare if differences are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f} (' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {regularization_loss:.3f}), ' + f'lr: {optimizer.current_learning_rate}')
    
    #backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()