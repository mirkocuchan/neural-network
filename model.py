import numpy as np
from nnfs.datasets import sine_data
from activations import Activation_ReLU, Activation_Linear
from layers import Layer_Dense
from losses import Loss_MeanSquaredError
from optimizers import Optimizer_Adam

X, y = sine_data()

#input "layer", this is not a real layer, but a placeholder for the input data. It doesn't have weights or biases, it just passes the input data to the next layer.
class Layer_Input:
    #forward pass, output of this layer is the input data itself, so we just save it to the output attribute.
    def forward(self, inputs):
        self.output = inputs

#model class
class Model:
    def __init__(self):
        #create a list of network objects
        self.layers = []
    #add objects to the model
    def add(self, layer):
        self.layers.append(layer) #adding layers to our model
    #set loss and optimizer, the * means that the arguments must be passed as keyword arguments, not positional arguments. This is a way to enforce clarity in the code, making it explicit which argument is being set. To use this method, you would call it like model.set(loss=loss_function, optimizer=optimizer), clearly indicating which argument is which. This can help prevent mistakes and improve code readability. 
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    #train the model
    def train(self, X, y, *, epochs=1, print_every=1):
        #initialize accuracy object
        self.accuracy.init(y)

        #main training loop
        for epoch in range(1, epochs+1):
            #perform the forward pass
            output = self.forward(X)

            #calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss
            
            #get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            
            exit()

    #finalize the model
    def finalize(self):
        #create and set the input layer
        self.input_layer = Layer_Input()
        #count all the objects
        layer_count = len(self.layers)
        #initialize a list containing trainable layers, not every layer has weights and biases, for example activation layers don't have them, so we need to keep track of which layers are trainable. This is important for the backward pass, where we need to update the weights and biases of the trainable layers based on the gradients calculated during backpropagation. By keeping a separate list of trainable layers, we can easily iterate over them and apply the necessary updates without having to check each layer's type or properties. This makes the code cleaner and more efficient.
        self.trainable_layers = []

        #iterate the objects
        for i in range(layer_count):
            #if it's the first layer, the previous layer object is the input layer, acá conectamos todas las capas, la primera capa tiene como previo a la capa de input, y la última capa tiene como next a la loss.
            #las capas del medio tienen como previo a la capa anterior y como next a la capa siguiente. la primera capa tiene como previo a la capa de input y como next a la segunda capa. la segunda capa tiene como previo a la primera capa y como next a la tercera capa. la tercera capa tiene como previo a la segunda capa y como next a la loss.
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            #all layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            #the last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                #save aside the reference to the last object whose output is the model's output, we are going to use it during the backward pass, because we need to calculate the gradient of the loss with respect to the output of the model, which is the output of the last layer. By saving a reference to this layer, we can easily access its output during backpropagation and compute the necessary gradients for updating the weights and biases of the trainable layers.

            #if layer contains an attribute called "weights", it's a trainable layer - add it to the list of trainable layers. We don't need to check for biases - checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        #update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

    #performs forward pass
    def forward(self, X):
        #call forward method on the input layer, this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X)
        #call forward method of every object in a chain, pass output of the previous object as a parameter. This is valid because the forward method is defined in the same way for all objects, so we can call it on any object and pass the output of the previous object as a parameter. This is a common pattern in neural networks, where each layer takes the output of the previous layer as its input. The forward method of each layer computes its output based on its inputs and parameters (weights and biases), and this output is then passed to the next layer in the chain. This allows us to build complex models by stacking layers on top of each other, with each layer transforming the data in some way before passing it on to the next layer.
        for layer in self.layers:
            layer.forward(layer.prev.output)
            # "layer" is now the last object from the list, so its output will be the output of the model. We return it at the end of the method, so we can use it for loss calculation and accuracy calculation. The output of the model is the output of the last layer, which is the input to the loss function. The loss function will then calculate the loss based on this output and the true labels.
        # return its output
        return layer.output
    
    #performs backward pass
    def backward(self, output, y):
        #first call backward method on the loss, this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)
        #call backward method going through all the objects in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


#common accuracy class
class Accuracy:
    #calculates an accuracy given predictions and ground truth values
    def calculate(self, predictions, y):
        #get comparison results
        comparisons = self.compare(predictions, y)
        #calculate an accuracy
        accuracy = np.mean(comparisons)
        #return accuracy
        return accuracy# Accuracy calculation for regression model

class Accuracy_Regression(Accuracy):
    def __init__(self):
        #create precision property
        self.precision = None
    #calculates precision value based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    #compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

#instantiate the model
model = Model()
#add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

#set loss and optimizer objects
model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),)

#finalize the model
model.finalize()

model.train(X, y, epochs=10000, print_every=100)
