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
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    #train the model
    def train(self, X, y, *, epochs=1, print_every=1):
        #main training loop
        for epoch in range(1, epochs+1):
            #perform the forward pass
            output = self.forward(X)
            #temporary
            print(output)
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
