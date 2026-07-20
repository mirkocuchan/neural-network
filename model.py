import numpy as np
from nnfs.datasets import sine_data, spiral_data
from activations import Activation_ReLU, Activation_Linear, Activation_Sigmoid, Activation_Softmax
from layers import Layer_Dense, Layer_Dropout
from losses import Loss_MeanSquaredError, Loss_BinaryCrossentropy, Loss_CategoricalCrossentropy, Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers import Optimizer_Adam
from accuracy import Accuracy_Regression, Accuracy_Categorical

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
        #softmax classifier's output object
        self.softmax_classifier_output = None

    #add objects to the model
    def add(self, layer):
        self.layers.append(layer) #adding layers to our model
    #set loss and optimizer, the * means that the arguments must be passed as keyword arguments, not positional arguments. This is a way to enforce clarity in the code, making it explicit which argument is being set. To use this method, you would call it like model.set(loss=loss_function, optimizer=optimizer), clearly indicating which argument is which. This can help prevent mistakes and improve code readability. 
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    #train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        #initialize accuracy object
        self.accuracy.init(y)
        #if there is the validation data, initialize the accuracy object with it as well
        train_steps = 1
        #if there is validation data passed, set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            #for better readability
            X_val, y_val = validation_data

        #calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            #dividing rounds down. if there are some remaining data, but not a full batch, this won't include it. add 1 to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                #dividing rounds down. if there are some remaining data, but nor full batch, this won't include it. add 1 to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        #main training loop
        for epoch in range(1, epochs+1):
            #print epoch number
            print(f'epoch: {epoch}')
            #reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            #iterate over steps
            for step in range(train_steps):
                #if batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                #otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                #perform the forward pass
                output = self.forward(batch_X, training=True)

                #calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                #get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                #perform backward pass
                self.backward(output, batch_y)

                #optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                #print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f} (' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')

            #get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' + f'acc: {epoch_accuracy:.3f}, ' + f'loss: {epoch_loss:.3f} (' + f'data_loss: {epoch_data_loss:.3f}, ' + f'reg_loss: {epoch_regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')

            #if there is the validation data
            if validation_data is not None:

                #reset accumulated values in loss and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                #iterate over steps
                for step in range(validation_steps):
                    #if batch size is not set, train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    #otherwise slice a batch
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]

                    #for better readability
                    #perform the forward pass
                    output = self.forward(batch_X, training=False)
                    #calculate the loss
                    loss = self.loss.calculate(output, batch_y)
                    #get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)
                #get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                #print a summary 
                print(f'validation, ' + f'acc: {validation_accuracy:.3f}, ' + f'loss: {validation_loss:.3f}')

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

        #if output activation is Softmax and loss function is Categorical Cross-Entropy, create an object of combined activation
        #and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            #create an object of combined activation and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
        #update loss object with trainable layers

    #performs forward pass
    def forward(self, X, training):
        #call forward method on the input layer, this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        #call forward method of every object in a chain, pass output of the previous object as a parameter. This is valid because the forward method is defined in the same way for all objects, so we can call it on any object and pass the output of the previous object as a parameter. This is a common pattern in neural networks, where each layer takes the output of the previous layer as its input. The forward method of each layer computes its output based on its inputs and parameters (weights and biases), and this output is then passed to the next layer in the chain. This allows us to build complex models by stacking layers on top of each other, with each layer transforming the data in some way before passing it on to the next layer.
        for layer in self.layers:
            layer.forward(layer.prev.output)
        # "layer" is now the last object from the list, so its output will be the output of the model. We return it at the end of the method, so we can use it for loss calculation and accuracy calculation. The output of the model is the output of the last layer, which is the input to the loss function. The loss function will then calculate the loss based on this output and the true labels.
        # return its output
        return layer.output
    
    #performs backward pass
    def backward(self, output, y):
        #if we used softmax classifier and categorical loss entropy
        if self.softmax_classifier_output is not None:
            #first call backward method on the combined activation/loss: this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            #since we'll not call backward method of the last layer (which is Softmax activation) as we used combined activation/loss object, let's set dinputs in this object
            #anterior a softmax necesita, para calcular su propio dinputs, el dinputs de la capa que le sigue (Softmax)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            #call backward method going through all the objects but last in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
    
        #first call backward method on the loss, this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)
        #call backward method going through all the objects in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

#create train and test dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

#instantiate the model
model = Model()
#add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

#set loss, optimizer and accuracy objects
model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5), accuracy=Accuracy_Categorical())

#finalize the model
model.finalize()

#train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)