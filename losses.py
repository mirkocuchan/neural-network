import numpy as np
from activations import Activation_Softmax

#common loss class
class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    #regularization loss calculation, what is regularization loss? it is a penalty term added to the loss function to prevent overfitting by discouraging complex models. It helps to keep the model weights small and reduces the risk of overfitting to the training data.
    def regularization_loss(self, layer):
        #0 by default
        regularization_loss = 0
        #calculate regularization los, iterate all trainable layers
        for layer in self.trainable_layers:
            #L1 regularization - weights, calculate only when factor (lambda) greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            #L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            #L1 regularization - biases calculate only when factor (lambda) greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            #L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    #calculates the data and regularization losses given model output and ground truth values (ground truth viene de la muestra)
    def calculate(self, output, y):
        #calculate sample losses
        sample_losses = self.forward(output, y)
        #calculate mean loss
        data_loss = np.mean(sample_losses)
        #return loss
        return data_loss, self.regularization_loss()

#cross-entropy loss, what is it? it is a loss function commonly used for multi-class classification. 
#it measures the difference between the predicted probability distribution and the true distribution (ground truth) of the classes. The goal is to minimize this loss, which indicates that the model's predictions are getting closer to the actual labels.
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

#binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    #forward pass, toma las predicciones y la ground truth de la muestra
    def forward(self, y_pred, y_true):
        #clip data to prevent division by 0, clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #calculate sample-wise loss, how? L=−ylog(y^​)−(1−y)log(1−y^​)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1) #why? because we want to average the loss across all output neurons for each sample, giving us a single loss value per sample. 
        #This is especially important in multi-output scenarios where each output contributes to the overall loss.
        #return losses
        return sample_losses
    #backward pass
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample, we'll use the first sample to count them
        outputs = len(dvalues[0])
        #clip data to prevent division by 0, clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        #calculate gradient, why this way? because the derivative of the binary cross-entropy loss with respect to the predicted output is given by the formula: ∂L/∂y^​ = -(y / y^​) + ((1 - y) / (1 - y^​)). This formula captures how the loss changes with respect to small changes in the predicted output, and it is derived from the definition of the binary cross-entropy loss function.
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples

#mean Squared Error loss, what is it? it is a loss function that measures the average squared difference between the predicted values and the actual values. It is commonly used for regression tasks, where the goal is to predict continuous values. The MSE loss penalizes larger errors more than smaller ones, making it sensitive to outliers. The formula for MSE is given by: MSE = (1/n) * Σ(y_true - y_pred)^2, where n is the number of samples, y_true are the true values, and y_pred are the predicted values.
class Loss_MeanSquaredError(Loss): # L2 loss
    #forward pass
    def forward(self, y_pred, y_true):
        #calculate loss, why axis=-1? Specifying axis=-1 sums the squared differences across all output neurons for each sample, resulting in a single loss value per sample.
        #axis=-1 generates a mean for each row.  
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        #return losses, why more than one loss? because we want to calculate the loss for each sample in the batch, giving us a single loss value per sample. This is especially important in multi-output scenarios where each output contributes to the overall loss.
        return sample_losses
        #se calcula el mean de estas muestras en la parte de calculate de la loss. ahora tenemos un loss por cada sample, que es lo que queremos, y luego se hace el mean de todos los samples para tener un solo loss. este loss de cada sample fue calculado con el mean de cada output de la sample, porque queremos un solo loss por sample, no por output. si tenemos 3 outputs, queremos un solo loss que represente a los 3 outputs, y luego un solo loss que represente a todas las samples.
    #backward pass
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample, we'll use the first sample to count them
        outputs = len(dvalues[0])
        #calculate gradient on values, formula: ∂L/∂y^​ = -2 * (y^​ - y) / n, where n is the number of outputs. This formula captures how the loss changes with respect to small changes in the predicted output, and it is derived from the definition of the mean squared error loss function.
        self.dinputs = -2 * (y_true - dvalues) / outputs
        #normalize gradient, why? because we want to average the gradient across all output neurons for each sample, giving us a single gradient value per sample. This is especially important in multi-output scenarios where each output contributes to the overall gradient.
        self.dinputs = self.dinputs / samples

#mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss
    def forward(self, y_pred, y_true):
        #calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    #backward pass
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample, we'll use the first sample to count them
        outputs = len(dvalues[0])
        #calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples