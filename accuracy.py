import numpy as np

#common accuracy class
class Accuracy:
    #calculates an accuracy given predictions and ground truth values
    def calculate(self, predictions, y):
        #get comparison results
        comparisons = self.compare(predictions, y)
        #calculate an accuracy
        accuracy = np.mean(comparisons)

        #add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        #return accuracy
        return accuracy

    #calculates accumulated accuracy
    def calculate_accumulated(self):
        #calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        #return the data and regularization losses
        return accuracy
    #reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

#accuracy calculation for regression model
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

#accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    #no initialization is needed
    def init(self, y):
        pass
    #compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y