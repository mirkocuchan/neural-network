import numpy as np

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