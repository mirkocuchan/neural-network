from weakref import finalize
from zipfile import ZipFile
import os
import urllib
import urllib.request
import numpy as np
import cv2
from accuracy import Accuracy_Categorical
from losses import Loss_CategoricalCrossentropy
from model import Model
from layers import Layer_Dense
from activations import Activation_ReLU, Activation_Softmax
from optimizers import Optimizer_Adam
from optimizers import Optimizer_Adam

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print('Done!')
#pip install opencv-python

#loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    #scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    #create lists for samples and labels
    X = []
    y = []
    #for each label folder
    for label in labels:
    #and for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
        #read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            #and append it and a label to the lists
            X.append(image)
            y.append(label)
    #convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

#MNIST dataset (train + test)
def create_data_mnist(path):
    #load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    #and return all the data
    return X, y, X_test, y_test

#create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#vamos a shuffle the dataset, to avoid any bias in the order of the samples. definimos keys as a numpy array of integers from 0 to the number of samples in X. This will be used to shuffle the dataset randomly.
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys) #shuffle the keys randomly, so that we can use them to shuffle the dataset.
X = X[keys] #shuffle the dataset using the shuffled keys. This will rearrange the samples in X randomly.
y = y[keys] #shuffle the labels using the same shuffled keys. This will rearrange the labels in y randomly, corresponding to the shuffled samples in X.

#scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
#reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#np.flatten() convierte (60000 × 28 × 28,) = (47,040,000,), en esto: un solo vector gigante, perdiste las muestras
#pythonX = X.reshape(X.shape[0], -1)
# X.shape[0] = 60000   ← mantenés esto
# -1 = aplanás 28×28 = 784
# (60000, 28, 28)  →  (60000, 784)

# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

#set loss, optimizer and accuracy objects
model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=1e-3), accuracy=Accuracy_Categorical())
#finalize the model
model.finalize()
#train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
