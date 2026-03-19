# Required libraries
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from keras import models, layers
import numpy as np
from abc import ABC, abstractmethod

# Interface for the classes
class MnistClassifierInterface(ABC):    
    @abstractmethod
    def train(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, y):
        pass

# Feef-Forward Neural Network
class MnistClassifierFFNN(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        # Model of the FFNN
        self.model = models.Sequential([
            layers.Flatten(),
            layers.Dense(128, 'relu'),
            layers.Dense(64, 'relu'),
            layers.Dense(10, 'softmax') # output classes
        ])
        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    def train(self, x, y):
        self.model.fit(x, y, batch_size=128, epochs=5, verbose=1)
        print('Model is now trained')
    
    def predict(self, x):
        return np.argmax(self.model.predict(x))

# Convolutional NN
class MnistClassifierCNN(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        # Model of the cnn
        self.model = models.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28, 28)), # adding grayscale channel
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax') # output classes
        ])
        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    def train(self, x, y):
        self.model.fit(x, y, batch_size=128, epochs=3, verbose=1)
        print('Model is now trained')
    
    def predict(self, x):
        return np.argmax(self.model.predict(x))

# Random forest
class MnistClassifierRF(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=50)
    
    def train(self, x, y):
        self.model.fit(x, y)
        print('Model is now trained')
    
    def predict(self, x):
        return self.model.predict(x)[0]

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm):
        # Correct options
        algs = {
            'cnn': MnistClassifierCNN,
            'rf': MnistClassifierRF,
            'nn': MnistClassifierFFNN
        }

        # Handling incorrect options
        if algorithm not in algs:
            raise ValueError(f'{algorithm} not in {list(algs.keys())}')
        self.alg = algorithm

        # Selecting the classifier
        self.model:MnistClassifierInterface = algs[algorithm]()
    
    def train(self, x, y):
        self.model.train(x, y)
    
    def predict(self, x):
        return self.model.predict(x)
