"""
Basado en: https://github.com/5663015/elm
"""
import numpy as np
from scipy.linalg import pinv2
import pdb

class Elm():
    def __init__(self, hidden_units, x, y):
        self.hidden_units = hidden_units
        self.x = x
        self.y = y
        self.class_num = np.unique(self.y).shape[0]     
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        
        # Para la salida softmax (cada elemento de y será un vector con la dimension=nº clases) 
        self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
        for i in range(self.y.shape[0]):
            self.one_hot_label[i, int(self.y[i])] = 1

        # Pesos de entrada W y bias aleatorio
        self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
        self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        self.temH = np.dot(self.W, x.T) + self.b
        # RELU
        self.H = self.temH * (self.temH > 0)
        return self.H

    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    def fit(self):

        self.H = self.__input2hidden(self.x)
        self.y_temp = self.one_hot_label
        # Calculo de los pesos de las neuronas de la capa oculta
        self.beta = np.dot(pinv2(self.H.T), self.y_temp)
        return self.beta

    def predict(self, x):
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        self.y_ = np.argmax(self.y_, axis=1).reshape(1,x.shape[0])[0]
        return self.y_