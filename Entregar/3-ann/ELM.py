"""
Basado en: https://github.com/5663015/elm
"""
import numpy as np
from scipy.linalg import pinv2, inv
import pdb

class Elm():
    def __init__(self, hidden_units, x, y, one_hot=True):
        self.hidden_units = hidden_units
        self.x = x
        self.y = y
        self.class_num = np.unique(self.y).shape[0]     
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        self.one_hot = one_hot
        
        # Para la salida softmax (cada elemento de y será un vector con la dimension=nº clases) 
        self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
        for i in range(self.y.shape[0]):
            self.one_hot_label[i, int(self.y[i])] = 1

        # Pesos de entrada W y bias aleatorio
        self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
        self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    # Computar salida de la capa H 
    def __input2hidden(self, x):
        self.temH = np.dot(self.W, x.T) + self.b
        # RELU ACTIVATION
        self.H = self.temH * (self.temH > 0)
        return self.H

    # Computar salida final
    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    def fit(self):

        self.H = self.__input2hidden(self.x)
        self.y_temp = self.one_hot_label

        # Calculo de los pesos de las neuronas de la capa oculta
        self.beta = np.dot(pinv2(self.H.T), self.y_temp)

        # Computar salidas finales
        self.result = self.__hidden2output(self.H)
        # Softmax de resultados
        self.result = np.exp(self.result)/np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

        # Evaluar el acierto del training
        self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
        self.correct = 0
        for i in range(self.y.shape[0]):
            if self.y_[i] == self.y[i]:
                self.correct += 1
        self.train_score = self.correct/self.y.shape[0]
        return self.beta, self.train_score

    def predict(self, x):
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        self.y_ = np.argmax(self.y_, axis=1).reshape(1,x.shape[0])[0]
        return self.y_