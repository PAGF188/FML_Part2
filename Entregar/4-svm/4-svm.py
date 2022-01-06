"""
Practica 3: Exercises of SVM classifier
Autor: Pablo García Fernández.

"""

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from sklearn.metrics import *
from sklearn.svm import *

def pr4(dataset_name, dataset_train=None, dataset_test=None):
    """ Función que ejecuta los 5 ejercicios mencionados
    en la practica 2.
    - ej2: Calcular acc., kappa y cm usando SVM (linear kernel) sobre todo el dataset (hepatitis, wine).
    - ej3: Repetir con kernel gaussiano (L=100, sigma=1/n)
    - ej4: Repetir con cross validation (k=4, k=10)
    - ej5: 
    """

    print("=============================================")
    print(f"   Ejecucion pr3 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # EJERCICIO 4 -----------------------------------------------------------------
    # 
    if dataset_name == 'LBP' or dataset_name == 'Coocur':
        pass
    # EJERCICIO 2,3 (sobre wine y hepatitis)
    else:
        ej2(dataset_name)


def ej2(dataset_name):
    """
    EJERCICIO 2:
    Calcular acc., kappa y cm usando SVM (linear kernel) sobre todo el dataset (hepatitis, wine).
    """

    print("EJERCICIO 2 (SVM  kernel lineal dataset completo train/test):")
    datos = np.loadtxt('../data/' + dataset_name)

    # Preprocesamiento
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    x = (x-np.mean(x,0))/np.std(x,0)
    C = len(np.unique(y))

    # Entrenamiento
    modelo=SVC(kernel = 'linear', verbose=False).fit(x,y)

    # Test
    z = modelo.predict(x)
    kappa = 100 * cohen_kappa_score(y,z); acc = 100 * accuracy_score(y,z)
    cf=confusion_matrix(y,z)
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z)
        f1 = 100 * f1_score(y,z)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n")            
    print("─────────────────────────────────────────")

# Ejecutamos
pr4("wine.data")
pr4("hepatitis.data")
pr4("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr4("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')


