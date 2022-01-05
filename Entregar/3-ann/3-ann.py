"""
Practica 3: Exercises of ANN classifiers
Autor: Pablo García Fernández.

"""

import numpy as np
from sklearn.neighbors import *
from sklearn.metrics import *
from crea_folds import crea_folds

def pr3(dataset_name, dataset_train=None, dataset_test=None):
    """ Función que ejecuta los 4 ejercicios mencionados
    en la practica 2.
    - ej2: Calcular acc., kappa y cm usando MLP y ELM sobre todo el dataset (hepatitis, wine).
    - ej3: Repetir con cross-validation k=4
    - ej4: Usar MLP y ELM en LBP y CooCu datasets
    """

    print("=============================================")
    print(f"   Ejecucion pr3 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # EJERCICIO 4 -----------------------------------------------------------------
    # 
    if dataset_name == 'LBP' or dataset_name == 'Coocur':
        print(f"EJERCICIO 5 (LDA sobre {dataset_name} dataset)")
        print("─────────────────────────────────────────")

        # Cargar datos
        data_train = np.loadtxt(dataset_train)
        y_train = data_train[:,-1]
        x_train = data_train[:,0:-1]

        data_test = np.loadtxt(dataset_test)
        y_test = data_test[:,-1]
        x_test = data_test[:,0:-1]

    # EJERCICIO 2,3 (sobre wine y hepatitis)
    else:
        # EJERCICIO 2 -------------------------------------------------------------
        # Calcular acc., kappa y cm usando LDA y todo el dataset
        print("EJERCICIO 2 (LDA whole dataset):")
        
        
        print(f"acc. (con sklearn): {acc:.2f}%")  # mismo resultado
        print("─────────────────────────────────────────\n\n\n")

        # EJERCICIO 3 -------------------------------------------------------------
        # Calcular acc., kappa y cm usando MLP y ELM con cross validation.
        print("EJERCICIO 2 (LDA whole dataset):")
        
        
        print(f"acc. (con sklearn): {acc:.2f}%")  # mismo resultado
        print("─────────────────────────────────────────\n\n\n")



# Ejecutamos
pr3("wine.data")
pr3("hepatitis.data")
pr3("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr3("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')
