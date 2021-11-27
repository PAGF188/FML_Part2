# Nota: Ejercicios 1 y 2 de la primera práctica.

from numpy import *
from sklearn.neighbors import *
from sklearn.metrics import *

def ejercicio1_y_2(dataset_train,dataset_test):
    """Ejercicio 1 y 2
    
    :param dataset_train archivo de datos de train
    :param dataset_test archivo dde datos de test

    Nota: Las etiquetas deben estar en la ÚLTIMA COLUMNA!
    
    """
    data_train = loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # Preprocesamiento
    med = mean(x_train,0); dev = std(x_train,0)
    x_train = (x_train-med)/dev
    x_test = (x_test-med)/dev

    # Ejecutamos KNN con V vecinos
    V=1
    modelo=KNeighborsClassifier(n_neighbors=V).fit(x_train,y_train)
    z=modelo.predict(x_test)

    # Evaluación de resultados.
    acc = 100 * accuracy_score(y_test,z)
    kappa = 100 * cohen_kappa_score(y_test,z)
    cf=confusion_matrix(y_test,z)

    print("Kappa: " + str(kappa) + " acc: " + str(acc))
    print("Matriz de confusion: \n", cf)

# -------------------------------------------------------------------------------------
print("EJERCICIO 1 Y 2 !!!!\n")

# PARTE 1: PARA Coocurrence matrix      
print("-> COOCURRENCE MATRIX")                           
dataset_train = 'trainCoocur.dat'
dataset_test = 'testCoocur.dat'
ejercicio1_y_2(dataset_train,dataset_test)

# -------------------------------------------------------------------------------------
# PARTE 2: PARA Local Binary Patterns      
print("\n-> LOCAL BINARY PATTERNS")                              
dataset_train = 'trainLBP.dat'
dataset_test = 'testLBP.dat'
ejercicio1_y_2(dataset_train,dataset_test)