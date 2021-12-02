from numpy import *
from sklearn.discriminant_analysis import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import *


def ejercicio1_2(dataset):

    x=loadtxt(dataset)

    # Preprocesamiento
    y=x[:,0]-1; x=delete(x,0,1)
    x = (x-mean(x,0))/std(x,0)
    C = len(unique(y))

    # Entrenamiento
    modelo = LinearDiscriminantAnalysis().fit(x,y)

    # Test
    z = modelo.predict(x)
    kappa = 100 * cohen_kappa_score(y,z)
    acc = 100 * accuracy_score(y,z)
    print("Kappa: " + str(kappa) + " acc: " + str(acc))
    cf=confusion_matrix(y,z)
    print("Matriz de confusion: \n", cf)
    
    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec = 100 * precision_score(y,z)
        rec = 100 * recall_score(y,z)
        f1 = 100 * f1_score(y,z)
        print("Precision:", prec, "Rec:", rec, "f1:", f1)


# -------------------------------------------------------------------------------------
print("EJERCICIO 1 y 2  !!!!\n")

# PARTE 1: Para WINE dataset      
print("-> WINE DATASET")                           
ejercicio1_2("wine.data")

# -------------------------------------------------------------------------------------
# PARTE 2: Para HEPATITIS dataset    
print("\n-> HEPATITIS DATASET")                              
ejercicio1_2("hepatitis.data")
