# Nota: Ejercicio 3 de la primera práctica.

from numpy import *
from sklearn.neighbors import *
from sklearn.metrics import *
from crea_folds import *

def ejercicio3(dataset_train,dataset_test):
    data_train = loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # Juntamos ambos sets para cross-validarion
    x = concatenate([x_train,x_test],axis=0)
    y = concatenate([y_train,y_test])

    # Generamos K folds
    K_=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K_)

    # Juntamos train y validación (en este ejercicio no hay sintonización de params.)
    for k in range(K_):
        tx[k]=vstack((tx[k],vx[k]))
        ty[k]=concatenate((ty[k],vy[k]))

        # preprocesamos
        med=mean(tx[k],0); dev=std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    # Ejecutamos KNN con V vecinos
    V = 1
    C = len(unique(y))

    mc = zeros([C,C])
    v_kappa=zeros(K_)
    v_accuracy = zeros(K_)

    for k in range(K_):
        modelo = KNeighborsClassifier(n_neighbors=V).fit(tx[k],ty[k])
        z = modelo.predict(sx[k]); y=sy[k]
        v_kappa[k] = 100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = mean(v_kappa)
    accuracy = mean(v_accuracy)
    mc/=K_

    print("Kappa: " + str(kappa) + " acc: " + str(accuracy))
    print("Matriz de confusion: \n", mc)


# -------------------------------------------------------------------------------------
print("EJERCICIO 3 !!!!\n")

# PARTE 1: PARA Coocurrence matrix      
print("-> COOCURRENCE MATRIX")                           
dataset_train = 'trainCoocur.dat'
dataset_test = 'testCoocur.dat'
ejercicio3(dataset_train,dataset_test)

# -------------------------------------------------------------------------------------
# PARTE 2: PARA Local Binary Patterns      
print("\n-> LOCAL BINARY PATTERNS")                              
dataset_train = 'trainLBP.dat'
dataset_test = 'testLBP.dat'
ejercicio3(dataset_train,dataset_test)