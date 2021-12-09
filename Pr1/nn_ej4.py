# Nota: Ejercicio 4 de la primera práctica.

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
    K=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=mean(tx[k],0); dev=std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    #sintonizacion
    print("% Sintonización:")
    V=[1,3,5,7,9,11]
    vkappa=zeros(K); kappa_mellor=-Inf; V_mellor=V[0]

    print('%10s %10s'%('V','Kappa(%)'))

    for v in V:
        for k in range(K):
            modelo = KNeighborsClassifier(n_neighbors=v).fit(tx[k],ty[k])
            z = modelo.predict(vx[k])
            vkappa[k]=cohen_kappa_score(vy[k],z)
        kappa_med=mean(vkappa)
        print('%10i %10.1f'%(v,100*kappa_med))
        if kappa_med>kappa_mellor:
            kappa_mellor=kappa_med; V_mellor=v

    print('V_mellor=%i kappa=%.2f%%'%(V_mellor,100*kappa_mellor))

    #test
    print("\n% Test:")
    C = len(unique(y))
    mc = zeros([C,C])
    v_accuracy = zeros(K)

    for k in range(K):
        tx[k]=vstack((tx[k],vx[k]))
        ty[k]=concatenate((ty[k],vy[k]))
        modelo = KNeighborsClassifier(n_neighbors=V_mellor).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = mean(vkappa)
    accuracy = mean(v_accuracy)
    mc/=K

    print("Kappa: " + str(kappa) + " acc: " + str(accuracy))
    print("Matriz de confusion: \n", mc)


# -------------------------------------------------------------------------------------
print("EJERCICIO 4 !!!!\n")

# PARTE 1: PARA Coocurrence matrix      
print("-> COOCURRENCE MATRIX\n")                           
dataset_train = 'trainCoocur.dat'
dataset_test = 'testCoocur.dat'
ejercicio3(dataset_train,dataset_test)

# -------------------------------------------------------------------------------------
# PARTE 2: PARA Local Binary Patterns      
print("\n-> LOCAL BINARY PATTERNS\n")                              
dataset_train = 'trainLBP.dat'
dataset_test = 'testLBP.dat'
ejercicio3(dataset_train,dataset_test)