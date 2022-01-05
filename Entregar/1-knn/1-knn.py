"""
Practica 1: Model selection and evaluation
Autor: Pablo García Fernández.

"""

import numpy as np
from sklearn.neighbors import *
from sklearn.metrics import *
from crea_folds import crea_folds
import matplotlib.pyplot as plt

def pr1(dataset_name, dataset_train, dataset_test):
    """ Función que ejecuta los 4 ejercicios mencionados
    en la practica 1.
    - ej1: Ejecutar 1-NN y reportar el accuraccy.
    - ej2: Usar otras métricas: kappa y confusion matrix.
    - ej3: Cross-validation usando 4 folds (k=4)
    - ej4: K-NN con sintonización de k (validation set)

    Los dataset a emplear son 2:
    - Coocur Mat. -> trainCoocur.dat' y 'testCoocur.dat'
    - LBP -> 'trainLBP.dat' y 'testLBP.dat'
    """

    print("=============================================")
    print(f"   Ejecucion pr1 sobre {dataset_name} dataset!")
    print("=============================================\n")

    ej1_2(dataset_name, dataset_train, dataset_test)
    ej3(dataset_name, dataset_train, dataset_test)
    ej4(dataset_name, dataset_train, dataset_test)

def ej1_2(dataset_name, dataset_train, dataset_test):
    """
    EJERCICIO 1 Y 2
    - Ejecutar 1-NN y reportar el accuraccy.
    - Usar otras métricas: kappa y confusion matrix.
    """
    # Carga de datos
    data_train = np.loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = np.loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # EJERCICIO 1 -------------------------------------------------
    # ejecutar 1-NN y reportar accuracy

    # Preprocesamiento
    med = np.mean(x_train,0); dev = np.std(x_train,0)
    x_train = (x_train-med)/dev
    x_test = (x_test-med)/dev

    modelo=KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    z = modelo.predict(x_test)
    acc = 100 * accuracy_score(y_test, z)
    print(f"EJERCICIO 1 (1-KNN):\nacc: {acc:.2f}%\n")
    print("─────────────────────────────────────────")

    # EJERCICIO 2 ---------------------------------------------------------------   
    # otra metricas
    kappa = 100 * cohen_kappa_score(y_test,z)
    cf=confusion_matrix(y_test,z)

    print("EJERCICIO 2 (otras metricas):")
    print(f"kappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print("─────────────────────────────────────────")

def ej3(dataset_name, dataset_train, dataset_test):
    """
    EJERCICIO 3
    Cross-validation usando 4 folds (k=4)
    """

    print("EJERCICIO 3 (1-NN + cross validation k=4):")

    # Carga de datos
    data_train = np.loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = np.loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # Juntamos train/test sets para cross-validation
    x = np.concatenate([x_train,x_test],axis=0)
    y = np.concatenate([y_train,y_test])

    # Generamos K folds
    K_ = 4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K_)

    # Juntamos train y validación (en este ejercicio no hay sintonización de params.)
    for k in range(K_):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))

        # Preprocesamos
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    # Ejecutamos 1-NN sobre los 4 folds. Tomamos las medias de las metricas
    C = len(np.unique(y))

    mc = np.zeros([C,C]); v_kappa=np.zeros(K_); v_accuracy = np.zeros(K_)

    for k in range(K_):
        modelo = KNeighborsClassifier(n_neighbors=1).fit(tx[k],ty[k])
        z = modelo.predict(sx[k]); y=sy[k]
        v_kappa[k] = 100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = np.mean(v_kappa); accuracy = np.mean(v_accuracy); mc/=K_
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}\n")
    print("─────────────────────────────────────────")

def ej4(dataset_name, dataset_train, dataset_test):
    """
    EJERCICIO 4
    K-NN con sintonización de k (validation set)
    """

    print("EJERCICIO 4 (K-nn con sintonización de K)")

    # Carga de datos
    data_train = np.loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = np.loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # Juntamos train/test sets para cross-validation
    x = np.concatenate([x_train,x_test],axis=0)
    y = np.concatenate([y_train,y_test])
    
    # k-NN con ajuste de k.
    K = 4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    #sintonizacion
    print("Sintonizacion:")
    V = np.arange(1, 31, 2)
    vkappa=np.zeros(K); kappa_mellor=-np.Inf; V_mellor=V[0]
    kappa_sintonizacion = []
    print('%10s %10s'%('V','Kappa(%)'))

    for v in V:
        for k in range(K):
            modelo = KNeighborsClassifier(n_neighbors=v).fit(tx[k],ty[k])
            z = modelo.predict(vx[k])
            vkappa[k]=cohen_kappa_score(vy[k],z)
        kappa_med=np.mean(vkappa)
        kappa_sintonizacion.append(kappa_med)
        print('%10i %10.1f'%(v,100*kappa_med))
        if kappa_med>kappa_mellor:
            kappa_mellor=kappa_med; V_mellor=v

    print('V_mejor=%i kappa=%.2f%%\n'%(V_mellor,100*kappa_mellor))

    # Grafico de sintonizacion:
    plt.plot(V, kappa_sintonizacion,marker='o', label='Evolucion del kappa con V')
    plt.title(f'dataset: {dataset_name}, V_mejor= {V_mellor} kappa={100*kappa_mellor:.2f}%')
    plt.xticks(V); plt.legend(); plt.grid(True)
    plt.xlabel('V'); plt.ylabel('kappa (%)')
    plt.savefig("sintonizacionV_" + dataset_name)
    plt.show()
   
    #test
    print("Test:")
    C = len(np.unique(y))
    mc = np.zeros([C,C])
    v_accuracy = np.zeros(K)

    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo = KNeighborsClassifier(n_neighbors=V_mellor).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}\n\n\n")


# Ejecutamos
pr1("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr1("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')



