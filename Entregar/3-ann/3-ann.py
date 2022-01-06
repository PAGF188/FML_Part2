"""
Practica 3: Exercises of ANN classifiers
Autor: Pablo García Fernández.

"""

import matplotlib.pyplot as plt
from ELM import Elm
from crea_folds import crea_folds
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
import warnings
warnings.filterwarnings("ignore")
import pdb

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
        pass
    # EJERCICIO 2,3 (sobre wine y hepatitis)
    else:
        #ej2(dataset_name)
        ej3(dataset_name)


def ej3(dataset_name):
    """
    EJERCICIO 3:
    Calcular acc., kappa y cm usando MLP y ELM + cross validation + sintonizacion.
    """
    #####################################################################################3
    # Ejercicio 3_1 -> MLP 
    print("EJERCICIO 3_1 (MLP cross validation):")

    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    K=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev
    
    #sintonizacion
    print("Sintonizacion:")
    H=3
    IK=30
    vkappa=np.zeros(K); kappa_mellor=-np.Inf
    neurons_mellor = []
    for i in range(1,H+1):
        print('%10s '%('H%i'%i),end='')
    print('%10s'%('Kappa(%)'))

    for h in range(1,H+1):
        for j in range(10,IK+1,10):
            neurons=[C]
            for m in range(1,h+1):
                neurons.insert(0,j)
                for k in range(K):
                    modelo = MLPClassifier(hidden_layer_sizes=neurons).fit(tx[k],ty[k])
                    z=modelo.predict(vx[k])
                    vkappa[k]=cohen_kappa_score(vy[k],z)
            kappa_med=np.mean(vkappa)
            for i in range(h):
                print('%10i'%(neurons[i]),end='')
            for i in range(h+1, H+1):
                print('%10s'%'',end='')
            print('%10.1f'%(100*kappa_med))
            if kappa_med>kappa_mellor:
                kappa_mellor=kappa_med
                neurons_mellor = neurons
    print(f"Mejor arquitectura: {neurons_mellor[: -1]}, kappa={100*kappa_mellor:.2f}%\n")

    #test
    print("Test:")
    v_accuracy = np.zeros(K)
    mc = np.zeros([C,C])
    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo = MLPClassifier(hidden_layer_sizes=neurons_mellor).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa=np.mean(vkappa); mc/=K; accuracy = np.mean(v_accuracy)
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")
    print("─────────────────────────────────────────\n\n\n")



    #####################################################################################3
    # Ejercicio 3_2 -> ELM 
    print("EJERCICIO 3_2 (ELM cross validation):")
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    K=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    #sintonizacion
    print("Sintonizacion:")
    IK = np.arange(5, 45, 5)
    vkappa=np.zeros(K); kappa_mellor=-np.Inf; 
    kappa_sintonizacion = []
    neurons_mellor = 5
    print('%10s %10s'%('IK','Kappa(%)'))

    for i in IK:
        for k in range(K):
            modelo = Elm(hidden_units=i, x=tx[k], y=ty[k])
            _ = modelo.fit()
            z=modelo.predict(vx[k])
            vkappa[k]=cohen_kappa_score(vy[k],z)

        kappa_med=np.mean(vkappa); kappa_sintonizacion.append(kappa_med)
        print('%10i %10.1f'%(i,100*kappa_med))
        if kappa_med>kappa_mellor:
            kappa_mellor = kappa_med
            neurons_mellor = i
    print(f"Mejor nº neuronas ocultas: {neurons_mellor}, kappa={100*kappa_mellor:.2f}%\n")

     # Grafico de sintonizacion:
    plt.plot(IK, kappa_sintonizacion,marker='o', label='Evolucion del kappa con V')
    plt.title(f'dataset: {dataset_name}, V_mejor= {neurons_mellor} kappa={100*kappa_mellor:.2f}%')
    plt.xticks(IK); plt.legend(); plt.grid(True)
    plt.xlabel('V'); plt.ylabel('kappa (%)')
    plt.savefig("sintonizacionIK_" + dataset_name + ".png")
    plt.show()

    #test
    print("Test:")
    v_accuracy = np.zeros(K)
    mc = np.zeros([C,C])
    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo = Elm(hidden_units=neurons_mellor, x=tx[k], y=ty[k])
        _ = modelo.fit()
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa=np.mean(vkappa); mc/=K; accuracy = np.mean(v_accuracy)
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")
    print("─────────────────────────────────────────\n\n\n")


# Ejecutamos
pr3("wine.data")
pr3("hepatitis.data")
pr3("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr3("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')
