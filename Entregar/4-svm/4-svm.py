"""
Practica 3: Exercises of SVM classifier
Autor: Pablo García Fernández.

"""

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from sklearn.metrics import *
from sklearn.svm import *
import pdb

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
        ej3(dataset_name)
        ej4(dataset_name)

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
    print("─────────────────────────────────────────────────")

def ej3(dataset_name):
    """
    EJERCICIO 3:
    Calcular acc., kappa y cm usando SVM (gaussian kernel) sobre todo el dataset (hepatitis, wine).
    Default params: reg=100, sigma = 1/n
    """

    print("EJERCICIO 3 (SVM gaussian kernel sobre dataset completo train/test):")
    datos = np.loadtxt('../data/' + dataset_name)

    # Preprocesamiento
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    x = (x-np.mean(x,0))/np.std(x,0)
    C = len(np.unique(y))

    # Entrenamiento
    modelo=SVC(C=100, kernel ='rbf', gamma= 1/x.shape[1], verbose=False).fit(x,y)

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
    print("─────────────────────────────────────────────────")

def ej4(dataset_name):
    """
    Repetir con cross validation (k=4, k=10)
    """

    # PARTE 1: CROSS VALIDATION LINEAR KERNEL (K=4)
    print("EJERCICIO 4_1 (SVM cross_validation, linear kernel, K=4):")
    ej4_linear_kernel_cross_validation(dataset_name, K=4)

    # PARTE 2: CROSS VALIDATION LINEAR KERNEL (K=10)
    print("EJERCICIO 4_2 (SVM cross_validation, linear kernel, K=10):")
    ej4_linear_kernel_cross_validation(dataset_name, K=10)

    # PARTE 3: CROSS VALIDATION LINEAR KERNEL (K=10)
    print("EJERCICIO 4_3 (SVM cross_validation, gaussian kernel, K=4):")
    ej4_gaussian_kernel_cross_validation(dataset_name, K=4)

    # PARTE 4: CROSS VALIDATION GAUSSIAN KERNEL (K=10)
    print("EJERCICIO 4_4 (SVM cross_validation, gaussian kernel, K=10):")
    ej4_gaussian_kernel_cross_validation(dataset_name, K=10)

def ej4_linear_kernel_cross_validation(dataset_name, K):
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    # Sintonizacion lambda
    print("Sintonizacion:")
    vL=2.**np.arange(-5, 16, 2)
    vkappa=np.zeros(K); kappa_mellor=-np.Inf; L_mellor=vL[0]
    kappa_sintonizacion = []
    print('%10s %10s'%('L','Kappa(%)'))

    for L in vL:
        for k in range(K):
            modelo = SVC(C=L, kernel='linear', verbose=False).fit(tx[k],ty[k])
            z = modelo.predict(vx[k])
            vkappa[k]=cohen_kappa_score(vy[k],z)
        kappa_med=np.mean(vkappa)
        kappa_sintonizacion.append(kappa_med)
        print('%10.2f %10.1f'%(L,100*kappa_med))
        if kappa_med>kappa_mellor:
            kappa_mellor=kappa_med; L_mellor=L

    print('V_mejor=%i kappa=%.2f%%\n'%(L_mellor,100*kappa_mellor))

    # Grafico de sintonizacion (escala logaritmica): 
    plt.plot(vL, kappa_sintonizacion, marker='o', label='Evolucion del kappa con L')
    plt.title(f'dataset: {dataset_name}, L_mejor= {L_mellor} kappa={100*kappa_mellor:.2f}%')
    plt.legend(); plt.grid(True)
    plt.xlabel('V'); plt.ylabel('kappa (%)')
    plt.xscale("log")
    plt.savefig("sintonizacionL_" + dataset_name + "_K="+str(K)+".png")
    plt.show()

    # TEST
    print("Test:")
    C = len(np.unique(y))
    mc = np.zeros([C,C])
    v_accuracy = np.zeros(K)

    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo = SVC(C=L_mellor, kernel='linear', verbose=False).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")
    print("─────────────────────────────────────────────────")

def ej4_gaussian_kernel_cross_validation(dataset_name, K):
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

    #preprocesamiento
    for k in range(K):
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        vx[k]=(vx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    # Sintonizacion lambda y sigma
    print("Sintonizacion:")
    vL=2.**np.arange(-5, 16, 2)
    vG=2.**np.arange(-7, 8, 2)

    kappa_sintonizacion=np.zeros((len(vL), len(vG))); 
    vkappa=np.zeros(K);kappa_mellor=-np.Inf; L_mellor=vL[0]; G_mellor=vG[0]
    print('%10s %15s %10s %10s'%('Lambda','Gamma', 'Kappa (%)', 'Mejor'))

    for i,L in enumerate(vL):
        for j,G in enumerate(vG):
            for k in range(K):
                modelo=SVC(C=L, kernel ='rbf', gamma=G, verbose=False).fit(tx[k],ty[k])
                z = modelo.predict(vx[k])
                vkappa[k]=cohen_kappa_score(vy[k],z) * 100
            kappa_med=np.mean(vkappa)
            kappa_sintonizacion[i,j] = kappa_med
            if kappa_med>kappa_mellor:
                kappa_mellor=kappa_med; L_mellor=L; G_mellor = G
            print('%.2f %15g %10.1f %10.1f'%(L,G, kappa_med, kappa_mellor))
           

    print('L_mejor=%g, G_mejor=%g, kappa=%.2f%%\n'%(L_mellor, G_mellor, kappa_mellor))

    # Grafico de sintonizacion:
    u=np.ravel(kappa_sintonizacion); plt.plot(u); plt.grid(True)
    plt.title(f'dataset: {dataset_name}, L_mejor={L_mellor}, G_mejor={G_mellor} kappa={kappa_mellor:.2f}%')
    plt.axis([1,len(u), -5, 100])
    plt.xlabel('Configuracion'); plt.ylabel('kappa (%)')
    plt.savefig("sintonizacion_SVC_gaussian_" + dataset_name + "_K="+str(K)+".png")
    plt.show()

    # TEST
    print("Test:")
    C = len(np.unique(y))
    mc = np.zeros([C,C])
    v_accuracy = np.zeros(K)

    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo=SVC(C=L_mellor, kernel ='rbf', gamma=G_mellor, verbose=False).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")
    print("─────────────────────────────────────────────────")



# Ejecutamos
pr4("wine.data")
pr4("hepatitis.data")
pr4("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr4("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')


