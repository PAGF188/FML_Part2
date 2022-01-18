"""
Practica 4: Exercises of SVM classifiers
Autor: Pablo García Fernández.

Es igual a la funcion ej4() de 4-svm.py, pero quitando las metricas para C==2, 
concatenando train/test splits ya establecidos en LBP y CooCur, midiendo tiempos y
probando OVA/OVO.
"""

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from sklearn.metrics import *
from sklearn.svm import *
from time import perf_counter
import seaborn as sns
import pdb

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

def linear_kernel_cross_validation(dataset_name, dataset_train, dataset_test):

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
    C = len(np.unique(y))

    K = 4
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
    plt.savefig("sintonizacionL_" + dataset_name + "_K="+str(K)+".png"); plt.clf()
    #plt.show()

    # TEST
    t_inicio = perf_counter()
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

    tiempo_test = perf_counter() - t_inicio   
    print("Tiempo test: %.4f"%(tiempo_test))

    cf_image = sns.heatmap(mc, cmap='Blues')
    figure = cf_image.get_figure()    
    figure.savefig('ej5_' + str(K) + '_folds_Linear_' + dataset_name + '.png'); plt.clf()
    print("─────────────────────────────────────────────────")


def gaussian_kernel_cross_validation(dataset_name, dataset_train, dataset_test, approach):
    
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
    C = len(np.unique(y))

    K = 4
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
                modelo=approach(SVC(C=L, kernel ='rbf', gamma=G, verbose=False), n_jobs=4).fit(tx[k],ty[k])
                z = modelo.predict(vx[k])
                vkappa[k]=cohen_kappa_score(vy[k],z) * 100
            kappa_med=np.mean(vkappa)
            kappa_sintonizacion[i,j] = kappa_med
            if kappa_med>kappa_mellor:
                kappa_mellor=kappa_med; L_mellor=L; G_mellor = G
            print('%.2f %15g %10.1f %10.1f'%(L,G, kappa_med, kappa_mellor))
           

    print('L_mejor=%g, G_mejor=%g, kappa=%.2f%%\n'%(L_mellor, G_mellor, kappa_mellor))

    # Grafico de sintonizacion:
    # u=np.ravel(kappa_sintonizacion); plt.plot(u); plt.grid(True)
    # plt.title(f'dataset: {dataset_name}, L_mejor={L_mellor}, G_mejor={G_mellor} kappa={kappa_mellor:.2f}%')
    # plt.axis([1,len(u), -5, 100])
    # plt.xlabel('Configuracion'); plt.ylabel('kappa (%)')
    # plt.savefig("sintonizacion_SVC_gaussian_" + dataset_name + "_K="+str(K)+".png")
    # plt.show()

    plt.imshow(kappa_sintonizacion);plt.colorbar()
    plt.xlabel('Regularizacion  ($log2\lambda$)');plt.ylabel('Ancho kernel gaussiano ($log2\gamma$)')
    plt.title(f'dataset: {dataset_name}, L_mejor= {L_mellor}, G_mejor: {G_mellor}, kappa={kappa_mellor:.2f}%')
    #plt.show()
    plt.savefig("sintonizacion_SVC_gaussian_" + dataset_name + "_K="+str(K)+".png"); plt.clf()


    # TEST
    t_inicio = perf_counter()
    print("Test:")
    C = len(np.unique(y))
    mc = np.zeros([C,C])
    v_accuracy = np.zeros(K)

    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo=approach(SVC(C=L_mellor, kernel ='rbf', gamma=G_mellor, verbose=False), n_jobs=4).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

    tiempo_test = perf_counter() - t_inicio   
    print("Tiempo test: %.4f"%(tiempo_test))

    cf_image = sns.heatmap(mc, cmap='Blues')
    figure = cf_image.get_figure()    
    #figure.savefig('ej5_' + str(K) + '_folds_Gaussian_'+ approach+ "_"+ dataset_name + '.png'); plt.clf()

    print("─────────────────────────────────────────────────")


def ej5(dataset_name, dataset_train, dataset_test):
    """
    EJERCICIO 5:
    """

    print("=============================================")
    print(f"   Ejecucion pr4 sobre {dataset_name} dataset!")
    print("=============================================\n")


    #####################################################################################3

    # PARTE 1: CROSS VALIDATION LINEAR KERNEL (K=4)
    print("EJERCICIO 5_1 (SVM cross_validation, linear kernel, K=4):")
    linear_kernel_cross_validation(dataset_name, dataset_train, dataset_test)

    # PARTE 2: CROSS VALIDATION GAUSSIAN KERNEL OVO (K=4)
    print("EJERCICIO 5_2 (SVM cross_validation, gaussian kernel, OVO, K=4):")
    gaussian_kernel_cross_validation(dataset_name, dataset_train, dataset_test, OneVsOneClassifier)

    # PARTE 3: CROSS VALIDATION GAUSSIAN KERNEL OVA (K=4)
    print("EJERCICIO 5_3 (SVM cross_validation, gaussian kernel, OVA, K=4):")
    gaussian_kernel_cross_validation(dataset_name, dataset_train, dataset_test, OneVsRestClassifier)



ej5("Coocur", '../../data/trainCoocur.dat', '../../data/testCoocur.dat')
ej5("LBP", '../../data/trainLBP.dat', '../../data/testLBP.dat')

    
