"""
Practica 4: Exercises of SVM classifier
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)
-Time (measure time)

"""

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from sklearn.metrics import *
from sklearn.svm import *
from time import perf_counter
import seaborn as sns
import pdb

def pr4(dataset_name):
    """ Función que ejecuta los 5 ejercicios mencionados
    en la practica 4.
    - ej2: Calcular acc., kappa y cm usando SVM (linear kernel) sobre todo el dataset (hepatitis, wine).
    - ej3: Repetir con kernel gaussiano (L=100, sigma=1/n)
    - ej4: Repetir con cross validation (k=4, k=10)
    - ej5: 
    """

    print("=============================================")
    print(f"   Ejecucion pr3 sobre {dataset_name} dataset!")
    print("=============================================\n")

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

    t_inicio = perf_counter()
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
    tiempo_total = perf_counter() - t_inicio   
    print("Tiempo total: %.4f"%(tiempo_total))

    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('Linearl_ej2_' + dataset_name + '.png'); plt.clf()

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
    t_inicio = perf_counter()
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
    tiempo_total =  perf_counter() - t_inicio   
    print("Tiempo total: %.4f"%(tiempo_total))    

    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('Gaussian_ej3_' + dataset_name + '.png'); plt.clf()

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
    plt.savefig("sintonizacionL_" + dataset_name + "_K="+str(K)+".png"); plt.clf()
    #plt.show()

    # TEST
    t_inicio = perf_counter()
    print("Test:")
    C = len(np.unique(y))
    mc = np.zeros([C,C])
    v_accuracy = np.zeros(K)
    if C==2:
        v_prec = []; v_rec = []; v_f1 = []
    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo = SVC(C=L_mellor, kernel='linear', verbose=False).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

        if C==2:
            prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z); f1 = 100 * f1_score(y,z)
            v_prec.append(prec); v_rec.append(rec); v_f1.append(f1)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec =np.mean(v_prec); rec = np.mean(v_rec); f1 = np.mean(v_f1)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n") 

    tiempo_test = perf_counter() - t_inicio   
    print("Tiempo test: %.4f"%(tiempo_test))

    cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej4_' + str(K) + '_folds_Linear_' + dataset_name + '.png'); plt.clf()
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
    if C==2:
        v_prec = []; v_rec = []; v_f1 = []

    for k in range(K):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))
        modelo=SVC(C=L_mellor, kernel ='rbf', gamma=G_mellor, verbose=False).fit(tx[k],ty[k])
        z=modelo.predict(sx[k]); y=sy[k]
        vkappa[k]=100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

        if C==2:
            prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z); f1 = 100 * f1_score(y,z)
            v_prec.append(prec); v_rec.append(rec); v_f1.append(f1)

    kappa = np.mean(vkappa)
    accuracy = np.mean(v_accuracy)
    mc/=K
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec =np.mean(v_prec); rec = np.mean(v_rec); f1 = np.mean(v_f1)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n") 

    tiempo_test = perf_counter() - t_inicio   
    print("Tiempo test: %.4f"%(tiempo_test))

    cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej4_' + str(K) + '_folds_Gaussian_' + dataset_name + '.png'); plt.clf()

    print("─────────────────────────────────────────────────")



# Ejecutamos
pr4("wine.data")
pr4("hepatitis.data")


