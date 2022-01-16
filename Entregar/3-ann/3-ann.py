"""
Practica 3: Exercises of ANN classifiers
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)

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
import seaborn as sns
import matplotlib.pyplot as plt

import pdb

def pr3(dataset_name):
    """ Función que ejecuta los 4 ejercicios mencionados
    en la practica 2.
    - ej2: Calcular acc., kappa y cm usando MLP y ELM sobre todo el dataset (hepatitis, wine).
    - ej3: Repetir con cross-validation k=4
    - ej4: Esta en el archivo ej4.py
    """

    print("=============================================")
    print(f"   Ejecucion pr3 sobre {dataset_name} dataset!")
    print("=============================================\n")

    ej2(dataset_name)
    ej3(dataset_name)


def ej2(dataset_name):
    """
    EJERCICIO 2:
    Calcular acc., kappa y cm usando MLP y ELM. Usando todo el dataset
    y probando diferentes valores de neuronas.
    """

    #####################################################################################3
    # Ejercicio 2_1 -> MLP 
    print("EJERCICIO 2_1 (MLP whole dataset diferentes neurons):") 

    datos = np.loadtxt('../data/' + dataset_name)

    # Preprocesamiento
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    x = (x-np.mean(x,0))/np.std(x,0)
    C = len(np.unique(y))

    # Probamos diferentes valores de capas y neuronas. Para cada configuracion imprimimos kappa
    H=3
    IK=30
    kappa_mellor=-np.Inf
    v_kappa = []; v_acc = []; v_cf = []
    if C==2:
        v_prec = []; v_rec = []; v_f1 = []
    for i in range(1,H+1):
        print('%10s '%('H%i'%i),end='')
    print('%10s'%('Kappa(%)'))
    indice_mejor = 0
    indice = 0
    for h in range(1,H+1):
        for j in range(10,IK+1,10):
            neurons=[C]
            for m in range(1,h+1):
                neurons.insert(0,j)
            modelo=MLPClassifier(hidden_layer_sizes=neurons).fit(x,y)
            z=modelo.predict(x)
            kappa = cohen_kappa_score(y,z)*100; acc = 100 * accuracy_score(y,z); cf=confusion_matrix(y,z)
            v_kappa.append(kappa); v_acc.append(acc); v_cf.append(cf)

            if C==2:
                prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z); f1 = 100 * f1_score(y,z)
                v_prec.append(prec); v_rec.append(rec); v_f1.append(f1)
            for i in range(h):
                print('%10i '%(neurons[i]),end='')			
            for i in range(h+1,H+1):
                print('%10s '%'',end='')
            print('%10.1f'%(kappa))
            if kappa>kappa_mellor:
                kappa_mellor=kappa; indice_mejor=indice
            indice +=1

    # Para la configuración que mejor funciona -> pintamos accuracy y confusion matrix (si lo hacemos para todas el reporte sería muy grande)
    print(f"acc.: {v_acc[indice_mejor]:.2f}%\nkappa: {v_kappa[indice_mejor]:.2f}%\ncf = \n{v_cf[indice_mejor]}\n")

    cf_image = sns.heatmap(v_cf[indice_mejor], cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('MLP_ej2_' + dataset_name + '.png'); plt.clf()

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        print(f"precision.: {v_prec[indice_mejor]:.2f}%\nrecall: {v_rec[indice_mejor]:.2f}%\nf1 = {v_f1[indice_mejor]:.2f}%\n")            
    print("─────────────────────────────────────────")


    #####################################################################################3
    # Ejercicio 2_2 -> ELM 
    print("EJERCICIO 2_2 (ELM whole dataset diferentes neurons):")

    datos = np.loadtxt('../data/' + dataset_name)

    # Preprocesamiento
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    x = (x-np.mean(x,0))/np.std(x,0)
    C = len(np.unique(y))

    # Probamos diferentes valores de neuronas. Para cada configuracion imprimimos kappa
    IK = np.arange(5, 45, 5)
    kappa_mellor=-np.Inf
    v_kappa = []; v_acc = []; v_cf = []
    if C==2:
        v_prec = []; v_rec = []; v_f1 = []
    print('%10s %10s'%('IK','Kappa(%)'))
    indice_mejor = 0
    indice = 0
    for i in IK:
        modelo = Elm(hidden_units=i, x=x, y=y)
        _ = modelo.fit()
        z=modelo.predict(x)
        kappa = cohen_kappa_score(y,z)*100; acc = 100 * accuracy_score(y,z); cf=confusion_matrix(y,z)
        v_kappa.append(kappa); v_acc.append(acc); v_cf.append(cf)

        if C==2:
            prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z); f1 = 100 * f1_score(y,z)
            v_prec.append(prec); v_rec.append(rec); v_f1.append(f1)

        print('%10i %10.1f'%(i,kappa))
        if kappa>kappa_mellor:
            kappa_mellor = kappa; indice_mejor=indice
        indice +=1

    # Para la configuración que mejor funciona -> pintamos accuracy y confusion matrix (si lo hacemos para todas el reporte sería muy grande)
    print(f"acc.: {v_acc[indice_mejor]:.2f}%\nkappa: {v_kappa[indice_mejor]:.2f}%\ncf = \n{v_cf[indice_mejor]}\n")

    cf_image = sns.heatmap(v_cf[indice_mejor], cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ELM_ej2_' + dataset_name + '.png'); plt.clf()

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        print(f"precision.: {v_prec[indice_mejor]:.2f}%\nrecall: {v_rec[indice_mejor]:.2f}%\nf1 = {v_f1[indice_mejor]:.2f}%\n")            
    print("─────────────────────────────────────────")


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
    neurons_mellor = None
    for i in range(1,H+1):
        print('%10s '%('H%i'%i),end='')
    print('%10s'%('Kappa(%)'))

    for h in range(1,H+1):
        for j in range(10,IK+1,10):
            neurons=[C]
            for m in range(1,h+1):
                neurons.insert(0,j)
            for k in range(K):
                modelo=MLPClassifier(hidden_layer_sizes=neurons).fit(tx[k],ty[k])
                z=modelo.predict(vx[k])
                vkappa[k]=cohen_kappa_score(vy[k],z)
            kappa_med=np.mean(vkappa)
            for i in range(h):
                print('%10i '%(neurons[i]),end='')			
            for i in range(h+1,H+1):
                print('%10s '%'',end='')
            print('%10.1f'%(100*kappa_med))
            if kappa_med>kappa_mellor:
                kappa_mellor=kappa_med;neurons_mellor=neurons

    print(f"Mejor arquitectura: {neurons_mellor[: -1]}, kappa={100*kappa_mellor:.2f}%\n")

    #test
    print("Test:")  
    if C==2:
        pre=np.zeros(K);re=np.zeros(K);f1=np.zeros(K)
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
        if C==2:
            pre[k]=precision_score(y,z) * 100
            re[k]=recall_score(y,z) * 100
            f1[k]=f1_score(y,z) * 100

    kappa=np.mean(vkappa); mc/=K; accuracy = np.mean(v_accuracy)
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")
    if C==2:
        prec =np.mean(pre); rec = np.mean(re)
        f1 = np.mean(f1)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n")  
        
    cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('MLP_ej3_' + dataset_name + '.png'); plt.clf()

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
    plt.plot(IK, kappa_sintonizacion,marker='o', label='Evolucion del kappa con IK')
    plt.title(f'dataset: {dataset_name}, IK_mejor= {neurons_mellor} kappa={100*kappa_mellor:.2f}%')
    plt.xticks(IK); plt.legend(); plt.grid(True)
    plt.xlabel('V'); plt.ylabel('kappa (%)')
    plt.savefig("sintonizacionIK_" + dataset_name + ".png")
    plt.show()

    #test
    if C==2:
        pre=np.zeros(K);re=np.zeros(K);f1=np.zeros(K)
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

        if C==2:
            pre[k]=precision_score(y,z) * 100
            re[k]=recall_score(y,z) * 100
            f1[k]=f1_score(y,z) * 100

    kappa=np.mean(vkappa); mc/=K; accuracy = np.mean(v_accuracy)
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

    if C==2:
        prec =np.mean(pre); rec = np.mean(re)
        f1 = np.mean(f1)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n")    

    cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ELM_ej3_' + dataset_name + '.png'); plt.clf()

    print("─────────────────────────────────────────\n\n\n")


# Ejecutamos
pr3("wine.data")
pr3("hepatitis.data")
pr3("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr3("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')
