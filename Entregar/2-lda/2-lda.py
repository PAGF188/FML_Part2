"""
Practica 2: Exercises of LDA classifier
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)
"""

import numpy as np
from sklearn.neighbors import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from crea_folds import crea_folds

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

def pr2(dataset_name, dataset_train=None, dataset_test=None):
    """ Función que ejecuta los 4 ejercicios mencionados
    en la practica 2.
    - ej2: Calcular acc., kappa y cm usando LDA y todo el dataset (hepatitis, wine).
    - ej3: Repetir con cross-validation k=4
    - ej4: Repetir con LOOCV
    - ej5: Usar LDA en LBP y CooCu datasets

    dataset_name se emplea para hepatitis y wine datasets (donde no hay 
    separacion entre train y test). dataset_train y dataset_test se emplea
    para LBP y Coocur (donde train y test ya están separados)
    """

    print("=============================================")
    print(f"   Ejecucion pr2 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # EJERCICIO 5 -----------------------------------------------------------------
    # LDA básico sobre LBP y CooCU
    if dataset_name == 'LBP' or dataset_name == 'Coocur':
        ej5(dataset_name, dataset_train, dataset_test)

    # EJERCICIO 2,3,4 (sobre wine y hepatitis)
    else:
        ej2(dataset_name)
        ej3(dataset_name)
        ej4(dataset_name)


def ej2(dataset_name):
    """
    EJERCICIO 2:
    Calcular acc., kappa y cm usando LDA y todo el dataset (hepatitis, wine).
    """

    print("EJERCICIO 2 (LDA dataset completo train/test):")
    datos = np.loadtxt('../data/' + dataset_name)

    # Preprocesamiento
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    x = (x-np.mean(x,0))/np.std(x,0)
    C = len(np.unique(y))

    # Entrenamiento
    modelo = LinearDiscriminantAnalysis().fit(x,y)

    # Test
    z = modelo.predict(x)
    kappa = 100 * cohen_kappa_score(y,z); acc = 100 * accuracy_score(y,z)
    cf=confusion_matrix(y,z)
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")

    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('LDA_ej2_' + dataset_name + '.png'); plt.clf()

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec = 100 * precision_score(y,z); rec = 100 * recall_score(y,z)
        f1 = 100 * f1_score(y,z)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n")            
    print("─────────────────────────────────────────")


def ej3(dataset_name):
    """
    EJERCICIO 3:
    Calcular acc., kappa y cm usando LDA y cross validation (k=4)
    sobre hepatitis y wine.
    """
    print("EJERCICIO 3 (LDA cross validation k=4):")

    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    K_=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K_)

    # Juntamos train y validación (en este ejercicio no hay sintonización de params.)
    for k in range(K_):
        tx[k]=np.vstack((tx[k],vx[k]))
        ty[k]=np.concatenate((ty[k],vy[k]))

        # preprocesamos
        med=np.mean(tx[k],0); dev=np.std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    mc = np.zeros([C,C])
    v_kappa=np.zeros(K_)
    v_accuracy = np.zeros(K_)
    prec_v = np.zeros(K_); rec_v = np.zeros(K_); f1_v = np.zeros(K_)
    for k in range(K_):
        modelo = LinearDiscriminantAnalysis().fit(tx[k],ty[k])
        z = modelo.predict(sx[k]); y=sy[k]
        v_kappa[k] = 100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)
        # Si es un problema de clasificación binaria reportamos más métricas.
        if C==2:
            prec_v[k] = 100 * precision_score(y,z)
            rec_v[k] = 100 * recall_score(y,z)
            f1_v[k] = 100 * f1_score(y,z)

    kappa = np.mean(v_kappa); accuracy = np.mean(v_accuracy); mc/=K_
    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}\n")

    cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('LDA_ej3_' + dataset_name + '.png'); plt.clf()

    # Si es un problema de clasificación binaria reportamos más métricas.
    if C==2:
        prec =np.mean(prec_v); rec = np.mean(rec_v)
        f1 = np.mean(f1_v)
        print(f"precision.: {prec:.2f}%\nrecall: {rec:.2f}%\nf1 = {f1:.2f}%\n")            
    print("─────────────────────────────────────────")

def ej4(dataset_name):
    """
    EJERCICIO 4:
    Calcular acc. usando LOOCV sobre hepatitis y wine.
    """

    print("EJERCICIO 4 (LDA LOOCV):")

    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    K_=len(y)
    errores = 0

    # K_ iteraciones. En cada una entrenamos con todos los patrones menos con el i-esimo.
    for i in range(K_):
        patron_test_x = np.reshape(x[i,:],(1,-1))
        patron_test_y = np.array([y[i]])
        x_nuevo = np.delete(x,i,0)
        y_nuevo = np.delete(y,i,0)

        # Preprocesamos
        med=np.mean(x_nuevo,0); dev=np.std(x_nuevo,0)
        x_nuevo=(x_nuevo-med)/dev
        patron_test_x=(patron_test_x-med)/dev

        # Entrenamos/testeamos
        modelo = LinearDiscriminantAnalysis().fit(x_nuevo, y_nuevo)
        z = modelo.predict(patron_test_x); 
        errores += abs(z-patron_test_y)

    accuracy = (K_ - errores[0]) / K_* 100
    print(f"acc. (a mano): {accuracy:.2f}%")            

    # CON LA AYUDA DE SKLEARN (da el mismo resultado)
    scaler = StandardScaler()
    loo = LeaveOneOut()
    modelo = LinearDiscriminantAnalysis()
    metrica = 'accuracy'
    pipeline = Pipeline([('transformer', scaler), ('estimator', modelo)])
    scores = cross_val_score(pipeline, x, y, scoring=metrica, cv=loo, n_jobs=-1)
    acc = np.mean(scores*100)
    print(f"acc. (con sklearn): {acc:.2f}%")  # mismo resultado
    print("─────────────────────────────────────────\n\n\n")

def ej5(dataset_name, dataset_train, dataset_test):
    """
    EJERCICIO 5:
    Usar LDA con LBP y CooCu datasets. Sin cross-validation 
    (particiones de train/test ya están establecidas).
    """
    print(f"EJERCICIO 5 (LDA sobre {dataset_name} dataset)")
    print("─────────────────────────────────────────")

    # Cargar datos
    data_train = np.loadtxt(dataset_train)
    y_train = data_train[:,-1]
    x_train = data_train[:,0:-1]

    data_test = np.loadtxt(dataset_test)
    y_test = data_test[:,-1]
    x_test = data_test[:,0:-1]

    # Preprocesamiento
    med = np.mean(x_train,0); dev = np.std(x_train,0)
    x_train = (x_train-med)/dev
    x_test = (x_test-med)/dev

    # Entrenamiento
    modelo = LinearDiscriminantAnalysis().fit(x_train,y_train)

    # Test
    z = modelo.predict(x_test)
    kappa = 100 * cohen_kappa_score(y_test,z)
    acc = 100 * accuracy_score(y_test,z)
    cf=confusion_matrix(y_test,z)

    cf_image = sns.heatmap(cf, cmap='Blues')
    figure = cf_image.get_figure()    
    figure.savefig('LDA_ej5_' + dataset_name + '.png'); plt.clf()

    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print("─────────────────────────────────────────\n\n\n")
        
# Ejecutamos
pr2("wine.data")
pr2("hepatitis.data")
pr2("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
pr2("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')
