"""
Practica 5: Exercises 1,2,3 with ensembles
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)
-Time

"""

from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.model_selection import *

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from time import perf_counter
import seaborn as sns
import pdb      



def pr5(dataset_name):
    """ Función que ejecuta los 5 ejercicios mencionados
    en la practica 5.
    """

    print("=============================================")
    print(f"   Ejecucion pr3 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # if dataset_name=='hepatitis.data':
    #     ej2(dataset_name)
    ej3(dataset_name)

def ej2(dataset_name):
    """
    Ejecutar RF y AdaBoost separando datos en train y test. Reportar todas las metricas
    """

    print("EJERCICIO 2 (RF y AdaBoost, con separacion tran/test):")

    # Leer datos
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    # Separacion train/test
    # Ponemos stratify a True para mantener poboaciones relativas!!
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=180, stratify=y)

    # Preprocesamiento
    media = np.mean(Xtrain); dv = np.std(Xtrain)
    Xtrain = (Xtrain-media)/dv
    Xtest = (Xtest-media)/dv

    # RF train/test
    t1 = perf_counter()
    model_RF = RandomForestClassifier(n_estimators=100).fit(Xtrain, ytrain)
    z_RF = model_RF.predict(Xtest)
    tiempoRF = perf_counter() - t1

    # Adaboost train/test
    t2 = perf_counter()
    model_ADA = AdaBoostClassifier(n_estimators=100).fit(Xtrain, ytrain)
    z_ADA = model_ADA.predict(Xtest)
    tiempoADA = perf_counter() - t2


    ##################################################
    #  Resultados RF:
    acc = accuracy_score(ytest, z_RF)* 100
    kappa = cohen_kappa_score(ytest, z_RF)* 100
    cf = confusion_matrix(ytest, z_RF)
    precission = precision_score(ytest, z_RF)* 100
    recall = recall_score(ytest, z_RF)* 100
    f1 = f1_score(ytest, z_RF)* 100

    print("Resultados RF: ")
    print("Time (s):", tiempoRF)
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print(f"precision.: {precission:.2f}%\nrecall: {recall:.2f}%\nf1 = {f1:.2f}%\n")   

    # ROC ----------------------------------
    aux=model_RF.predict_proba(Xtest)
    p=aux[:,1] # probability of class 1
    fpr, tpr, thresholds = roc_curve(ytest,p)
    plt.clf(); plt.plot(fpr,tpr,'bs--'); 
    plt.ylabel('True positive rate') 
    plt.xlabel('False positive rate')
    plt.title('AUC= %.4f | dataset=%s'% (roc_auc_score(ytest,p), dataset_name))
    plt.grid(True)
    plt.savefig("ej2_ROC_RF" + dataset_name + ".png"); plt.clf()

    # cm ----------------------------------
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej2_RF' + dataset_name + '.png'); plt.clf()

    print("─────────────────────────────────────────────────")

    ##################################################
    # Resultados ADABoost
    acc = accuracy_score(ytest, z_ADA) * 100
    kappa = cohen_kappa_score(ytest, z_ADA)* 100
    cf = confusion_matrix(ytest, z_ADA)
    precission = precision_score(ytest, z_ADA)* 100
    recall = recall_score(ytest, z_ADA)* 100
    f1 = f1_score(ytest, z_ADA)* 100

    print("Resultados AdaBoost: ")
    print("Time (s):", tiempoADA)
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print(f"precision.: {precission:.2f}%\nrecall: {recall:.2f}%\nf1 = {f1:.2f}%\n")   

    # ROC ----------------------------------
    aux=model_ADA.predict_proba(Xtest)
    p=aux[:,1] # probability of class 1
    fpr, tpr, thresholds = roc_curve(ytest,p)
    plt.clf(); plt.plot(fpr,tpr,'bs--'); 
    plt.ylabel('True positive rate') 
    plt.xlabel('False positive rate')
    plt.title('AUC= %.4f | dataset=%s'% (roc_auc_score(ytest,p), dataset_name))
    plt.grid(True)
    plt.savefig("ej2_ROC_ADABOOST" + dataset_name + ".png"); plt.clf()

    # cm ----------------------------------
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej2_ADA' + dataset_name + '.png'); plt.clf()

    print("─────────────────────────────────────────────────")
   

def ej3(dataset_name):
    
    print("EJERCICIO 3 (RF y AdaBoost, cross validation (and n_estimator tuning in ADABOOST)):\n")

    # Leer datos
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    ####################################################################################
    # RANDOM FOREST
    ####################################################################################
    # RF -> Cross validation usando la funcion de scikit-learn. K=4,5,10
    model_RF = RandomForestClassifier(n_estimators=100)
    ej3_cross_validation(dataset_name, model_RF, x, y, "RF")


    ####################################################################################
    # ADABOOST
    ####################################################################################
    # El ejercicio pide sintonizar el numero de clasificadores. Como solo permite usar
    # funciones de python, usamos Grid search (ademas no se uso en ninguna otra practica
    # por lo que esta bien probarlo al menos 1 vez).

    # Reseteamos los datos por si alguna de las funciones anteriores los modifican
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    C = len(np.unique(y))

    # Sintonizacion estimadores
    score = "f1"   # usamos f1 como metrica
    n_estima = list(range(40, 220, 20))
    tunedParameters = {'n_estimators': n_estima }
    model_ADA = GridSearchCV(RandomForestClassifier(random_state=0), tunedParameters, scoring='%s_macro' % score)
    model_ADA.fit(x, y)
    print("Best parameters set found on development set:")
    print(model_ADA.best_params_)
    print("Grid scores on development set:")
    means = model_ADA.cv_results_['mean_test_score']
    stds = model_ADA.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_ADA.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # Grafico de sintonizacion:
    plt.clf()
    plt.plot(n_estima, means, marker='o', label='Evolucion de f1-score con n_estimadores')
    plt.title(f'Tuning n_estimators, {dataset_name}, mejor= {model_ADA.best_params_["n_estimators"]}')
    plt.xticks(n_estima); plt.legend(); plt.grid(True)
    plt.xlabel('no. estimadores'); plt.ylabel('f1-score')
    plt.savefig("ej3_sintonizacionN_" + dataset_name + ".png")
    #plt.show(); 
    plt.clf()
    print()

    # De nuevo evaluacion con cross-validation
    ej3_cross_validation(dataset_name, model_ADA, x, y, "ADA")



def ej3_cross_validation(dataset_name, model, x, y, model_name):
    print("Resultados " + model_name + "\n")
    k_v = [4,5,10]
    accura_v = []
    kappa_v = []
    cm_v = []
    time_v = []
    for k_ in k_v:
        t0 = perf_counter()
        z=cross_val_predict(model, x, y, cv=k_)
        time_v.append(perf_counter()-t0)
        accura_v.append(accuracy_score(y,z) * 100)
        kappa_v.append(cohen_kappa_score(y, z) * 100)
        cm_v.append(confusion_matrix(y,z))
    
    for i in range(len(k_v)):
        print("k=%d, acc. = %.2f, kappa=%.2f, time=%.4f"%(k_v[i], accura_v[i], kappa_v[i], time_v[i]))
        # matrix confusion
        plt.clf()
        cf_image = sns.heatmap(cm_v[i], cmap='Blues', annot=True, fmt='g')
        figure = cf_image.get_figure()    
        figure.savefig('ej3_' + model_name + '_k='+ str(k_v[i]) + "_" + dataset_name + '.png'); 

    print("─────────────────────────────────────────────────")



# Ejecutamos
pr5("wine.data")
pr5("hepatitis.data")
