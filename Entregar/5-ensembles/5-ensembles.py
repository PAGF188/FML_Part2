"""
Practica 5: Exercises with ensembles
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)
-time

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

    if dataset_name=='hepatitis.data':
        ej2(dataset_name)

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
    # Ponemos stratify a True para mantener poboaciones relativas
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=180, stratify=y)

    # Preprocesamiento
    media = np.mean(Xtrain); dv = np.std(Xtrain)
    Xtrain = (Xtrain-media)/dv
    Xtest = (Xtest-media)/dv

    # Creamos RF y AdaBoost y entrenamos.
    model_RF = RandomForestClassifier().fit(Xtrain, ytrain)
    model_ADA = AdaBoostClassifier().fit(Xtrain, ytrain)

    # Test ambos modelos
    z_RF = model_RF.predict(Xtest)
    z_ADA = model_ADA.predict(Xtest)

    # Resultados RF:
    acc = accuracy_score(ytest, z_RF)* 100
    kappa = cohen_kappa_score(ytest, z_RF)* 100
    cf = confusion_matrix(ytest, z_RF)
    precission = precision_score(ytest, z_RF)* 100
    recall = recall_score(ytest, z_RF)* 100
    f1 = f1_score(ytest, z_RF)* 100
    roc = roc_curve(ytest, z_RF)
    auc = roc_auc_score(ytest, z_RF)* 100

    print("Resultados RF: ")
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print(f"precision.: {precission:.2f}%\nrecall: {recall:.2f}%\nf1 = {f1:.2f}%\n")   
    print("Roc: ", roc)
    print("AUC: ", auc,  "\n")
    print("─────────────────────────────────────────────────")

    # Resultados ADABoost
    acc = accuracy_score(ytest, z_ADA) * 100
    kappa = cohen_kappa_score(ytest, z_ADA)* 100
    cf = confusion_matrix(ytest, z_ADA)
    precission = precision_score(ytest, z_ADA)* 100
    recall = recall_score(ytest, z_ADA)* 100
    f1 = f1_score(ytest, z_ADA)* 100
    roc = roc_curve(ytest, z_ADA)
    auc = roc_auc_score(ytest, z_ADA)* 100

    print("Resultados AdaBoost: ")
    print(f"acc.: {acc:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{cf}\n")
    print(f"precision.: {precission:.2f}%\nrecall: {recall:.2f}%\nf1 = {f1:.2f}%\n")   
    print("Roc: ", roc)
    print("AUC: ", auc)
    print("─────────────────────────────────────────────────")
   


# Ejecutamos
pr5("wine.data")
pr5("hepatitis.data")
