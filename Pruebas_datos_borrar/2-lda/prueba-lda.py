import numpy as np
from sklearn.neighbors import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from crea_folds import crea_folds

import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter

def ej5_2(dataset_name, dataset_train, dataset_test):
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
    t0 = perf_counter()
    for k in range(K_):
        modelo = LinearDiscriminantAnalysis().fit(tx[k],ty[k])
        z = modelo.predict(sx[k]); y=sy[k]
        v_kappa[k] = 100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    print(f"Tiempo: {perf_counter()- t0}")
    kappa = np.mean(v_kappa); accuracy = np.mean(v_accuracy); mc/=K_

    # cf_image = sns.heatmap(mc, cmap='Blues')
    # figure = cf_image.get_figure()    
    # figure.savefig('1NN_4folds_cm_' + dataset_name + '.png'); plt.clf()

    print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}\n")
    print("─────────────────────────────────────────")


ej5_2("Coocur", '../data/trainCoocur.dat', '../data/testCoocur.dat')
ej5_2("LBP", '../data/trainLBP.dat', '../data/testLBP.dat')