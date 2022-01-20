"""
Practica 5: Exercise 4,5 with ensembles
Autor: Pablo García Fernández.

Requirements
------------
-Numpy
-Scikit-learn
-Matplotlib
-Seaborn (to plot confusion matrix as image)
-Time

"""

from statistics import mode
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.model_selection import *

import matplotlib.pyplot as plt
from crea_folds import crea_folds
import numpy as np
from time import perf_counter
import seaborn as sns
import pdb      

def pintar_grafico_sintonizacion(model_name, tunedParameters, dataset_name, means, model, sintonizacion_n):
     # Pintar grafico de sintonizacion
    if model_name=='RF':
        len_max_features = len(tunedParameters['max_features'])
        len_estimadores = len(tunedParameters['n_estimators'])
        v_f_tuning = np.zeros((len_max_features, len_estimadores))
        for i, f1 in enumerate(means):
            i_aux, j_aux = np.unravel_index(i, (len_max_features,len_estimadores))
            v_f_tuning[i_aux,j_aux] = f1

        plt.clf()
        plt.imshow(v_f_tuning); plt.colorbar()
        plt.ylabel('max features');plt.xlabel('no. estimators')
        plt.title(f'RF tuning, {dataset_name}, mejor= {model.best_params_["n_estimators"]}, {model.best_params_["max_features"]}')
        plt.yticks(list(range(0, len_max_features)), tunedParameters['max_features'])
        plt.xticks(list(range(0, len_estimadores)), tunedParameters['n_estimators'])
        #plt.show()
        plt.savefig("ej4_RF_sintonizacion1_" + dataset_name + ".png"); plt.clf()
    elif model_name=='ADA':
        plt.clf()
        plt.plot(tunedParameters['n_estimators'], means, marker='o', label='Evolucion f1-score con n_estimadores')
        plt.title(f'AdaBoost tuning, {dataset_name}, mejor= {model.best_params_["n_estimators"]}')
        plt.xticks(tunedParameters['n_estimators']); plt.legend(); plt.grid(True)
        plt.xlabel('no. estimadores'); plt.ylabel('f1-score')
        plt.savefig("ej4_ADA_sintonizacion"+ sintonizacion_n + "_" + dataset_name + ".png")
        #plt.show(); 
        plt.clf()


def sintonizacion1(dataset_name, model_used, x, y, model_name):
    """
    Approach 1 to tune the hyper-parameters: divide the whole dataset into two
    datasets (train set and test set), using the train set to tune the hyper-parameters
    using the GridSearchCV() function and, once the best values for the hyper-parmeters
    are calculated, test the RF classifier over the test set
    """

    print("=============================================")
    print(f" Sintonizacion1 sobre {dataset_name} Modelo: {model_name}!")
    print("=============================================\n")

    # Separacion train/test
    # Ponemos stratify a True para mantener poboaciones relativas!!
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=180, stratify=y)


    # Dependiendo de RF o ADABOOST sintonizamos unos u otros parametros
    if model_name == 'RF':
        tunedParameters = {'n_estimators': list(range(40, 220, 20)), 'max_features': [3,5,7,9,11,13]}
    elif model_name == 'ADA':
        tunedParameters = {'n_estimators': list(range(40, 220, 20)) }

    score = 'f1' 

    model = GridSearchCV(model_used(random_state=0), tunedParameters, scoring='%s_macro' % score)
    # preprocessing: mean 0, desviation 1
    mx=np.mean(Xtrain,0); stdx=np.std(Xtrain,0)
    Xtrain=(Xtrain-mx)/stdx
    model.fit(Xtrain, ytrain)
    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    # Pintamos grafico sintonizacion
    pintar_grafico_sintonizacion(model_name, tunedParameters, dataset_name, means, model, "1")

    # Test
    Xtest=(Xtest-mx)/stdx
    # Compute the classifier prediction on the test set
    z = model.predict(Xtest)
    kappa = cohen_kappa_score(ytest, z) * 100
    accuracy = accuracy_score(ytest, z) * 100
    cf = confusion_matrix(ytest, z)
    print("acc. = %.2f, kappa=%.2f"%(accuracy, kappa))

    # cm ----------------------------------
    plt.clf()
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej4_cm_sint1'+ model_name + "_" + dataset_name + '.png'); plt.clf()


def sintonizacion2(dataset_name, model_used, x, y, model_name):
    """
    Appoach 2 to tune the hyper-parameters: use the whole dataset to tune the
    hyper-parameters using the function GridSearchCV() and, once the best values for
    the hyper-parmeters are calculated, test the RF classifier over the whole dataset using
    cross-validation (using the cross val predict() function).

    """

    print("=============================================")
    print(f" Sintonizacion2 sobre {dataset_name} Modelo: {model_name}!")
    print("=============================================\n")

    # Dependiendo de RF o ADABOOST sintonizamos unos u otros parametros
    if model_name == 'RF':
        tunedParameters = {'n_estimators': list(range(40, 220, 20)), 'max_features': [3,5,7,9,11,13]}
    elif model_name == 'ADA':
        tunedParameters = {'n_estimators': list(range(40, 220, 20)) }
    
    x=(x-np.mean(x,0))/np.std(x,0)
    score = 'f1'

    model = GridSearchCV(model_used(random_state=0), tunedParameters, scoring='%s_macro' % score)
    model.fit(x, y)
    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    # Pintamos gráfico sintonizacion
    pintar_grafico_sintonizacion(model_name, tunedParameters, dataset_name, means, model, "2")

    # 4-fold cross validation-------------------------------------
    z=cross_val_predict(model, x, y, cv=4)
    acc = accuracy_score(y,z) * 100
    kappa = cohen_kappa_score(y, z) * 100
    cf=confusion_matrix(y,z)
    print("acc. = %.2f, kappa=%.2f"%(acc, kappa))

    # cm ----------------------------------
    plt.clf()
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig('ej4_cm_sint2'+ model_name + "_" + dataset_name + '.png'); plt.clf()

def sintonizacion3(dataset_name, x, y, model_name):

    """
    Appoach 3: Método usado en todas las práctica anteriores.
    Usando la funcion crea folds
    """

    print("=============================================")
    print(f" Sintonizacion3 sobre {dataset_name} Modelo: {model_name}!")
    print("=============================================\n")

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

    # Dependiendo de RF o ADABOOST sintonizamos unos u otros parametros
    if model_name == 'RF':
        vne=list(range(40, 220, 20)) # number of estimators
        vmf=[3,5,7,9,11,13] # max features
        kappa_sintonizacion=np.zeros((len(vmf), len(vne))); 
        vkappa=np.zeros(K);kappa_best=-np.Inf

        print('%10s %10s %10s'%('#estimators','max features','kappa(%)'))
        for i,mf in enumerate(vmf):
            for j,ne in enumerate(vne):
                for k in range(K):
                    model=RandomForestClassifier(n_estimators=ne, max_features=mf,random_state=0).fit(tx[k],ty[k])
                    z=model.predict(vx[k])
                    vkappa[k]=cohen_kappa_score(vy[k],z)
                kappa_mean=np.mean(vkappa)
                kappa_sintonizacion[i,j] = kappa_mean
                print('%10i %10i %10.2f%%'%(ne,mf,100*kappa_mean))
                if kappa_mean>kappa_best:
                    kappa_best=kappa_mean; ne_best=ne; mf_best=mf

        print('Best parameters: #estimators=%i max features=%i kappa=%.2f%%'%(ne_best,mf_best,100*kappa_best))
        
        # Grafico de sintonizacion:
        plt.clf()
        plt.imshow(kappa_sintonizacion); plt.colorbar()
        plt.ylabel('max features');plt.xlabel('no. estimators')
        plt.title(f'RF tuning, {dataset_name}, mejor= {mf_best}, {ne_best}')
        plt.yticks(list(range(0, len(vmf))), vmf)
        plt.xticks(list(range(0, len(vne))), vne)
        #plt.show()
        plt.savefig("ej4_RF_sintonizacion3_" + dataset_name + ".png"); plt.clf()

        print("Test:")
        C = len(np.unique(y))
        mc = np.zeros([C,C])
        v_accuracy = np.zeros(K)

        for k in range(K):
            tx[k]=np.vstack((tx[k],vx[k]))
            ty[k]=np.concatenate((ty[k],vy[k]))
            model=RandomForestClassifier(n_estimators=ne_best, max_features=mf_best,random_state=0).fit(tx[k],ty[k])
            z=model.predict(sx[k]); y=sy[k]
            vkappa[k]=100*cohen_kappa_score(y,z)
            v_accuracy[k] = 100*accuracy_score(y,z)
            mc+=confusion_matrix(y,z)

        kappa = np.mean(vkappa)
        accuracy = np.mean(v_accuracy)
        mc/=K
        print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

        cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
        figure = cf_image.get_figure()    
        figure.savefig('ej4_cm_sint3'+ model_name + "_" + dataset_name + '.png'); plt.clf()


    ### Para ADA
    elif model_name == 'ADA':
        vne=list(range(40, 220, 20)) # number of estimators
        vkappa=np.zeros(K);kappa_best=-np.Inf
        kappa_sintonizacion = []
        print('%10s %10s %10s'%('#estimators','max features','kappa(%)'))
        for ne in vne:
            for k in range(K):
                model=AdaBoostClassifier(n_estimators=ne, random_state=0).fit(tx[k],ty[k])
                z=model.predict(vx[k])
                vkappa[k]=cohen_kappa_score(vy[k],z)
            kappa_mean=np.mean(vkappa)
            kappa_sintonizacion.append(kappa_mean)
            print('%10i %10.2f%%'%(ne,100*kappa_mean))
            if kappa_mean>kappa_best:
                kappa_best=kappa_mean; ne_best=ne

        print('Best parameters: #estimators=%i kappa=%.2f%%'%(ne_best,100*kappa_best))
        
        # Grafico de sintonizacion: 
        plt.clf()
        plt.plot(vne, kappa_sintonizacion, marker='o', label='kappa vs no. estimators')
        plt.title(f'AdaBoost tuning, {dataset_name}, mejor= {model.best_params_["n_estimators"]}')
        plt.legend(); plt.grid(True)
        plt.xlabel('no. estimadores'); plt.ylabel('kappa')
        plt.savefig("ej4_ADA_sintonizacion3_" + dataset_name + ".png")
        #plt.show()

        print("Test:")
        C = len(np.unique(y))
        mc = np.zeros([C,C])
        v_accuracy = np.zeros(K)

        for k in range(K):
            tx[k]=np.vstack((tx[k],vx[k]))
            ty[k]=np.concatenate((ty[k],vy[k]))
            model=AdaBoostClassifier(n_estimators=ne_best,random_state=0).fit(tx[k],ty[k])
            z=model.predict(sx[k]); y=sy[k]
            vkappa[k]=100*cohen_kappa_score(y,z)
            v_accuracy[k] = 100*accuracy_score(y,z)
            mc+=confusion_matrix(y,z)

        kappa = np.mean(vkappa)
        accuracy = np.mean(v_accuracy)
        mc/=K
        print(f"acc.: {accuracy:.2f}%\nkappa: {kappa:.2f}%\ncf = \n{mc}")

        cf_image = sns.heatmap(mc, cmap='Blues', annot=True, fmt='g')
        figure = cf_image.get_figure()    
        figure.savefig('ej4_cm_sint3'+ model_name + "_" + dataset_name + '.png'); plt.clf()




def ej4(dataset_name):

    print("=============================================")
    print(f"   Ejecucion ej4 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # Probamos los 3 metodosde tuning para cada clasificador
    datos = np.loadtxt('../data/' + dataset_name)
    y=datos[:,0]-1; x=np.delete(datos,0,1)
    
    sintonizacion1(dataset_name, RandomForestClassifier, x, y, 'RF')
    sintonizacion1(dataset_name, AdaBoostClassifier, x, y, 'ADA')

    sintonizacion2(dataset_name, RandomForestClassifier, x, y, 'RF')
    sintonizacion2(dataset_name, AdaBoostClassifier, x, y, 'ADA')

    sintonizacion3(dataset_name, x, y, 'RF')
    sintonizacion3(dataset_name, x, y, 'ADA')
    


ej4("hepatitis.data")
ej4("wine.data")

    


