from numpy import *
from sklearn.discriminant_analysis import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import *
from crea_folds import *

def ejercicio3(dataset):

    x=loadtxt(dataset)

    y=x[:,0]-1; x=delete(x,0,1)
    C = len(unique(y))


    # Generamos K folds
    K_=4
    [tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K_)

    # Juntamos train y validación (en este ejercicio no hay sintonización de params.)
    for k in range(K_):
        tx[k]=vstack((tx[k],vx[k]))
        ty[k]=concatenate((ty[k],vy[k]))

        # preprocesamos
        med=mean(tx[k],0); dev=std(tx[k],0)
        tx[k]=(tx[k]-med)/dev
        sx[k]=(sx[k]-med)/dev

    mc = zeros([C,C])
    v_kappa=zeros(K_)
    v_accuracy = zeros(K_)
    
    for k in range(K_):
        modelo = LinearDiscriminantAnalysis().fit(tx[k],ty[k])
        z = modelo.predict(sx[k]); y=sy[k]
        v_kappa[k] = 100*cohen_kappa_score(y,z)
        v_accuracy[k] = 100*accuracy_score(y,z)
        mc+=confusion_matrix(y,z)

    kappa = mean(v_kappa)
    accuracy = mean(v_accuracy)
    mc/=K_

    print("Kappa: %.2f acc: %.2f "%(kappa,accuracy) )
    print("Matriz de confusion: \n", mc)

    # Forma alternativa:
    # modelo = LinearDiscriminantAnalysis()
    # K=4
    # z = cross_val_predict(modelo,x,y,cv=K)
    # kappa = 100 * cohen_kappa_score(y,z)
    # acc = 100 * accuracy_score(y,z)
    # cf=confusion_matrix(y,z)
    # print("Kappa: " + str(kappa) + " acc: " + str(acc))
    # print("Matriz de confusion: \n", cf)


# -------------------------------------------------------------------------------------
print("EJERCICIO 3 !!!!\n")

# PARTE 1: Para WINE dataset      
print("-> WINE DATASET")                           
ejercicio3("wine.data")

# -------------------------------------------------------------------------------------
# PARTE 2: Para HEPATITIS dataset    
print("\n-> HEPATITIS DATASET")                              
ejercicio3("hepatitis.data")
