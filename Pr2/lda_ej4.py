from numpy import *
from sklearn.discriminant_analysis import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import *
from crea_folds import *
import pdb

def ejercicio4(dataset):

    x=loadtxt(dataset)

    y=x[:,0]-1; x=delete(x,0,1)
    C = len(unique(y))

    # Leave-one-pattern-out
    K_=len(y)
    mc = zeros([C,C])
    v_kappa=zeros(K_)
    v_accuracy = zeros(K_)

    for i in range(K_):
        patron_x = reshape(x[i,:],(1,-1))
        patron_y = array([y[i]])
        x_nuevo = delete(x,i,0)
        y_nuevo = delete(y,i,0)

        med=mean(x_nuevo,0); dev=std(x_nuevo,0)
        x_nuevo=(x_nuevo-med)/dev
        patron_x=(patron_x-med)/dev

        modelo = LinearDiscriminantAnalysis().fit(x_nuevo, y_nuevo)
        z = modelo.predict(patron_x); 

        v_kappa[i] = 100*cohen_kappa_score(patron_y,z)
        v_accuracy[i] = 100*accuracy_score(patron_y,z)
        #mc+=confusion_matrix(patron_y,z)

    kappa = mean(v_kappa)
    accuracy = mean(v_accuracy)
    #mc/=K_

    print("Kappa: " + str(kappa) + " acc: " + str(accuracy))
    print("Matriz de confusion: \n", mc)


# -------------------------------------------------------------------------------------
print("EJERCICIO 4 !!!!\n")

# PARTE 1: Para WINE dataset      
print("-> WINE DATASET")                           
ejercicio4("wine.data")

# -------------------------------------------------------------------------------------
# PARTE 2: Para HEPATITIS dataset    
print("\n-> HEPATITIS DATASET")                              
ejercicio4("hepatitis.data")
