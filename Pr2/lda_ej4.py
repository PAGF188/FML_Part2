from numpy import *
from sklearn.discriminant_analysis import *
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import *
from crea_folds import *
import pdb

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def ejercicio4(dataset):

    x=loadtxt(dataset)

    y=x[:,0]-1; x=delete(x,0,1)
    C = len(unique(y))

    # Leave-one-pattern-out
    K_=len(y)
    errores = 0

    for i in range(K_):
        patron_test_x = reshape(x[i,:],(1,-1))
        patron_test_y = array([y[i]])
        x_nuevo = delete(x,i,0)
        y_nuevo = delete(y,i,0)

        med=mean(x_nuevo,0); dev=std(x_nuevo,0)
        x_nuevo=(x_nuevo-med)/dev
        patron_test_x=(patron_test_x-med)/dev

        modelo = LinearDiscriminantAnalysis().fit(x_nuevo, y_nuevo)
        z = modelo.predict(patron_test_x); 

        errores += abs(z-patron_test_y)

    #kappa = mean(v_kappa)
    accuracy = (K_ - errores) / K_
    print(accuracy)

    # CON LA AYUDA DE SKLEARN

    x=loadtxt(dataset)
    y=x[:,0]-1; x=delete(x,0,1)

    scaler = StandardScaler()
    loo = LeaveOneOut()
    modelo = LinearDiscriminantAnalysis()
    metrica = 'accuracy'
    pipeline = Pipeline([('transformer', scaler), ('estimator', modelo)])
    scores = cross_val_score(pipeline, x, y, scoring=metrica, cv=loo, n_jobs=-1)
    
    print(mean(scores*100))


# -------------------------------------------------------------------------------------
print("EJERCICIO 4 !!!!\n")

# PARTE 1: Para WINE dataset      
print("-> WINE DATASET")                           
ejercicio4("wine.data")

# -------------------------------------------------------------------------------------
# PARTE 2: Para HEPATITIS dataset    
print("\n-> HEPATITIS DATASET")                              
ejercicio4("hepatitis.data")
