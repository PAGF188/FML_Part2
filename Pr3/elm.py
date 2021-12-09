import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import logistic

def train_elm(x,y,h):
    A = 2*np.random.rand(h,x.shape[1])-1
    H = insertarActivacion(np.matmul(A,x.T))
    B = np.matmul(np.linalg.pinv(H.T),y)
    z = np.matmul(H.T,B)
    perf = mean_squared_error(y,z)
    print(perf)

dataset = 'hepatitis.data' 
fn = 'hepatitis.data'
x=np.loadtxt(fn)
y=x[:,0]-1; x=np.delete(x,0,1)
C = len(np.unique(y))
print("Dataset:", dataset)

train_elm(x,y,10)