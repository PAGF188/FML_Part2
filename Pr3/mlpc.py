from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
import warnings
warnings.filterwarnings("ignore")

dataset = 'hepatitis.data' 
fn = 'hepatitis.data'
x=loadtxt(fn)
y=x[:,0]-1; x=delete(x,0,1)
C = len(unique(y))
print("Dataset:", dataset)

def crea_folds(x,y,K):
    from numpy.random import shuffle,seed
    seed(100)
    [N,n] = x.shape; C = len(unique(y))
    ntf = K-2; nvf=1

    tx=[]; ty=[]; vx=[]; vy=[]; sx=[]; sy=[]
    for i in range(K):
        tx.append(zeros([1,n]))
        vx.append(zeros([1,n]))
        sx.append(zeros([1,n]))

        ty.append(array([],'int'))
        vy.append(array([],'int'))
        sy.append(array([],'int'))

    for i in range(C):
        # t-> indices de patrones de la clase i
        t=where(y==i)[0]; npc=len(t)
        shuffle(t)
        npf=int(npc/K); ntp=npf*ntf
        nvp = npf*nvf; nsp=npc-ntp-nvp
        start=0; u=[]
        for k in range(K):
            p=start; u=array([],'int')
            for l in range(ntp):
                m=t[p]; u=append(u,m); p=(p+1)%npc
            tx[k]=vstack((tx[k],x[u]))
            ty[k]=append(ty[k],y[u]); u=array([],'int') 
            for l in range(nvp):
                m=t[p]; u=append(u,m); p=(p+1)%npc
            vx[k]=vstack((vx[k],x[u]))
            vy[k]=append(vy[k],y[u]); u=array([],'int') 
            for l in range(nsp):
                m=t[p]; u=append(u,m); p=(p+1)%npc
            sx[k]=vstack((sx[k],x[u]))
            sy[k]=append(sy[k],y[u]); start = start+npf
    for k in range(K):
        tx[k]= delete(tx[k],0,0); vx[k]=delete(vx[k],0,0)
        sx[k]=delete(sx[k],0,0)
    return [tx,ty,vx,vy,sx,sy]

K=4
[tx,ty,vx,vy,sx,sy] = crea_folds(x,y,K)

#preprocesamiento
for k in range(K):
    med=mean(tx[k],0); dev=std(tx[k],0)
    tx[k]=(tx[k]-med)/dev
    vx[k]=(vx[k]-med)/dev
    sx[k]=(sx[k]-med)/dev

#sintonizacion
H=3
IK=30
vkappa=zeros(K); kappa_mellor=-Inf
neurons_mellor = []
for i in range(1,H+1):
    print('%10s '%('H%i'%i),end='')
print('%10s'%('Kappa(%)'))

for h in range(1,H+1):
    for j in range(10,IK+1,10):
        neurons=[C]
        for m in range(1,h+1):
            neurons.insert(0,j)
            for k in range(K):
                modelo = MLPClassifier(hidden_layer_sizes=neurons).fit(tx[k],ty[k])
                z=modelo.predict(vx[k])
                vkappa[k]=cohen_kappa_score(vy[k],z)
        kappa_med=mean(vkappa)
        for i in range(h):
            print('%10i'%(neurons[i]),end='')
        for i in range(h+1, H+1):
            print('%10s'%'',end='')
        print('%10.1f'%(100*kappa_med))
        if kappa_med>kappa_mellor:
            kappa_mellor=kappa_med
            neurons_mellor = neurons
print('Mejor arquitectura'); print(neurons_mellor[: -1])
print('kappa=%.2f%%'%(100*kappa_mellor))

#test
mc = zeros([C,C])
if C==2:
    pre=zeros(K); re=zeros(K); f1=zeros(K)
for k in range(K):
    tx[k]=vstack((tx[k],vx[k]))
    ty[k]=concatenate((ty[k],vy[k]))
    modelo = MLPClassifier(hidden_layer_sizes=neurons_mellor).fit(tx[k],ty[k])
    z=modelo.predict(sx[k]); y=sy[k]
    vkappa[k]=100*cohen_kappa_score(y,z)
    mc+=confusion_matrix(y,z)
    if C==2:
        pre[k]=100*precision_score(y,z)
        re[k]=100*recall_score(y,z)
        f1[k]=100*f1_score(y,z)

kappa=mean(vkappa);mc/=K
print('Dataset=%s kappa=%.2f%%'%(dataset,kappa))
print("Matriz confusion:")
print(mc)

if C==2:
    print('prec=%.1f%% rec=%.1f%% f1=%.1f%%'%(mean(pre),mean(re),mean(f1)))



