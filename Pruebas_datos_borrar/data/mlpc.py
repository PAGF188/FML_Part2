from numpy import *
from sklearn.neural_network import *
from sklearn.metrics import *
#from sklearn.model_selection import *

import warnings
warnings.filterwarnings("ignore")

dataset='hepatitis';  # hepatitis (2 clases), wine (3 clases)
nf='%s.data'%dataset;x=loadtxt(nf)
y=x[:,0]-1;x=delete(x,0,1);C=len(unique(y))
print('MLP dataset %s'%dataset)

def createFolds(x,y,K):
	from numpy.random import shuffle,seed
	seed(100)
	[N,n]=x.shape;C=len(unique(y));ntf=K-2;nvf=1
	ti=[[]]*K;vi=[[]]*K;si=[[]]*K
	for i in range(C):
		t=where(y==i)[0];npc=len(t);shuffle(t)
		npf=int(npc/K);ntp=npf*ntf
		nvp=npf*nvf;nsp=npc-ntp-nvp;start=0
		for k in range(K):
			p=start;u=[]
			for l in range(ntp):
				u.append(t[p]);p=(p+1)%npc
			ti[k]=ti[k]+u;u=[]
			for l in range(nvp):
				u.append(t[p]);p=(p+1)%npc
			vi[k]=vi[k]+u;u=[]
			for l in range(nsp):
				u.append(t[p]);p=(p+1)%npc    
			si[k]=si[k]+u;start=start+npf
	tx=[];ty=[];vx=[];vy=[];sx=[];sy=[]
	for k in range(K):
		i=ti[k];tx.append(x[i,:]);ty.append(y[i])
		i=vi[k];vx.append(x[i,:]);vy.append(y[i])
		i=si[k];sx.append(x[i,:]);sy.append(y[i])
	return [tx,ty,vx,vy,sx,sy]

K=4;
tx,ty,vx,vy,sx,sy=createFolds(x,y,K)

# preprocesamento: media 0, desviacion 1
for k in range(K):
	med=mean(tx[k],0);dev=std(tx[k],0)
	tx[k]=(tx[k]-med)/dev
	vx[k]=(vx[k]-med)/dev
	sx[k]=(sx[k]-med)/dev
vkappa=zeros(K);kappa_mellor=-Inf;
H=3     # number of hidden layers
IK=30   # number of neurons by layer
for i in range(1,H+1):
	print('%10s '%('H%i'%i),end='')
print('%10s'%'Kappa(%)')
for h in range(1,H+1):
	for j in range(10,IK+1,10):
		neurons=[C]
		for m in range(1,h+1):
			neurons.insert(0,j)
		for k in range(K):
			modelo=MLPClassifier(hidden_layer_sizes=neurons).fit(tx[k],ty[k])
			z=modelo.predict(vx[k])
			vkappa[k]=cohen_kappa_score(vy[k],z)
		kappa_med=mean(vkappa)
		for i in range(h):
			print('%10i '%(neurons[i]),end='')			
		for i in range(h+1,H+1):
			print('%10s '%'',end='')
		print('%10.1f'%(100*kappa_med))
		if kappa_med>kappa_mellor:
			kappa_mellor=kappa_med;neurons_mellor=neurons
print('mellor arquitectura');print(neurons_mellor[:-1])
print('kappa=%.1f%%\n'%(100*kappa_mellor))
mc=zeros([C,C])
if C==2:
	pre=zeros(K);re=zeros(K);f1=zeros(K)
for k in range(K):
	tx[k]=vstack((tx[k],vx[k]));ty[k]=concatenate((ty[k],vy[k]))
	mx=mean(tx[k],0);stdx=std(tx[k],0);tx2=(tx[k]-mx)/stdx
	modelo=MLPClassifier(hidden_layer_sizes=neurons).fit(tx2,ty[k])
	sx2=(sx[k]-mx)/stdx
	z=modelo.predict(sx2);y=sy[k]
	vkappa[k]=cohen_kappa_score(y,z)
	mc+=confusion_matrix(y,z)
	if C==2:
		pre[k]=precision_score(y,z)
		re[k]=recall_score(y,z)
		f1[k]=f1_score(y,z)
kappa=mean(vkappa);mc/=K
print('MLP dataset=%s kappa=%.2f%%'%(dataset,100*kappa))
print('matriz de confusion:'); print(mc)
