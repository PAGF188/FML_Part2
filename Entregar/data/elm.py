from numpy import *
from sklearn.metrics import *
from sys import exit

#dataset='hepatitis'
dataset='wine'

nf='%s.data'%dataset;x=loadtxt(nf)
y=x[:,0]-1;x=delete(x,0,1);
N=shape(x)[0];C=len(unique(y))
print('ELM dataset %s'%dataset)

def vec2ind(x):
	n=len(x);m=len(unique(x));y=zeros([n,m])
	for i in range(n):
		j=int(x[i]);y[i,j]=1
	return y

def elm(x,y,h,z):
	from numpy.random import random
	from numpy.linalg import pinv
	#from sys import exit
	N,n=shape(x);C=len(unique(y))
	a=2*random([h,n])-1  # input weights
	H=1/(1+exp(-dot(a,x.T))); # matrix with activity of neurons in hidden layer 
	t=vec2ind(y)  # true outputs for neurons in the output layer (t[i,j]=1 only if pattern i is of class j)
	b=dot(pinv(H.T),t)  # output weights
	# output calculation--------------------------
	H=1/(1+exp(-dot(a,z.T)))  # activity in hidden layer
	q=dot(H.T,b)  # outputs of neurons in output layer
	z=argmax(q,1)  # predicted class label
	return z

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
vkappa=zeros(K);kappa_best=-Inf;h_best=1
print('%10s %10s'%('H','Kappa(%)'))
for h in range(3,N):
	for k in range(K):
		z=elm(tx[k],ty[k],h,vx[k])
		vkappa[k]=100*cohen_kappa_score(vy[k],z)
	kappa_mean=mean(vkappa)
	print('%10i %10.1f'%(h,kappa_mean))			
	if kappa_mean>kappa_best:
		kappa_best=kappa_mean;h_best=h
print('h_best=%i kappa_best=%.1f%%'%(h_best,kappa_best))
mc=zeros([C,C])
if C==2:
	pre=zeros(K);re=zeros(K);f1=zeros(K)
for k in range(K):
	tx[k]=vstack((tx[k],vx[k]));ty[k]=concatenate((ty[k],vy[k]))
	mx=mean(tx[k],0);stdx=std(tx[k],0);tx2=(tx[k]-mx)/stdx
	sx2=(sx[k]-mx)/stdx;y=sy[k]
	z=elm(tx2,ty[k],h_best,sx2)
	#modelo=MLPClassifier(hidden_layer_sizes=neurons).fit(tx2,ty[k])
	#sx2=(sx[k]-mx)/stdx
	#z=modelo.predict(sx2);y=sy[k]
	vkappa[k]=100*cohen_kappa_score(y,z)
	mc+=confusion_matrix(y,z)
	if C==2:
		pre[k]=precision_score(y,z)
		re[k]=recall_score(y,z)
		f1[k]=f1_score(y,z)
kappa=mean(vkappa);mc/=K
print('ELM dataset=%s kappa=%.2f%%'%(dataset,kappa))
print('confusion matrix:'); print(mc)
