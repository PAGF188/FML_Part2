# NN sintonizando o no. V de vecinhos con validacion cruzada 
#   K-fold e particions de entrenamento, validacion e teste
from numpy import *
from sklearn.svm import *
from sklearn.metrics import *

dataset='hepatitis';  # hepatitis (2 clases), wine (3 clases)
nf='%s.data'%dataset;x=loadtxt(nf)
y=x[:,0]-1;x=delete(x,0,1);C=len(unique(y))
print('SVC dataset %s'%dataset)

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

# preprocesamento: media 0, desviacion 1-----------------------
for k in range(K):
	med=mean(tx[k],0);dev=std(tx[k],0)
	tx[k]=(tx[k]-med)/dev
	vx[k]=(vx[k]-med)/dev
	sx[k]=(sx[k]-med)/dev
# sintonizacion de hiper-parametros-----------------------------
kappa_mellor=-100;kappa=zeros([1,K]);
vL=2.**arange(-5,16,2);nL=len(vL);  # regularizacion (lambda)
vG=2.**arange(-10,11,2);nG=len(vG); # ancho cerne gausiano (gamma)
vkappa=zeros([nL,nG]);kappa=zeros(K);kappa_mellor=-inf;
print('%10s %15s %10s %10s'%('Lambda','Gamma','Kappa','Best'))
for i in range(nL):
    L=vL[i]
    for j in range(nG):
        G=vG[j]
        for k in range(K):
            modelo=SVC(C=L,kernel='rbf',gamma=G,verbose=False).fit(tx[k],ty[k])
            z=modelo.predict(vx[k])
            kappa[k]=100*cohen_kappa_score(vy[k],z)
        kappa_med=mean(kappa);vkappa[i,j]=kappa_med
        if kappa_med>kappa_mellor:
            kappa_mellor=kappa_med;L_mellor=L;G_mellor=G
        print('%10i %15g %10.1f %10.1f'%(L,G,kappa_med,kappa_mellor))
print('L_mellor=%g G_mellor=%g kappa=%.1f%%'%(L_mellor,G_mellor,kappa_mellor))
from pylab import *
# grafica coa sintonizacion dos hiper-parametros L,G-------
figure(1);clf();u=ravel(vkappa);plot(u);grid(True)
axis([1,len(u),-5,100])
xlabel('Configuracion');ylabel('Kappa (%)')
title('Kappa (%%) sintonizacion de SVC %s'%dataset)
show()
savefig('sintonizacion_svc_%s.eps'%dataset);show()
#grafica 3D-----------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
fig=figure(2);clf();ax=Axes3D(fig)
[X,Y]=meshgrid(log2(vL),log2(vG));ax.plot_surface(X,Y,vkappa,rstride=1,cstride=1,cmap='hot')
xlabel('$log_2 \lambda$');ylabel('$log_2 \gamma$')
title('Kappa (%%) sintonizacion SVC 3D %s'%dataset);colorbar
show()
# mapa de calor--------------------------------------------
figure(3);clf();imshow(vkappa);colorbar()
xlabel('Regularizacion ($log_2 \lambda$)');ylabel('Ancho do cerne gausiano ($log_2 \gamma$)')
title('Sintonizacion SVC mapa calor %s'%dataset)
show()
# test-----------------------------------------------------
mc=zeros([C,C])
if C==2:
	pre=zeros(K);re=zeros(K);f1=zeros(K)
for k in range(K):
	x=vstack((tx[k],vx[k]));y=concatenate((ty[k],vy[k]))
	modelo=SVC(C=L_mellor,kernel='rbf',gamma=G_mellor,verbose=False).fit(x,y)
	z=modelo.predict(sx[k]);y=sy[k]
	kappa[k]=100*cohen_kappa_score(y,z)
	mc+=confusion_matrix(y,z)
	if C==2:
		pre[k]=precision_score(y,z)
		re[k]=recall_score(y,z)
		f1[k]=f1_score(y,z)
kappa_med=mean(kappa);mc/=K
print('SVC dataset=%s L=%g G=%g kappa=%.1f%%'%(dataset,L_mellor,G_mellor,kappa_med))
print('matriz de confusion:'); print(mc)
