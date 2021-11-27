from numpy import *

def crea_folds(x,y,K):
    """ Crea folds.
    :param x matriz de patrones
    :param y vector de etiquetas
    :param K nº de "folds"
    """
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