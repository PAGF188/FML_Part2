from numpy import *

def crea_folds(x,y,K):
    """ Crea folds.
    :param x matriz de patrones
    :param y vector de etiquetas
    :param K nº de "folds"
    """
    from numpy.random import shuffle, seed
    seed(100)
    [N,n] = x.shape; C = len(unique(y))
    ntf = K-2; nvf = 1

    ti =[[]]*K; vi =[[]]*K ; si =[[]]*K

    for i in range(C):
        # t-> indices de patrones de la clase i
        t=where(y==i)[0]; npc=len(t)
        shuffle(t)
        npf=int(npc/K); ntp=npf*ntf
        nvp = npf*nvf; nsp=npc-ntp-nvp
        start=0
        for k in range(K):
            p=start; u=[]
            for l in range(ntp):
                u.append(t[p]); p=(p+1)%npc
            ti[k]=ti[k]+u; u=[]
            for l in range(nvp):
                u.append(t[p]); p=(p+1)%npc
            vi[k]=vi[k]+u; u=[]
            for l in range(nsp):
                u.append(t[p]); p=(p+1)%npc
            si[k]=si[k]+u; start=start+npf

    tx=[]; ty=[]; vx=[]; vy=[]; sx=[]; sy=[]
    for k in range(K):
        i = ti[k]; tx.append(x[i,:]); ty.append(y[i])
        i = vi[k]; vx.append(x[i,:]); vy.append(y[i])
        i = si[k]; sx.append(x[i,:]); sy.append(y[i])
    return [tx,ty,vx,vy,sx,sy]
