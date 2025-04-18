import cupy as cp

def killer(struct, nelz, nely, nelx):

    augstruct = cp.zeros((nelz, nely+1, nelx+1),'f');    augstruct[:,:-1,:-1] = struct;    augstruct = augstruct.flatten()
    alive =cp.asarray(cp.nonzero(augstruct),cp.int32)*3


    cx = (nelx + 1) * 3;    cxy = (nelx + 1) * (nely + 1) * 3
    caplus = [    0,    1,    2,    3,    4,    5,    cx+3,    cx+4,    cx+5,    cx,    cx+1,    cx+2,
              cxy+0,cxy+1,cxy+2,cxy+3,cxy+4,cxy+5,cxy+cx+3,cxy+cx+4,cxy+cx+5,cxy+cx,cxy+cx+1,cxy+cx+2]
    tunion=alive
    for killi in range(24):
        tunion = cp.union1d(tunion, alive+caplus[killi])
    #print(tunion.dtype)
    augstruct = None; alive=None
    return cp.sort(tunion)

def survivor(struct):
    nelz, nely, nelx = struct.shape
    augstruct = cp.zeros((nelz, nely+1, nelx+1),'f');    augstruct[:,:-1,:-1] = struct;    augstruct = augstruct.flatten()
    alive =cp.asarray(cp.nonzero(augstruct),cp.int32)*3
    augstruct = None
    return alive


def getij(nelz, nely, nelx):

    tem1 = cp.arange(0, nelx , dtype=int)
    t0 = cp.array([0]) 
    for i in range(nely):
        t0 = cp.concatenate((t0, tem1 + (nelx + 1) * i), axis=0)
    t0 = t0[1:]  
    t = cp.array([0]) 
    for i in range(nelz):
        t = cp.concatenate((t, t0 + (nelx + 1) * (nely + 1) * i), axis=0)
    t = t[1:]  
    t = t*3
    return t



