import cupy as cp
from materialInfo import Q_r
from FE import FE
from Index import getij

def shapesens(struct,U,tol,caplus,E,nu,tunion):
    nelz, nely, nelx = struct.shape

    QE= Q_r(E,nu)
    U,iteration=FE(struct,U,tol,caplus,tunion)

    shapesen = cp.zeros((nelz * nely * nelx),'f')
    t11 = getij(nelz, nely, nelx)

    for i in range(24):
        coli = cp.zeros((nelz*nely*nelx),'f')
        for j in range(i,24):
            coli = coli + QE[i,j] * U[t11+caplus[j]]
        shapesen = shapesen + U[t11+caplus[i]] * coli
    shapesen = shapesen * struct.flatten()
    shapesen = -cp.reshape(shapesen,(nelz,nely,nelx))
    return shapesen,U,iteration


