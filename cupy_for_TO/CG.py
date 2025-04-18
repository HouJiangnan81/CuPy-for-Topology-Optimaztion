from SpMV import *
from Index import survivor
import time
import gc


def CG(struct, b, x, tol ,itermax,fixed,caplus,tunion):
    #print(struct.dtype, b.dtype, x.dtype,tunion.dtype)

    nelz, nely, nelx = struct.shape
    np3 = (nelx + 1) * (nely + 1) * (nelz + 1) * 3
    t11 = survivor(struct)

    Ax0=honorAb(x,t11,caplus,np3,tunion)
    Ax0[fixed]=0
    r0 = b - Ax0

    del Ax0
    gc.collect()
    tol = cp.linalg.norm(r0) * tol

    p = r0
    for i in range(itermax):

        Ap = honorAb(p,t11,caplus,np3,tunion)
        Ap[fixed]=0
        r02=cp.dot(r0, r0)
        alpha = r02 / cp.dot(p, Ap)
        x = x + alpha * p
        r1 = r0 - alpha * Ap
        if (cp.linalg.norm(r1) < tol):
            print('CG_iteration',i)
            break
        beta = cp.dot(r1, r1) / r02
        p = r1 + beta * p
        r0 = r1
      
    return x,i
