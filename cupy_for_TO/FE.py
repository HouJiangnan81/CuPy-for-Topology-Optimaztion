import cupy as cp;from CG import CG;import math


def FE(struct, U, tol, caplus, tunion):
    nelz, nely, nelx = struct.shape
    np = (nelx + 1) * (nely + 1) * (nelz + 1)
    oo = 3 * (nelx + 1) * (nely + 1)
    o1=oo-3*nelx
    ou=nelz*oo
    fixeddofs = (cp.array([1,2,3,o1-2,o1-1,o1,1+ou,2+ou,3+ou,o1-2+ou,o1-1+ou,o1+ou])-1).astype(cp.int32)


    Fz = cp.zeros((nelz + 1, nely + 1, nelx + 1),'f')
    r = 1

    Fz[ int((nelz + 1) / 2) - r:int((nelz + 1) / 2) + r,
        int((nely + 1) / 2) - r:int((nely + 1) / 2) + r,
        -2* r:] = 1
    Fz = Fz.flatten()

    F = cp.zeros((3 * np),'f')
    F[2::3] = Fz
    Fz = None


    st, i = CG(struct, F, U, tol, 10000, fixeddofs, caplus, tunion)
    return st, i
