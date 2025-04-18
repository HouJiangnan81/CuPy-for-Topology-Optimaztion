import numpy as np
import cupy as cp
def mesh(nelz,nely,nelx):
    x = np.arange(0, nelx + 1 , 1) 
    y = np.arange(0, nely + 1 , 1)
    z = np.arange(0, nelz + 1 , 1)
    X, Y, Z = np.meshgrid(x, y, z)

    del x, y, z
    X = X.swapaxes(2, 1)
    Y = Y.swapaxes(2, 1).swapaxes(1, 0)
    Z = Z.swapaxes(2, 1).swapaxes(1, 0)
    p = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    return p.T