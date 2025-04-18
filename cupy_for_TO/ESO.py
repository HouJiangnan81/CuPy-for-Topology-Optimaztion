import cupy as cp;  import numpy as np
from scipy.ndimage import correlate
def ESO(shapesen, struct, filterF, nelz, nely, nelx, G):
    
    shapesen = shapesen.get(); struct = struct.get()
    shapesen = correlate(shapesen, filterF.get())
    #shapesen = shapesen + np.flip(shapesen, 2)
    shapesen = shapesen + np.flip(shapesen, 1)
    #shapesen = shapesen + shapesen.swapaxes(1, 2)
    
    shapesen[struct < 0.001] = -100000000000
    maxk = cp.sort(shapesen.flatten())
    struct[shapesen >= maxk[-G]] = 0
    r = 1
    
    struct[ int((nelz + 1) / 2) - r:int((nelz + 1) / 2) + r,
            int((nely + 1) / 2) - r:int((nely + 1) / 2) + r,
            -2* r:] = 1
   
    return cp.asarray(struct)