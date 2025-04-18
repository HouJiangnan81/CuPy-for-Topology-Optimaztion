import numpy as np
import cupy as cp
from scipy import ndimage
from Index import killer
def new_original_value(struct,U):
    nelz, nely, nelx = struct.shape;    struct = struct.get();    U = U.get()
    labeled_array, num_features = ndimage.label(struct,np.ones((3,3,3)))
    biggest_label_volume = 0;    biggest_label_number = 0
    for i in range(1,num_features+1):
        label_volume = ndimage.sum(struct, labeled_array, index=i)
        if label_volume > biggest_label_volume:
            biggest_label_number = i
            biggest_label_volume = label_volume
    structure_connected = labeled_array==biggest_label_number

    struct = struct * structure_connected
    struct = cp.asarray(struct,'f');    U = cp.asarray(U,'f')

    tunion = killer(struct, nelz, nely, nelx)
    triple_struct = cp.zeros((3*(nelz+1)*(nely+1)*(nelx+1)),'f')
    triple_struct[tunion.astype(cp.int32)] = 1
    U = U * triple_struct
    return struct, U
