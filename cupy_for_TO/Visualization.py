import numpy as np
import matplotlib.pyplot as plt
struct= np.load('struct.npy')
nelz,nely,nelx=struct.shape
vol=sum(sum(sum(struct)))/(nelz*nelx*nely)
print(vol)
m=np.max(struct.flatten())
#struct=struct**(1/3)

nnp=nelz*nely*nelx
struct=struct.flatten()
maxk = np.sort(struct)
#struct[np.where(struct>=maxk[int(nnp*0.9)])] = 1
#struct[np.where(struct<=maxk[int(nnp*0.9)])] = 0
#struct=struct/m
struct = np.reshape(struct, (nelz, nely, nelx))
vol=sum(sum(sum(struct)))/(nelz*nelx*nely)
print(vol)

struct=struct.swapaxes(2, 0)


from mayavi import mlab
src = mlab.pipeline.volume(mlab.pipeline.scalar_field(struct*1000))
lut = src.module_manager.scalar_lut_manager.lut.table.to_array()

mlab.show()