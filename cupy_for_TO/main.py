import cupy as cp;  import numpy as np; import time
from kernel_writer import autowrite;  from Index import killer
from ESO import ESO;  from new_original_value import new_original_value

nelx=128;  nely=64;  nelz=64;  N=nelz*nely*nelx
cx = (nelx + 1) * 3;    cxy = (nelx + 1) * (nely + 1) * 3
caplus =[    0,    1,    2,    3,    4,    5,    cx+3,    cx+4,    cx+5,    cx,    cx+1,    cx+2,
         cxy+0,cxy+1,cxy+2,cxy+3,cxy+4,cxy+5,cxy+cx+3,cxy+cx+4,cxy+cx+5,cxy+cx,cxy+cx+1,cxy+cx+2]

tol=0.001;  Vreq=0.1;  Q=0.8;  M=20;  E=1;  nu=0.3;  filterF=cp.ones((3,3,3));  G=int(N*(1-Vreq)*(1-Q)/(1-Q**M))
useless=autowrite(nelz, nely, nelx,E,nu);   from shapesens import shapesens
struct = cp.ones((nelz,nely,nelx),'f');  U = cp.zeros((nelx+1)*(nely+1)*(nelz+1)*3,'f')
_t1=time.time();  timelist = [];  relvol = [];  iterTvol = [];  iterationlist=[];  shapesenlist=[]
###############################################################################
for i in range(M):
    print('Iteration',i+1)
    t00=time.time()
    tunion = killer(struct, nelz, nely, nelx)
    itervol=int(sum(sum(sum(struct))))/N
    #print(U.nbytes/1024/1024)
    shapesen, U, iteration = shapesens(struct, U, tol, caplus, E, nu, tunion);shapesum=-sum(sum(sum(shapesen)));shapesenlist.append(shapesum)
    timelist.append(time.time()-t00);relvol.append(itervol);iterationlist.append(iteration)

    struct = ESO(shapesen, struct, filterF, nelz, nely, nelx, G); G = int(Q*G)
    struct, U = new_original_value(struct, U);   itervol=int(sum(sum(sum(struct))))/N;print('volume fraction',itervol)

    _struct=cp.asnumpy(struct);    np.save('struct.npy',_struct)


y = [float("{:.5g}".format(i)) for i in timelist]
s = [float("{:.5g}".format(i)) for i in relvol]
t = [float("{:.5g}".format(i)) for i in iterationlist]
e = [float("{:.5g}".format(i.item())) for i in shapesenlist]

print("time history",y);    print("volume history",s);    print("total_CG_iteration",sum(t),t);    print("objective history",e)
t3=time.time();     print("total time", round(t3-_t1,2));    


