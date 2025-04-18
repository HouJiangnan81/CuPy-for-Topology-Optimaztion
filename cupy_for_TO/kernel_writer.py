from materialInfo import materialInfo
import numpy as np
import time
def autowrite(nelz, nely, nelx,E,nu):
    K = materialInfo(E,nu).get()

    np3 =(nelx+1)*(nely+1)*(nelz+1)*3
    cx=(nelx+1)*3
    cxy=(nelx+1)*(nely+1)*3
    caplus = [    0,    1,    2,    3,    4,    5,    cx+3,    cx+4,    cx+5,    cx,    cx+1,    cx+2,
              cxy+0,cxy+1,cxy+2,cxy+3,cxy+4,cxy+5,cxy+cx+3,cxy+cx+4,cxy+cx+5,cxy+cx,cxy+cx+1,cxy+cx+2]

    C = [str(-num+np3) for num in caplus]
    #print(C)
    len=str(np3)


    U=np.empty((24,24), dtype='<U15')
    for i in range(24):
        for j in range(24):
            if K[i,j]>0:
                U[i,j]='+'+str(K[i,j])
            else:
                U[i, j] =str(K[i, j])

    with open("ker.py", "w", encoding='utf-8') as file:
        print('import cupy as cp', '\n', file=file)
        print('#nelx=',nelx,'  ,nely=',nely,'  ,nelz=',nelz,'\n', file=file)
        for i in range(24):
            Uu=U[i,:]
            form=str(i)
            print(
                'add_roll'+form+' = cp.ElementwiseKernel(','\n'
                '           \'float32 x,raw int32 t,raw float32 y\',','\n'
                '           \'float32 z'+'\''+','+'\n',
                '\'\'\'\n'
                '           int u = t[i];','\n'
                '           z = ','\n'
                
    '       ','x     ', Uu[0],'*y[u]         ',Uu[1],'*y[(u+'+C[1]+')%'+len+']  ',Uu[2],'*y[(u+'+C[2]+')%'+len+']'   ,'\n'
    '       ', Uu[3],'*y[(u+'+C[3]+')%'+len+']  ',Uu[4],'*y[(u+'+C[4]+')%'+len+']  ',Uu[5],'*y[(u+'+C[5]+')%'+len+']','\n'
    '       ', Uu[6],'*y[(u+'+C[6]+')%'+len+']  ',Uu[7],'*y[(u+'+C[7]+')%'+len+']  ',Uu[8],'*y[(u+'+C[8]+')%'+len+']','\n'
    '       ', Uu[9],'*y[(u+'+C[9]+')%'+len+']  ',Uu[10],'*y[(u+'+C[10]+')%'+len+']  ',Uu[11],'*y[(u+'+C[11]+')%'+len+']','\n','\n'
    
    '       ', Uu[12],'*y[(u+'+C[12]+')%'+len+']  ',Uu[13],'*y[(u+'+C[13]+')%'+len+']  ',Uu[14],'*y[(u+'+C[14]+')%'+len+']','\n'
    '       ', Uu[15],'*y[(u+'+C[15]+')%'+len+']  ',Uu[16],'*y[(u+'+C[16]+')%'+len+']  ',Uu[17],'*y[(u+'+C[17]+')%'+len+']','\n'
    '       ', Uu[18],'*y[(u+'+C[18]+')%'+len+']  ',Uu[19],'*y[(u+'+C[19]+')%'+len+']  ',Uu[20],'*y[(u+'+C[20]+')%'+len+']','\n'
    '       ', Uu[21],'*y[(u+'+C[21]+')%'+len+']  ',Uu[22],'*y[(u+'+C[22]+')%'+len+']  ',Uu[23],'*y[(u+'+C[23]+')%'+len+'];','\n'
            '\'\'\'\n'
                ',','\'add_roll'+str(i)+'\''       ,')'      ,'\n','\n'
                ,file=file)

    time.sleep(2)
    return 0
