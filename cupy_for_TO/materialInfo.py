import cupy as cp

""""""""""""""""""""""""""""""""""""#Enu2Lame
def Enu2Lame(E,nu):
    mu=E/(2*(1+nu))
    lambdaa =E*nu/((1+nu)*(1-nu))
    return mu,lambdaa

""""""""""""""""""""""""""""""""""""#Dmatrix
def Dmatrix(mu,lambdaa):
    D1=cp.array([[2,0,0,  0,0,0], [0,2,0,  0,0,0], [0,0,2,  0,0,0], [0,0,0,  1,0,0], [0,0,0,  0,1,0], [0,0,0,  0,0,1]])*mu
    D2=cp.array([[1,1,1,  0,0,0], [1,1,1,  0,0,0], [1,1,1,  0,0,0], [0,0,0,  0,0,0], [0,0,0,  0,0,0], [0,0,0,  0,0,0]])*lambdaa
    return D1+D2

""""""""""""""""""""""""""""""""""""#Gausspoints
def Gausspoints():
    qwgts = cp.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])/8
    rspts = cp.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]])/cp.sqrt(3)#正方体的高斯点
    return rspts, qwgts

""""""""""""""""""""""""""""""""""""#referanceunit
def referanceunit():
    x = cp.array([0,1,1,0,0,1,1,0])
    y = cp.array([0,0,1,1,0,0,1,1])
    z = cp.array([0,0,0,0,1,1,1,1])
    return x, y, z

""""""""""""""""""""""""""""""""""""#cubeshapes
def cubeshapes(r, s, tt):
    r1=1-r
    r2=1+r
    s1=1-s
    s2=1+s
    t1=1-tt
    t2=1+tt
    # 计算S、dSdr、dSds和dSdt
    S = cp.array([r1 * s1 * t1,
                  r2 * s1 * t1,
                  r2 * s2 * t1,
                  r1 * s2 * t1,
                  r1 * s1 * t2,
                  r2 * s1 * t2,
                  r2 * s2 * t2,
                  r1 * s2 * t2]) / 8

    dSdr = cp.array([-s1*t1,
                      s1*t1,
                      s2*t1,
                     -s2*t1,
                     -s1*t2,
                      s1*t2,
                      s2*t2,
                     -s2*t2]) / 8
    dSds = cp.array([-r1*t1,
                     -r2*t1,
                      r2*t1,
                      r1*t1,
                     -r1*t2,
                     -r2*t2,
                      r2*t2,
                      r1*t2]) / 8
    dSdt = cp.array([-r1*s1,
                     -r2*s1,
                     -r2*s2,
                     -r1*s2,
                      r1*s1,
                      r2*s1,
                      r2*s2,
                      r1*s2]) / 8
    return S, dSdr, dSds, dSdt

""""""""""""""""""""""""""""""""""""#Isopmap
def Isopmap(x, y, z, r, s, tt ):
    S, dSdr, dSds, dSdt = cubeshapes(r, s, tt)

    j11 = cp.sum(dSdr * cp.array(x), axis=0)
    j12 = cp.sum(dSdr * cp.array(y), axis=0)
    j13 = cp.sum(dSdr * cp.array(z), axis=0)
    j21 = cp.sum(dSds * cp.array(x), axis=0)
    j22 = cp.sum(dSds * cp.array(y), axis=0)
    j23 = cp.sum(dSds * cp.array(z), axis=0)
    j31 = cp.sum(dSdt * cp.array(x), axis=0)
    j32 = cp.sum(dSdt * cp.array(y), axis=0)
    j33 = cp.sum(dSdt * cp.array(z), axis=0)
    detJ = j11*j22*j33 - j11*j23*j32 - j12*j21*j33 + j12*j23*j31 + j13*j21*j32 - j13*j22*j31
    #print(detJ)
    dSdx = ((j22 * j33 - j23 * j32) * dSdr + (j13 * j32 - j12 * j33) * dSds + (j12 * j23 - j13 * j22) * dSdt) / detJ
    dSdy = ((j23 * j31 - j21 * j33) * dSdr + (j11 * j33 - j13 * j31) * dSds + (j13 * j21 - j11 * j23) * dSdt) / detJ
    dSdz = ((j21 * j32 - j22 * j31) * dSdr + (j12 * j31 - j11 * j32) * dSds + (j11 * j22 - j12 * j21) * dSdt) / detJ

    return S, dSdx, dSdy, dSdz, detJ
#已通过验收

def materialInfo(E,nu):

    mu,lambdaa=Enu2Lame(E,nu)
    D=Dmatrix(mu,lambdaa)
    rspts,qwgts=Gausspoints()
    x,y,z=referanceunit()
    KK = cp.zeros((24, 24))#,cp.float32)
    for q in range(8):  # 
        r = rspts[q, 0]  # r 
        s = rspts[q, 1]  # s 
        tt= rspts[q, 2]  # t 
        S, b, c, d, detJ = Isopmap(x, y, z, r, s, tt)
        wxarea = qwgts[q]*abs(detJ)* 8 

        BK = cp.zeros((6, 24))
        for k in [0,1,2,3,4,5,6,7]:
            k3=k*3
            BK[0,0+k3]=b[k]
            BK[1,1+k3]=c[k]
            BK[2,2+k3]=d[k]

            BK[3,0+k3]=c[k]
            BK[3,1+k3]=b[k]
            BK[4,1+k3]=d[k]
            BK[4,2+k3]=c[k]
            BK[5,0+k3]=d[k]
            BK[5,2+k3]=b[k]
        KKq = cp.matmul(cp.matmul(BK.T, D), BK)
        KKq = KKq * wxarea
        KK = KK + KKq
    return KK

def Q_r(E,nu):
    KK=materialInfo(E,nu)
    for i in range(24):
        for j in range(24):
            if i!=j:
                KK[i,j] = 2 * KK[i,j]
    return KK