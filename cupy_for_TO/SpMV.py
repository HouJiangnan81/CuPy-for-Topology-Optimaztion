from ker import *
add_r=[add_roll0, add_roll1, add_roll2, add_roll3, add_roll4, add_roll5, add_roll6, add_roll7,
       add_roll8, add_roll9, add_roll10,add_roll11,add_roll12,add_roll13,add_roll14,add_roll15,
       add_roll16,add_roll17,add_roll18,add_roll19,add_roll20,add_roll21,add_roll22,add_roll23]

def honorAb(b,t11,caplus,np3,tunion):

    sum1 = cp.zeros((np3), dtype='f')
    sumsub = sum1[tunion]
    for mulj in range(24):     
        Ab0j = cp.zeros((np3),dtype='f')
        Ab0j[t11] =b[t11 + caplus[mulj]]
        #print(Ab0j.dtype,b.dtype,t11.dtype)
        sumsub = add_r[mulj](sumsub,tunion, Ab0j)
    sum1[tunion] = sumsub
    return sum1
