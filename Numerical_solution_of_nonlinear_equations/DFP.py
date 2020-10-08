'''
Author: Abel-Cat
Date: 2020-10-08 15:41:32
LastEditors: Abel-Cat
LastEditTime: 2020-10-08 15:55:37
Motto: May the force be with you.
'''

import numpy as np

def F(x):
    y=np.array([  [np.power(x[0][0],2)+np.power(x[1][0],2)-5],
                  [(x[0][0]+1)*x[1][0]-(3*x[0][0]+1)] ])
    return y

def DFP(x0 ,B0 , N , e):
    """
    DFP method  
    """
    k=0
    err=1
    xk=x0
    Bk=B0

    while k<N and err>e:
        sk = -np.dot(Bk,F(xk))
        err = np.linalg.norm(sk)
        xk1=xk+sk
        yk = F(xk1) - F(xk)
        add_B = ( ( sk @ np.transpose(sk) ) /( np.transpose(sk)@yk ) ) - ( ( Bk@yk@np.transpose(yk)@Bk ) / (np.transpose(yk)@Bk@yk)  )
        Bk=Bk+add_B
        k = k+1
        xk = xk1

    return xk

if __name__ == "__main__":
    x0 = np.array([ [1],
                    [1] ])
    N = 100
    e = 0.001
    B0=np.eye(2)
    ans=DFP(x0 , B0 , N ,e)
    print(ans)