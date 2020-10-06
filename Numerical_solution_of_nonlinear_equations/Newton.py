'''
Author: Abel-Cat
Date: 2020-10-05 23:12:12
LastEditors: Abel-Cat
LastEditTime: 2020-10-06 22:42:07
Motto: May the force be with you.
'''

from __future__ import print_function
import numpy as np

def F(x):
    y=np.array([  np.power(x[0],2)+np.power(x[1],2)-5,
                  (x[0]+1)*x[1]-(3*x[0]+1) ])
    return y

def J(x):
    y=np.array([ [2*x[0] , 2*x[1]],
                 [x[1]-3 , x[0]+1] ])
    return y

def Newton(x0,N,e):
    k=0
    err=1
    xk=x0
    while k<N and err>e:
        dk = np.linalg.solve(J(xk), -F(xk))
        err = np.linalg.norm(dk)
        xk1 = xk+dk
        k = k+1
        xk = xk1

    return xk

if __name__ == "__main__":
    x0 = np.array([ 1,
                    1])
    N = 100
    e = 0.001
    ans=Newton(x0,N,e)
    print(ans)

    # k=0
    # err=1
    # xk=x0
    # j=J(xk)
    # f=F(xk)
    # print(j)
    # print('------------------')
    # print(f.shape)
    # while k<N and err>e:
    #     dk = np.linalg.solve(J(xk), -F(xk))
    #     print (dk)
    #     err = np.linalg.norm(dk)
    #     xk1 = xk+dk
    #     k = k+1
    #     xk = xk1

    # print(xk)


    
