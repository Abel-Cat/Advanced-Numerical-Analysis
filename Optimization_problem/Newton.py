'''
Author: Abel-Cat
Date: 2020-10-08 16:04:40
LastEditors: Abel-Cat
LastEditTime: 2020-10-08 17:03:49
Motto: May the force be with you.
'''

import numpy as np

def F(x):
    """
    待优化的函数
    """
    y = 100*np.power( x[1][0]-np.power(x[0][0],2) , 2 ) + np.power(1-x[0][0],2)
    return y

def Gradient(x):
    """
    梯度函数
    """
    y = np.array([ 
                   [-400*( x[1][0]-np.power(x[0][0],2) )*x[0][0] -2+2*x[0][0] ],
                   [ 200*x[1][0] - 200*np.power( x[0][0] ,2 ) ] 
    ])
    return y

def Hesse(x):
    """
    海塞矩阵函数
    """
    y = np.array([
                    [ 1200*np.power(x[0][0],2) - 400*x[1][0]+2 ,-400*x[0][0] ],
                    [ -400*x[0][0] , 200]
    ])
    return y

def Newton( x0, N ,e):
    """
    docstring
    """
    k=0
    xk=x0
    err=np.linalg.norm(Gradient(xk))
    while k<N and err>e :
        dk = np.linalg.solve( Hesse(xk) , -Gradient(xk))
        xk = xk +dk
        err = np.linalg.norm(Gradient(xk))
        k=k+1

    return xk

if __name__ == "__main__":
    x0 = np.array([
        [-1.2],
        [1]
    ]) 
    e=0.001
    N=20
    xk = Newton( x0 , N ,e)
    print("最优化问题的近似解：{}".format(xk))
    print("在最优点待优化函数的取值: {}".format( F(xk)) )