'''
Author: Abel-Cat
Date: 2020-10-08 22:03:40
LastEditors: Abel-Cat
LastEditTime: 2020-10-09 00:16:21
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

def BFGS(x0, B0 , N , e):
    """
    拟牛顿法本质上是用Bk近似替代Hesse矩阵，本函数使用
    BFGS公式进行Bk的迭代
    """
    xk = x0
    Bk = B0
    k = 0
    err = np.linalg.norm( Gradient(xk) )
    j = 0
    while k < N and err > e :
        dk = np.linalg.solve( Bk , -Gradient(xk) )
        Beta = 0.5
        while True :
            left = F( xk+np.power(Beta,j)*dk )
            right = F( xk ) + ( 0.1 * np.power(Beta,j) * np.transpose( Gradient(xk) ) @ dk )
            if left >= right[0] :
                j=j+1
            else :
                break
        
        alpha = np.power( Beta , j )
        xk1 = xk + alpha * dk
        sk = xk1 - xk
        yk = Gradient( xk1 ) - Gradient( xk )
        add_Bk1 = - ( (Bk @ sk @ np.transpose(sk) @ Bk) / ( np.transpose(sk) @ Bk @sk ) )
        add_Bk2 =  (yk @ np.transpose(yk) ) / (np.transpose(yk) @ sk) 
        Bk = Bk + add_Bk1 + add_Bk2
        xk = xk1
        k = k+1
        print(xk)
    return xk

if __name__ == "__main__":
    x0 = np.array([ [-1.2],
                    [1] ])
    N = 100
    e = 0.001
    B0=  np.eye(2)
    ans=BFGS( x0 , B0 , N ,e)
    print( ans )