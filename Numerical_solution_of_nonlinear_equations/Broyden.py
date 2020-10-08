'''
Author: Abel-Cat
Date: 2020-10-06 22:52:52
LastEditors: Abel-Cat
LastEditTime: 2020-10-07 22:06:02
Motto: May the force be with you.
'''
import numpy as np

def F(x):
    y=np.array([  [np.power(x[0][0],2)+np.power(x[1][0],2)-5],
                  [(x[0][0]+1)*x[1][0]-(3*x[0][0]+1)] ])
    return y

def Broyden(x0 ,A0 , N , e):
    """
    Broyden method  本质就是在Newton法中用A矩阵替代Jacobi矩阵
    """
    k=0
    err=1
    xk=x0
    Ak=A0

    while k<N and err>e:
        dk = np.linalg.solve(Ak, -F(xk))
        err = np.linalg.norm(dk)
        xk1 = xk
        xk1 = xk+dk
        yk = F(xk1)-F(xk)
        sk = xk1-xk
        # 注意: 在python numpy 中行向量乘以列向量 要用 @
        sksk =  np.transpose(sk) @ sk
        add_A = np.dot( (yk-np.dot(Ak,sk)) , np.transpose(sk) ) / sksk
        Ak=Ak+add_A
        k = k+1
        xk = xk1

    return xk

if __name__ == "__main__":
    x0 = np.array([ [1],
                    [1] ])
    N = 100
    e = 0.001
    A0=np.eye(2)
    ans=Broyden(x0 , A0 , N ,e)
    print(ans)