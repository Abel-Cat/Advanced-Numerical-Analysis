'''
Author: Abel-Cat
Date: 2020-10-15 21:17:59
LastEditors: Abel-Cat
LastEditTime: 2020-10-19 16:19:00
Motto: May the force be with you.
'''


import numpy as np
import matplotlib.pyplot as plt

def load_data( filename ):
    """
    load the data
    """
    print('当前读取数据来源于:'+filename)
    data = []
    with open( filename, 'r' ) as f:
       for line in f:
           data.append( map(float, line.strip().split() ) )
        #    strip :  删除文件每行开头与结尾的空格  
        #    split :  将数据按有空格进行切分  返回由字符串组成的列表的迭代器
       x,y = zip( *data )
    
    return np.asarray(x) ,np.asarray(y)


def LSM( x, y):
    """
     Least Square Method
    """
    # x = x.reshape(1,300)
    # x = np.insert( x, 1 , values=np.ones([300]) ,axis =0)
    # y = y.reshape(300,1)
    w = np.linalg.inv(x@x.T) @ x @y
    print (w)

    return w


def LMS( x,y, epoch , eta):
    """
     Least Mean Squares  method
    x : x   y : y  
    epoch : number of iterations
    eta : learning rate
    """
    
    # x = x.reshape(1,300)
    # x = np.insert( x, 1 , values=np.ones([300]) ,axis =0)
    # y = y.reshape(300,1)
    w = np.zeros([2])
    w = w.reshape(2,1)
    print (w)
    for i in range(epoch) :
        w = w + eta*x@ ( y-(np.transpose(x)@w) )
        print (w)
    return w

    
def evaluate(a,y):
    """
    evaluate
    """
    std = np.sqrt(np.mean( np.abs(y-a) **2 ) )
    return std
         

if __name__ == "__main__":
    filename = './Machine_learning/Linear_regression/train.txt'
    x,y = load_data(filename=filename)
    x = np.expand_dims(x,axis= 0)
    x = np.insert( x, 1 , values=np.ones([300]) ,axis =0)
    y = y.reshape(300,1)
    w = LSM(x,y)
    y_pre = np.transpose(x) @ w
    loss = evaluate( y, y_pre )
    print (loss)

    # LMS(x,y,20,0.01)