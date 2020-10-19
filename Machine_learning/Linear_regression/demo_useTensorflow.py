'''
Author: Abel-Cat
Date: 2020-10-19 15:31:18
LastEditors: Abel-Cat
LastEditTime: 2020-10-19 23:40:45
Motto: May the force be with you.
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers , layers ,Model

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

class Linear_Model(Model):
    def __init__(self , ndim):
        super( Linear_Model , self).__init__()
        self.w = tf.Variable(
            shape=[ndim, 1], 
            initial_value=tf.random.uniform(
                [ndim,1], minval=-0.1, maxval=0.1, dtype=tf.float32)) 

    def call(self , x):
        """
        
        """
        return None

@tf.function
def train_one_step(model, xs, ys):
    with tf.GradientTape() as t:
        # 这一步已经使用了 w 接下来可以对其求导
        y_preds = model(xs)
        loss = tf.reduce_mean(tf.sqrt(1e-12+(ys-y_preds)**2))
    grads = t.gradient (loss , model.w)
    optimizer = optimizers.Adam(0.1)
    optimizer.apply_gradients([(grads , model.w)])
    return loss

@tf.function
def predict(model, xs):
    y_preds = model(xs)
    return y_preds

def evaluate(a,y):
    """
    evaluate
    """
    std = np.sqrt(np.mean( np.abs(y-a) **2 ) )
    return std


if __name__ == "__main__":
    (xs, ys) = load_data('./Machine_learning/Linear_regression/train.txt')        
    xs = np.expand_dims(xs , axis= 0)
    xs = np.insert( xs, 1 , values=np.ones([300]) ,axis =0)
    ys = np.expand_dims(ys , axis=1)
    ndim = xs.shape[0]

    model = Linear_Model(ndim=ndim)   

    for i in range(1000):
        loss = train_one_step(model, xs, ys)
        if i % 100 == 1:
            print(f'loss is {loss:.4}')
        
        
    y_preds = predict(model, xs)
    std = evaluate(ys, y_preds)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    (xs_test, ys_test) = load_data('./Machine_learning/Linear_regression/test.txt')

    y_test_preds = predict(model, xs_test)
    std = evaluate(ys_test, y_test_preds)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    # plt.plot(o_x, o_y, 'ro', markersize=3)
    # plt.plot(o_x_test, y_test_preds, 'k')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Linear Regression')
    # plt.legend(['train', 'test', 'pred'])
    # plt.show()   