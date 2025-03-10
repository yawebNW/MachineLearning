from xmlrpc.client import MAXINT

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(21)
#数据集模拟
ArraySize = 200
x = np.random.randint(1, 100, ArraySize)
biases = np.random.randint(-1, 1, ArraySize)
y = x * 2 + biases
#初始参数设定
k = 0
b = 0
alpha = 0.00002
alpha2 = 0.1
delta = 0.01
#评估值计算
def calcu_predict_value(k,b,x):
    return k * x + b
#损失函数计算
def cost_function(y_hat,y):
    return sum((y_hat - y)**2)/(2*len(y))
#梯度下降法过程
def calcu_gradient(x,y,k,b):
    m = len(x)
    y_hat = x * k + b
    dj_dk = sum((y_hat - y) * x) / m
    dj_db = sum(y_hat - y) / m
    return dj_dk, dj_db

num_iters = 1000
cost_history = np.zeros(num_iters)
k_history = np.zeros(num_iters)
b_history = np.zeros(num_iters)
for i in range(num_iters):
    dj_dk, dj_db = calcu_gradient(x,y,k,b)
    k = k - alpha * dj_dk
    b = b - alpha2 * dj_db
    k_history[i] = k
    b_history[i] = b
    y_hat = k * x + b
    cost_history[i] = cost_function(y_hat,y)

#最终结果绘图
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
x_set = np.arange(1, 100, 1)
y_hat = x_set * k + b
plt.plot(x_set,y_hat,color = 'r', label=f"f(x) = kx + b, k = {k}, b = {b}")
plt.scatter(x,y)
plt.legend()
plt.show()
#梯度下降过程绘图
fig1 = plt.figure(1)
plt.plot(range(1,num_iters+1),cost_history, label = "cost")
plt.title("成本曲线")
plt.show()
fig2 = plt.figure(2)
plt.plot(range(1,num_iters+1),k_history, label = "k")
plt.title("k曲线")
plt.show()
fig3 = plt.figure(3)
plt.plot(range(1,num_iters+1),b_history, label = "b")
plt.title("b")
plt.show()

