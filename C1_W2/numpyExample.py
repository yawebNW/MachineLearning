import numpy as np
import time

a = np.arange(5)
b = np.arange(20.)
print(f"a={a},a.shape={a.shape},a.dtype={a.dtype},b={b},b.shape={b.shape},b.dtype={b.dtype}")

#索引
print(f"a[0] = {a[0]}")
print(f"a[-1] = {a[-1]}")

#切片
a = np.arange(0,60,1).reshape(3,4,5)
print(f"np.median(a) = {np.median(a)}")
print(f"a={a}")
print(f"a[:] = {a[:]}")
print(f"a[:,0]={a[:,0]}")
print(f"a[:,:,0]={a[:,:,0]}")
print(f"a[1,2,3]={a[1,2,3]}")
print(f"a[1][2][3] = {a[1][2][3]}")
print(f"b[1:] = {b[1:]}")
print(f"b[1:15:-3] = {b[1:15:-3]}")
print(f"b[-1:10:-3] = {b[-1:10:-3]}")
print(f"b[-1:-10:-3] = {b[-1:-10:-3]}")
print(f"b[-1:-10] = {b[-1:-10]}")
#单一向量操作
print(f"np.mean(a) = {np.mean(a)}")
print(f"np.std(a) = {np.std(a)}")
print(f"np.median(a) = {np.median(a)}")
print(f"np.min(a) = {np.min(a)}")
print(f"np.max(a) = {np.max(a)}")
print(f"np.sum(a) = {np.sum(a)}")
#点积实现
def my_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))
a = np.random.randint(1,10,5)
b = np.random.randint(1,10,5)
print(f"my_dot(a,b) = {my_dot(a,b)}")
print(f"np.dot(a,b) = {np.dot(a,b)}")