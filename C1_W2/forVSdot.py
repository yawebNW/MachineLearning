import time
import numpy as np
from matplotlib import pyplot as plt

loopNum = 20
time_for = np.zeros(loopNum)
time_dot = np.zeros(loopNum)
size = 10

for i in range(loopNum):
    x1 = np.random.randint(1, 100, size, dtype=np.int64)
    k1 = np.random.randint(1, 100, size, dtype=np.int64)
    x2 = x1.copy()  # 避免重复生成
    k2 = k1.copy()

    # For-loop计时
    t0 = time.perf_counter_ns()
    sum_loop = 0
    for j in range(size):
        sum_loop += x1[j] * k1[j]
    t1 = time.perf_counter_ns()
    time_for[i] = t1 - t0

    # Numpy计时
    t2 = time.perf_counter_ns()
    sum_np = np.dot(x2, k2)
    t3 = time.perf_counter_ns()

    time_dot[i] = t3 - t2

    size *= 2

plt.plot(range(loopNum), time_for)
plt.plot(range(loopNum), time_dot)
plt.legend(['for', 'dot'])
plt.title('time per loop')
plt.show()

fig = plt.figure()
plt.title('for')
plt.plot(range(loopNum), time_for)
plt.show()

fig2 = plt.figure()
plt.title('dot')
plt.plot(range(loopNum), time_dot)
plt.show()