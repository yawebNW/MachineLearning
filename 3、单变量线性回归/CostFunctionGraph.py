# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
# 需要切换图形后端
import matplotlib
matplotlib.use('Qt5Agg')  # 使用PyQt5后端（需提前安装）

# 生成模拟数据
np.random.seed(42)
# x从1到99的等差数列
x = np.arange(1, 100, 1)
# 生成随机斜率，范围在2到3之间，形状与x一致
slopes = np.random.uniform(2, 3, x.shape)
# 生成随机截距，范围在-50到50的整数，形状与x一致
biases = np.random.randint(-50, 51, size=x.shape)
# 生成带噪声的线性数据 y = wx + b
y = x * slopes + biases

# 定义参数搜索空间
# 生成w的候选值：-1到7之间均匀分布的81个点
w_set = np.arange(0, 5, 0.1)
# 生成b的候选值：-40到40之间的整数，间隔为1
b_set = np.arange(10, 60, 1)


# 定义线性预测函数
def predict(w, b, x):
    return w * x + b


# 定义均方误差成本函数
def cost(y, y_hat):
    return np.sum((y - y_hat) ** 2) / (2 * len(y))


# 生成参数网格
# 用meshgrid生成w和b的二维网格坐标矩阵
w_grid, b_grid = np.meshgrid(w_set, b_set)
print(f"w_grid.shape = {w_grid.shape}")  # 显示网格形状（81w x 80b）
print(f"b_grid.shape = {b_grid.shape}")

# 扩展x的维度用于广播计算（1行n列）
x_expanded = x.reshape(1, -1)
# 计算所有(w,b)组合的预测值，利用广播生成三维数组
# 维度说明：[b数量, w数量, 数据点数量]
y_hat = w_grid[:, :, np.newaxis] * x_expanded + b_grid[:, :, np.newaxis]
print(f"y_hat.shape = {y_hat.shape}")  # (80, 81, 99)

# 计算预测误差（每个数据点的误差）
errors = y_hat - y
print(f"errors.shape = {errors.shape}")  # 保持三维结构

# 计算每个(w,b)组合的总成本
# 沿第三个轴（数据点维度）求和，得到二维成本矩阵
costs = np.sum(errors  **  2, axis = 2) / (2 * len(y))
print(f"costs.shape = {costs.shape}")  # (80, 81)

# 寻找最小成本对应的参数
minInx = np.argmin(costs)  # 找到展平后的最小索引
best_cost = costs.ravel()[minInx]  # 最小成本值
best_w = w_grid.ravel()[minInx]  # 最佳权重w
best_b = b_grid.ravel()[minInx]  # 最佳截距b

# 绘制数据点与最佳拟合线
plt.plot(x, y, marker='o', label='Actual Values')  # 原始数据点
best_y_hat = predict(best_w, best_b, x)
plt.plot(x, best_y_hat, marker='o',
         label=f'Predicted Values,b={best_b},w={best_w}',
         color='r')  # 最佳拟合线
plt.legend()
plt.show()

# 绘制三维成本函数曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制表面图，步长为1保证平滑，颜色映射为jet
ax.plot_surface(w_grid, b_grid, costs,
                rstride=1, cstride=1,
                cmap='jet')
ax.set_xlabel('Weight w')  # 坐标轴标签
ax.set_ylabel('Bias b')
ax.set_zlabel('Cost')
plt.show()