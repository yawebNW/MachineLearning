# ==================== 导入库 ====================
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 数据可视化库


# ==================== 函数定义优化 ====================
def compute_predict(x,w,b):

    y_hat = np.pow(x,2) * w+b
    return y_hat
def compute_cost(x, y, w, b):
    """
    计算均方误差代价函数
    参数:
    x -- 特征矩阵 (m x d)
    y -- 真实标签向量 (m,)
    w -- 权重向量 (d,)
    b -- 偏置标量

    返回:
    cost -- 均方误差值
    """
    error = compute_predict(x,w,b) - y
    return np.sum(error ** 2) / (2 * len(y))  # MSE公式

def compute_gradient(x, y, w, b):
    """
    计算梯度（完全向量化实现）
    参数同上
    返回:
    dj_dw -- 权重梯度 (d,)
    dj_db -- 偏置梯度 (1,)
    """
    m = len(y)
    error = compute_predict(x,w,b) - y  # 复用误差计算
    dj_dw = np.sum(error * (x ** 2)) / m
    dj_db = np.sum(error) / m  # 偏置梯度：误差求和 / 样本数
    return dj_dw, dj_db

# ==================== 参数设置 ====================
dataSize = 100  # 生成的样本数量
alpha = 0.3  # 学习率（参数更新步长）
np.random.seed(10)  # 设置随机种子保证结果可复现
epsilon = 0.00001  # 提前终止训练的阈值（代价函数变化小于该值时停止）

# ==================== 数据生成优化 ====================
# 生成特征矩阵 (10000xd)
x = np.random.normal(10,10,dataSize)

# 生成真实模型参数
true_slopes = np.random.randint(1,10)
true_bias = 10  # 真实偏置项（常数项）

y = (x ** 2) * true_slopes + true_bias + np.random.normal(0,100,dataSize)
X = x ** 2

print(f"真实权重: {true_slopes}, 真实偏置: {true_bias}")

# ==================== 模型初始化 ====================
w = 0.0  # 初始化权重向量为0
b = 0.0  # 初始化偏置项为0
max_iters = 10000  # 最大迭代次数（安全限制）

x_mean = np.mean(x)
x_std = np.std(x)
x_nor = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_nor = (y - y_mean) / y_std

# ==================== 训练过程 ====================
# 初始化历史记录（预分配内存提升性能）
cost_history = np.zeros(max_iters)  # 代价函数历史
w_history = np.zeros(max_iters)  # 权重更新历史
b_history = np.zeros(max_iters)  # 偏置更新历史

# 记录初始状态
cost_history[0] = compute_cost(x_nor, y, w, b)

# 梯度下降主循环
for i in range(1, max_iters):
    # 计算梯度
    dj_dw, dj_db = compute_gradient(x_nor, y, w, b)

    # 同步更新参数
    w -= alpha * dj_dw  # 更新权重
    b -= alpha * dj_db  # 更新偏置

    # 记录历史状态
    cost_history[i] = compute_cost(x_nor, y, w, b)
    w_history[i] = w
    b_history[i] = b

    # 提前终止条件：当代价变化小于阈值时停止
    if np.abs(cost_history[i - 1] - cost_history[i]) < epsilon:
        max_iters = i  # 更新实际迭代次数
        break

# 截断历史记录数组
cost_history = cost_history[0:max_iters]
w_history = w_history[0:max_iters]
b_history = b_history[0:max_iters]

# ==================== 结果输出 ====================
print("\n训练结果:")
print(f"最优权重: {w/np.pow(x_std,2)}")  # 输出反归一化后的权重
print(f"最优偏置: {b - w * np.pow(x_mean / x_std,2)}")  # 反归一化后的偏置
print(f"最终代价: {cost_history[-1]:.2f}")

# ==================== 可视化增强 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建画布（2x2布局）
plt.figure(figsize=(12, 8))

# 子图1：代价函数下降曲线
plt.subplot(2, 2, 1)
plt.plot(cost_history, 'b', lw=1)
plt.title('代价函数下降曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价值')
plt.grid(True)

# 子图2：权重更新轨迹
plt.subplot(2, 2, 2)
plt.plot(w_history,label = 'w')
plt.title('权重更新轨迹')
plt.xlabel('迭代次数')
plt.ylabel('权重值')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# 子图3：偏置更新轨迹
plt.subplot(2, 2, 3)
plt.plot(b_history, 'r', lw=1)
plt.title('偏置更新轨迹')
plt.xlabel('迭代次数')
plt.ylabel('偏置值')
plt.grid(True)

# 子图4：实际值 vs 预测值散点图
plt.subplot(2, 2, 4)
x_axis = np.arange(np.min(x),np.max(x),1)
y_pred = compute_predict(x_axis,w/np.pow(x_std,2),b - w * np.pow(x_mean / x_std,2))
plt.scatter(x,y,c = 'b',label = '准确值')
plt.plot(x_axis,y_pred,c = 'r',label = '预估值')
plt.title('实际值 vs 预测值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)

# 调整布局并显示
plt.tight_layout()
plt.show()