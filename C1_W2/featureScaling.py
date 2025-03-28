# ==================== 导入库 ====================
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 数据可视化库

# ==================== 参数设置 ====================
d = 4  # 特征维度（每个样本有4个特征）
dataSize = 10000  # 生成的样本数量
alpha = 0.1  # 学习率（参数更新步长）
np.random.seed(10)  # 设置随机种子保证结果可复现
epsilon = 0.00001  # 提前终止训练的阈值（代价函数变化小于该值时停止）

# ==================== 数据生成优化 ====================
# 生成特征矩阵 (10000x4)
x = np.zeros((dataSize, d))
for i in range(d):
    # 每个特征列的范围是10^i到10^(i+1)，模拟不同数量级的特征
    x[:, i] = np.random.randint(10 ** i, 10 ** (i + 1), dataSize)

# 生成真实模型参数
true_slopes = np.random.randint(-10, 10, d)  # 生成4个-10到10之间的整数作为真实权重
true_bias = 1000  # 真实偏置项（常数项）

# 生成目标值 y = X * true_slopes + true_bias + 高斯噪声
y = np.dot(x, true_slopes) + true_bias + np.random.normal(0, 1000, dataSize)

print(f"特征矩阵形状: {x.shape}")  # 输出 (10000, 4)
print(f"真实权重: {true_slopes}, 真实偏置: {true_bias}")

# ==================== 模型初始化 ====================
w = np.zeros(d)  # 初始化权重向量为0
b = 0.0  # 初始化偏置项为0
max_iters = 10000  # 最大迭代次数（安全限制）

# ==================== 归一化处理 ====================
x_std = np.std(x, axis=0)  # 计算每个特征的方差
x_mean = np.mean(x, axis=0)  # 计算每个特征的均值
x_nor = (x - x_mean) / x_std  # 标准化处理：(原始值-均值)/标准差


# ==================== 函数定义优化 ====================
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
    error = np.dot(x, w) + b - y  # 预测值与真实值的差
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
    error = np.dot(x, w) + b - y  # 复用误差计算
    dj_dw = np.dot(x.T, error) / m  # 权重梯度：X转置点乘误差 / 样本数
    dj_db = np.sum(error) / m  # 偏置梯度：误差求和 / 样本数
    return dj_dw, dj_db


# ==================== 训练过程 ====================
# 初始化历史记录（预分配内存提升性能）
cost_history = np.zeros(max_iters)  # 代价函数历史
w_history = np.zeros((max_iters, d))  # 权重更新历史
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
    w_history[i] = w.copy() / x_std  # 将归一化的权重转换回原始尺度
    b_history[i] = b - np.sum(w * x_mean / x_std)  # 偏置项的反归一化计算

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
print(f"最优权重: {w / x_std}")  # 输出反归一化后的权重
print(f"最优偏置: {b - np.sum(w * x_mean / x_std):.2f}")  # 反归一化后的偏置
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
for i in range(d):
    plt.plot(w_history[:, i], label=f'w[{i}]')
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
y_pred = np.dot(x_nor, w) + b
plt.scatter(y, y_pred, s=5, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)  # 绘制对角线
plt.title('实际值 vs 预测值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)

# 调整布局并显示
plt.tight_layout()
plt.show()