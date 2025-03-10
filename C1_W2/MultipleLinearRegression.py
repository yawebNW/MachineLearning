import numpy as np
import matplotlib.pyplot as plt

# ==================== 参数设置 ====================
d = 4  # 特征维度
dataSize = 1000  # 样本数量
alpha_w = 0.001  # 权重学习率
alpha_b = 0.4  # 偏置学习率（调整至合理范围）
np.random.seed(10)  # 固定随机种子保证可复现性
epsilon = 0.1

# ==================== 数据生成优化 ====================
# 生成特征矩阵 (1000x4)
x = np.random.randint(1, 20, (dataSize, d))

# 生成真实模型参数
true_slopes = np.random.randint(-1000, 1000, d)  # 真实权重
true_bias = 1000  # 真实偏置（固定值）

# 生成目标值 y = X * true_slopes + true_bias + 高斯噪声
y = np.dot(x, true_slopes) + true_bias + np.random.normal(0, 1000, dataSize)

print(f"特征矩阵形状: {x.shape}")
print(f"真实权重: {true_slopes}, 真实偏置: {true_bias}")

# ==================== 模型初始化 ====================
w = np.zeros(d)  # 权重初始化
b = 0.0  # 偏置初始化
max_iters = 10000  # 最大迭代次数（避免与内置函数iter重名）


# ==================== 函数定义优化 ====================
def compute_cost(x, y, w, b):
    """
    计算均方误差代价
    优化点：使用向量化计算提高效率
    """
    error = np.dot(x, w) + b - y  # 计算误差
    return np.sum(error ** 2) / (2 * len(y))  # 均方误差公式


def compute_gradient(x, y, w, b):
    """
    计算梯度（完全向量化实现）
    优化点：复用误差计算，避免重复运算
    """
    m = len(y)
    error = np.dot(x, w) + b - y  # 复用误差计算
    dj_dw = np.dot(x.T, error) / m  # 权重梯度 (4,)
    dj_db = np.sum(error) / m  # 偏置梯度 (1,)
    return dj_dw, dj_db


# ==================== 训练过程 ====================
# 初始化历史记录（预分配数组提升性能）
cost_history = np.zeros(max_iters)
w_history = np.zeros((max_iters, d))
b_history = np.zeros(max_iters)

# 初始状态记录
cost_history[0] = compute_cost(x, y, w, b)

# 梯度下降主循环
for i in range(1, max_iters):
    # 计算梯度
    dj_dw, dj_db = compute_gradient(x, y, w, b)

    # 同步更新参数
    w -= alpha_w * dj_dw
    b -= alpha_b * dj_db

    # 记录历史状态
    cost_history[i] = compute_cost(x, y, w, b)
    w_history[i] = w.copy()  # 避免引用问题
    b_history[i] = b

    #当cost差异比delta小时，提前结束学习
    if np.abs(cost_history[i - 1] - cost_history[i]) < epsilon:
        max_iters = i
        break

cost_history = cost_history[0:max_iters]
w_history = w_history[0:max_iters]
b_history = b_history[0:max_iters]
# ==================== 结果输出 ====================
print("\n训练结果:")
print(f"最优权重: {w}")
print(f"最优偏置: {b:.2f}")
print(f"最终代价: {cost_history[-1]:.2f}")

# ==================== 可视化增强 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
plt.figure(figsize=(12, 8))

# 代价函数曲线
plt.subplot(2, 2, 1)
plt.plot(cost_history, 'b', lw=1)
plt.title('代价函数下降曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价值')
plt.grid(True)

# 权重变化曲线
plt.subplot(2, 2, 2)
for i in range(d):
    plt.plot(w_history[:, i], label=f'w[{i}]')
plt.title('权重更新轨迹')
plt.xlabel('迭代次数')
plt.ylabel('权重值')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# 偏置变化曲线
plt.subplot(2, 2, 3)
plt.plot(b_history, 'r', lw=1)
plt.title('偏置更新轨迹')
plt.xlabel('迭代次数')
plt.ylabel('偏置值')
plt.grid(True)

# 实际vs预测散点图（新增诊断图）
plt.subplot(2, 2, 4)
y_pred = np.dot(x, w) + b
plt.scatter(y, y_pred, s=5, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
plt.title('实际值 vs 预测值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)

plt.tight_layout()
plt.show()
