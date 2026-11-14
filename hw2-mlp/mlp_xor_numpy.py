# -*- coding: utf-8 -*-
"""
人工智能课程作业二：多层感知机（MLP）学习参数的实现与可视化
用 2-2-1 的 MLP 解决 XOR（异或）问题
纯 NumPy 手写实现：前向传播 -> 二元交叉熵损失 -> 反向传播 -> Adam 更新
网络结构：2 输入 -> 2 个 sigmoid 隐层单元 -> 1 个 sigmoid 输出
超参数与老师 PPT 一致：Adam, learning_rate=0.03, 训练 300 轮, 输出 > 0.5 判为 1
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(0)   # 固定随机种子，保证结果可复现（种子 0 能稳定收敛到 100%）

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 数据：四个 XOR 样本 ----------
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])          # 形状 (4, 2)
y = np.array([[0.0], [1.0], [1.0], [0.0]])   # 形状 (4, 1)

# ---------- 2. 参数初始化（与老师 PPT 一致：隐层偏置 -1） ----------
n_in, n_hidden, n_out = 2, 2, 1
W1 = np.random.randn(n_in, n_hidden) * 1.0    # 输入->隐层权重
b1 = np.full((1, n_hidden), -1.0)             # 隐层偏置初始化为 -1
W2 = np.random.randn(n_hidden, n_out) * 1.0   # 隐层->输出权重
b2 = np.zeros((1, n_out))                     # 输出偏置初始化为 0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bce_loss(y_true, y_pred):
    eps = 1e-12
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# ---------- 3. Adam 优化器状态 ----------
lr = 0.03
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
t = 0

EPOCHS = 300
loss_history = []
acc_history = []
snap_epochs = {1, 5, 20, 100, 300}     # 记录这些轮次的隐层映射，用于可视化
h_space_history = {}                    # epoch -> 隐层输出 (4,2)
weight_history = []                     # 每 10 轮记录 W1 的两列（两棵"判别线"方向）

for epoch in range(1, EPOCHS + 1):
    # ----- 前向传播 -----
    z1 = X @ W1 + b1            # (4,2)
    h = sigmoid(z1)             # 隐层特征 (h1, h2)
    z2 = h @ W2 + b2            # (4,1)
    y_pred = sigmoid(z2)        # 输出概率

    loss = bce_loss(y, y_pred)
    loss_history.append(loss)
    acc = np.mean((y_pred > 0.5).astype(float) == y)
    acc_history.append(acc)

    if epoch in snap_epochs:
        h_space_history[epoch] = h.copy()

    if epoch % 10 == 0:
        weight_history.append((epoch, W1.copy(), W2.copy()))

    # ----- 反向传播（二元交叉熵 + sigmoid => 梯度形式极简） -----
    d_z2 = (y_pred - y) / len(X)            # (4,1)  BCE+Sigmoid 的梯度
    d_W2 = h.T @ d_z2                        # (2,1)
    d_b2 = d_z2.sum(axis=0, keepdims=True)   # (1,1)
    d_h = d_z2 @ W2.T                        # (4,2)
    d_z1 = d_h * h * (1 - h)                 # sigmoid 的导数 h(1-h)
    d_W1 = X.T @ d_z1                        # (2,2)
    d_b1 = d_z1.sum(axis=0, keepdims=True)   # (1,2)

    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}

    # ----- Adam 更新（依据梯度调整权重，梯度本身不更新） -----
    t += 1
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

# 最终结果
z1 = X @ W1 + b1; h = sigmoid(z1)
y_pred = sigmoid(h @ W2 + b2)
print("=== 训练 300 轮后的结果 ===")
for xi, yi, p in zip(X, y, y_pred):
    print(f"输入 {xi}  真实 {int(yi[0])}  预测概率 {p[0]:.4f}  判类 {int(p[0] > 0.5)}")
print(f"最终准确率: {np.mean((y_pred > 0.5) == y) * 100:.0f}%")
print(f"最终损失: {loss_history[-1]:.4f}")
print("隐层特征 (h1, h2):")
for xi, hi, yi in zip(X, h, y):
    print(f"  输入 {xi} -> 隐层 ({hi[0]:.3f}, {hi[1]:.3f})  y={int(yi[0])}")

# ---------- 4. 可视化 ----------
C0, C1 = "#e6a817", "#c0392b"   # 黄色=0类，红色=1类（与PPT配色一致）

# 图 1：损失曲线
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(range(1, EPOCHS + 1), loss_history, color="#1a7f5a", lw=2)
ax.set_xlabel("训练轮数 (epoch)")
ax.set_ylabel("二元交叉熵损失")
ax.set_title("训练 300 轮的损失下降曲线（Adam, lr=0.03）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curve.png"), dpi=150)
plt.close(fig)

# 图 2：原空间 vs 隐层特征空间（XOR 变得线性可分）
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for yi, xi in zip(y, X):
    axes[0].scatter(xi[0], xi[1], c=C1 if yi[0] == 1 else C0, s=260,
                    edgecolors="k", zorder=3)
    axes[0].annotate(f"{int(xi[0])}{int(xi[1])} · y={int(yi[0])}",
                     (xi[0], xi[1]), textcoords="offset points",
                     xytext=(8, 8), fontsize=10)
axes[0].set_title("原始输入空间 (x1, x2)：同类点交错，一条直线分不开")
axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
axes[0].set_xlim(-0.35, 1.35); axes[0].set_ylim(-0.35, 1.35)
axes[0].grid(alpha=0.3)

red_idx = 0
for yi, hi in zip(y, h):
    axes[1].scatter(hi[0], hi[1], c=C1 if yi[0] == 1 else C0, s=260,
                    edgecolors="k", zorder=3)
    # 错开标签位置，避免重叠
    if hi[0] > 0.9 and hi[1] > 0.5:
        off = (-10, 12)
    elif hi[0] > 0.9:
        # 01 和 10 的隐层坐标几乎重合，一个标在上方、一个标在下方
        off = (-78, 16) if red_idx == 0 else (-78, -24)
        red_idx += 1
    else:
        off = (10, -16)
    axes[1].annotate(f"({hi[0]:.2f}, {hi[1]:.2f})", (hi[0], hi[1]),
                     textcoords="offset points", xytext=off, fontsize=10)
# 训练出的输出层判别线：W2[0]*h1 + W2[1]*h2 + b2 = 0
hh = np.linspace(min(h[:, 0].min(), 0) - 0.05, 1.05, 50)
if abs(W2[1, 0]) > 1e-6:
    line = -(W2[0, 0] * hh + b2[0, 0]) / W2[1, 0]
    axes[1].plot(hh, line, "--", color="#1a7f5a", lw=2, label="输出层分类线")
    axes[1].legend()
axes[1].set_title("隐层特征空间 (h1, h2)：一条直线可分")
axes[1].set_xlabel("h1（近似 OR）"); axes[1].set_ylabel("h2（近似 AND）")
axes[1].set_xlim(-0.1, 1.12); axes[1].set_ylim(-0.35, 1.12)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hidden_space.png"), dpi=150)
plt.close(fig)

# 图 3：输入平面上的决策边界（背景色 = 网格预测概率）
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200), np.linspace(-0.3, 1.3, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
gg = sigmoid(sigmoid(grid @ W1 + b1) @ W2 + b2).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6.6, 5.4))
cf = ax.contourf(xx, yy, gg, levels=30, cmap="RdYlGn_r", alpha=0.85)
fig.colorbar(cf, ax=ax, label="输出概率 P(y=1)")
ax.contour(xx, yy, gg, levels=[0.5], colors="k", linewidths=2)
for yi, xi in zip(y, X):
    ax.scatter(xi[0], xi[1], c="white", s=280, edgecolors="k", zorder=3)
    ax.annotate(str(int(yi[0])), (xi[0], xi[1]), ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=4)
ax.set_title("MLP 学到的决策边界（黑线 = P=0.5）")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "decision_boundary.png"), dpi=150)
plt.close(fig)

# 图 4：隐层映射随训练的演化
fig, axes = plt.subplots(1, len(h_space_history), figsize=(4.2 * len(h_space_history), 4.4))
for ax, (ep, hh_) in zip(axes, sorted(h_space_history.items())):
    for yi, hi in zip(y, hh_):
        ax.scatter(hi[0], hi[1], c=C1 if yi[0] == 1 else C0, s=200, edgecolors="k")
    ax.set_title(f"第 {ep} 轮")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
fig.suptitle("隐层特征空间 (h1, h2) 随训练的演化：两类点逐渐拉开", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hidden_evolution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 图 5：准确率曲线
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(range(1, EPOCHS + 1), acc_history, color="#c0392b", lw=2)
ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("训练轮数 (epoch)")
ax.set_ylabel("准确率")
ax.set_title("训练过程中的准确率变化")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_curve.png"), dpi=150)
plt.close(fig)

print("\n可视化图片已保存到:", OUT)
