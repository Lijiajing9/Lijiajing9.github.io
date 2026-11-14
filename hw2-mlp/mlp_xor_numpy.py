# -*- coding: utf-8 -*-
"""
人工智能课程作业二：多层感知机（MLP）学习参数的实现与可视化
用 2-2-1 的 MLP 解决 XOR（异或）问题
纯 NumPy 手写实现：前向传播 -> 二元交叉熵损失 -> 反向传播 -> Adam 更新
网络结构：2 输入 -> 2 个 sigmoid 隐层单元 -> 1 个 sigmoid 输出
超参数与老师 PPT 一致：Adam, learning_rate=0.03, 训练 300 轮, 输出 > 0.5 判为 1
图表风格：黑白印刷风（实心点 = y=1，空心圈 = y=0），图片自动保存到 images/

在 PyCharm 中运行：右键本文件 -> Run 'mlp_xor_numpy'
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 只保存图片，不弹窗；想弹窗可删掉这一行
import matplotlib.pyplot as plt
import os

np.random.seed(0)   # 固定随机种子，保证结果可复现（种子 0 能稳定收敛到 100%）

# ---------- 全局黑白风格（宋体 + 黑白印刷感） ----------
plt.rcParams["font.family"] = "SimSun"        # 宋体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#1a1a1a"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["text.color"] = "#1a1a1a"
plt.rcParams["axes.labelcolor"] = "#1a1a1a"
plt.rcParams["xtick.color"] = "#4a4a4a"
plt.rcParams["ytick.color"] = "#4a4a4a"
plt.rcParams["font.size"] = 10.5
INK = "#111111"      # 主墨色
GRAY = "#9a9a9a"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 数据：四个 XOR 样本 ----------
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])          # 形状 (4, 2)
y = np.array([[0.0], [1.0], [1.0], [0.0]])   # 形状 (4, 1)

# ---------- 2. 参数初始化（隐层偏置初始化为 -1，与老师 PPT 一致） ----------
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
h_space_history = {}

for epoch in range(1, EPOCHS + 1):
    # ----- 前向传播：输入沿连接权重层层计算 -----
    z1 = X @ W1 + b1            # (4,2) 隐层加权输入
    h = sigmoid(z1)             # 隐层特征 (h1, h2)
    z2 = h @ W2 + b2            # (4,1) 输出加权输入
    y_pred = sigmoid(z2)        # 输出概率

    loss = bce_loss(y, y_pred)
    loss_history.append(loss)
    acc_history.append(np.mean((y_pred > 0.5).astype(float) == y))

    if epoch in snap_epochs:
        h_space_history[epoch] = h.copy()

    # ----- 反向传播：链式法则求各层权重对损失的影响 -----
    d_z2 = (y_pred - y) / len(X)             # BCE + sigmoid 的梯度恰好是 (预测-真实)
    d_W2 = h.T @ d_z2                        # 输出层权重梯度
    d_b2 = d_z2.sum(axis=0, keepdims=True)
    d_h = d_z2 @ W2.T
    d_z1 = d_h * h * (1 - h)                 # sigmoid 导数 h(1-h)
    d_W1 = X.T @ d_z1                        # 隐层权重梯度
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}

    # ----- Adam 更新：依据梯度调整权重（梯度本身不更新） -----
    t += 1
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

# 最终结果
h = sigmoid(X @ W1 + b1)
y_pred = sigmoid(h @ W2 + b2)
print("=== 训练 300 轮后的结果 ===")
for xi, yi, p in zip(X, y, y_pred):
    print(f"输入 {xi}  真实 {int(yi[0])}  预测概率 {p[0]:.4f}  判类 {int(p[0] > 0.5)}")
print(f"最终准确率: {np.mean((y_pred > 0.5) == y) * 100:.0f}%")
print(f"最终损失: {loss_history[-1]:.4f}")
print("隐层特征 (h1, h2):")
for xi, hi, yi in zip(X, h, y):
    print(f"  输入 {xi} -> 隐层 ({hi[0]:.3f}, {hi[1]:.3f})  y={int(yi[0])}")

# ---------- 4. 可视化（黑白印刷风：实心点 = y=1，空心圈 = y=0） ----------
def draw_points(ax, xs, ys, s=190, labels=None):
    for i, (x1, x2) in enumerate(xs):
        c = 1 if ys[i][0] == 1 else 0
        ax.scatter(x1, x2, s=s, marker="o",
                   facecolors=INK if c == 1 else "white",
                   edgecolors=INK, linewidths=1.6, zorder=3)
        if labels:
            ax.annotate(labels[i], (x1, x2), textcoords="offset points",
                        xytext=(11, 9), fontsize=9.5, color="#3a3a3a")

# 图 1：损失曲线
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(range(1, EPOCHS + 1), loss_history, color=INK, lw=1.8)
ax.set_xlabel("训练轮数 epoch")
ax.set_ylabel("二元交叉熵损失")
ax.set_title("图 1  损失随训练轮数的下降", fontsize=12)
ax.grid(True, linestyle=":", color="#d8d8d8")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curve.png"), dpi=160)
plt.close(fig)

# 图 2：原空间 vs 隐层特征空间
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
draw_points(axes[0], X, y, labels=[f"({int(a)},{int(b)})" for a, b in X])
axes[0].set_title("（a）原始输入空间：同类点对角交错", fontsize=11.5)
axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
axes[0].set_xlim(-0.35, 1.35); axes[0].set_ylim(-0.35, 1.35)
axes[0].grid(True, linestyle=":", color="#d8d8d8")

red_idx = 0
for yi, hi in zip(y, h):
    c = 1 if yi[0] == 1 else 0
    axes[1].scatter(hi[0], hi[1], s=190, marker="o",
                    facecolors=INK if c == 1 else "white",
                    edgecolors=INK, linewidths=1.6, zorder=3)
    if hi[0] > 0.9 and hi[1] > 0.5:
        off = (-12, 14)
    elif hi[0] > 0.9:
        off = (-80, 14) if red_idx == 0 else (-80, -24)
        red_idx += 1
    else:
        off = (12, -18)
    axes[1].annotate(f"({hi[0]:.2f}, {hi[1]:.2f})", (hi[0], hi[1]),
                     textcoords="offset points", xytext=off, fontsize=9.5, color="#3a3a3a")

hh = np.linspace(min(h[:, 0].min(), 0) - 0.05, 1.05, 50)
if abs(W2[1, 0]) > 1e-6:
    line = -(W2[0, 0] * hh + b2[0, 0]) / W2[1, 0]
    axes[1].plot(hh, line, "--", color=INK, lw=1.8, label="输出层分类线")
    axes[1].legend(frameon=False, loc="lower left", fontsize=9.5)
axes[1].set_title("（b）隐层特征空间：一条直线可分", fontsize=11.5)
axes[1].set_xlabel("h1（近似 OR）"); axes[1].set_ylabel("h2（近似 AND）")
axes[1].set_xlim(-0.1, 1.14); axes[1].set_ylim(-0.38, 1.14)
axes[1].grid(True, linestyle=":", color="#d8d8d8")
for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("实心点 = y=1　　空心圈 = y=0", fontsize=10, color="#4a4a4a", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hidden_space.png"), dpi=160)
plt.close(fig)

# 图 3：输入平面上的决策边界
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 220), np.linspace(-0.3, 1.3, 220))
grid = np.c_[xx.ravel(), yy.ravel()]
gg = sigmoid(sigmoid(grid @ W1 + b1) @ W2 + b2).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
cf = ax.contourf(xx, yy, gg, levels=24, cmap="Greys", alpha=0.75)
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label("输出概率 P(y=1)")
ax.contour(xx, yy, gg, levels=[0.5], colors=INK, linewidths=2.0)
for yi, xi in zip(y, X):
    c = 1 if yi[0] == 1 else 0
    ax.scatter(xi[0], xi[1], s=170, facecolors=INK if c == 1 else "white",
               edgecolors=INK, linewidths=1.6, zorder=3)
    ax.annotate(str(int(yi[0])), (xi[0], xi[1]), ha="center", va="center",
                fontsize=9, color="white" if c == 1 else INK,
                fontweight="bold", zorder=4)
ax.set_title("图 3  学到的决策边界（黑线 = P 0.5）", fontsize=12)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "decision_boundary.png"), dpi=160)
plt.close(fig)

# 图 4：隐层映射随训练的演化
fig, axes = plt.subplots(1, len(h_space_history), figsize=(3.9 * len(h_space_history), 4.1))
for ax, (ep, hh_) in zip(axes, sorted(h_space_history.items())):
    for yi, hi in zip(y, hh_):
        c = 1 if yi[0] == 1 else 0
        ax.scatter(hi[0], hi[1], s=150, facecolors=INK if c == 1 else "white",
                   edgecolors=INK, linewidths=1.5, zorder=3)
    ax.set_title(f"第 {ep} 轮", fontsize=10.5)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06)
    ax.grid(True, linestyle=":", color="#d8d8d8")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("图 4  隐层特征空间随训练的演化：两类点逐渐拉开", fontsize=11.5, y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hidden_evolution.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# 图 5：准确率曲线
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.step(range(1, EPOCHS + 1), acc_history, color=INK, lw=1.8, where="post")
ax.set_ylim(-0.05, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("训练轮数 epoch")
ax.set_ylabel("准确率")
ax.set_title("图 5  准确率随训练轮数的变化", fontsize=12)
ax.grid(True, linestyle=":", color="#d8d8d8")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_curve.png"), dpi=160)
plt.close(fig)

print("\n可视化图片已保存到:", OUT)
