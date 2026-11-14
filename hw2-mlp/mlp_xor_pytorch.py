# -*- coding: utf-8 -*-
"""
人工智能课程作业二（PyTorch 版，对照老师 PPT 第 88 页写法）
2 输入 -> 2 个 sigmoid 隐层单元 -> 1 个 sigmoid 输出
二元交叉熵 + Adam（lr=0.03），训练 300 轮，输出 > 0.5 判为 1
"""
import torch

torch.manual_seed(4)   # 固定随机种子保证可复现（种子 4 能稳定收敛到 100%）

# 1. 数据：四个 XOR 样本（float32）
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# 2. 模型参数：直接用 Parameter 张量手写前向传播（与 PPT 一致）
W1 = torch.nn.Parameter(torch.randn(2, 2))   # 输入 -> 隐层
b1 = torch.nn.Parameter(torch.tensor([-1., -1.]))  # 隐层偏置初始化为 -1
W2 = torch.nn.Parameter(torch.randn(2, 1))   # 隐层 -> 输出
b2 = torch.nn.Parameter(torch.tensor([0.]))  # 输出偏置初始化为 0

optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=0.03)
loss_fn = torch.nn.BCELoss()   # 二元交叉熵

for epoch in range(1, 301):
    optimizer.zero_grad()                    # 清掉上一轮的梯度
    h = torch.sigmoid(X @ W1 + b1)           # 前向：隐层特征 (h1, h2)
    y_pred = torch.sigmoid(h @ W2 + b2)      # 前向：输出概率
    loss = loss_fn(y_pred, y)                # 损失：衡量当前模型的误差
    loss.backward()                          # 反向传播：求各层权重对损失的梯度
    optimizer.step()                         # Adam 依据梯度调整权重（梯度本身不更新）

    if epoch % 50 == 0:
        with torch.no_grad():
            acc = ((y_pred > 0.5).float() == y).float().mean()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc.item():.0%}")

# 最终结果
with torch.no_grad():
    h = torch.sigmoid(X @ W1 + b1)
    y_pred = torch.sigmoid(h @ W2 + b2)
    print("\n=== 训练 300 轮后的结果 ===")
    for xi, yi, p in zip(X, y, y_pred):
        print(f"输入 {xi.tolist()}  真实 {int(yi.item())}  预测概率 {p.item():.4f}  判类 {int(p.item() > 0.5)}")
    print("隐层特征 (h1, h2):")
    for xi, hi, yi in zip(X, h, y):
        print(f"  输入 {xi.tolist()} -> ({hi[0].item():.3f}, {hi[1].item():.3f})  y={int(yi.item())}")
