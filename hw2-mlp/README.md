# 作业二 · MLP 学习参数的实现与可视化

## 文件清单

- `index.html` —— 作业二展示网页（单文件、响应式，图片用相对路径）
- `mlp_xor_numpy.py` —— NumPy 手写版：前向传播 / 二元交叉熵 / 反向传播 / Adam 全部手写
- `mlp_xor_pytorch.py` —— PyTorch 版：对照老师 PPT 第 88 页的写法
- `images/` —— 训练过程生成的 5 张可视化图

## 如何跑代码

```bash
pip install numpy matplotlib        # 跑 NumPy 版
python mlp_xor_numpy.py             # 训练 300 轮并重新生成 images/ 下的图

pip install torch                   # 跑 PyTorch 版（可选）
python mlp_xor_pytorch.py
```

## 如何发布到 GitHub Pages

1. 在仓库 `Lijiajing9.github.io` 根目录新建文件夹 `hw2-mlp/`；
2. 把 `index.html`、两个 `.py` 文件和整个 `images/` 文件夹原样上传（保持相对路径不变，图片才能显示）；
3. 上传后访问 `https://lijiajing9.github.io/hw2-mlp/` 即为作业二页面；
4.（可选）在作业一页面的合适位置加一个入口链接：
   `<a href="hw2-mlp/">作业二 · MLP 的实现与可视化</a>`

## 实验设置（与老师 PPT 要求一致）

| 项目 | 设置 |
|------|------|
| 结构 | 2 输入 → 2 个 sigmoid 隐层单元 → 1 个 sigmoid 输出 |
| 数据 | 4 个 XOR 样本（00→0, 01→1, 10→1, 11→0） |
| 损失 | 二元交叉熵 BCE |
| 优化器 | Adam，学习率 0.03 |
| 轮数 | 300 轮，输出概率 > 0.5 判为 1 |
| 结果 | 准确率 100%，最终损失约 0.03~0.04 |

注意：随机种子固定过（NumPy 用 0，PyTorch 用 4）。换种子可能掉进局部最优（准确率 75%），这也是网页"踩坑记录"里分析的点。
