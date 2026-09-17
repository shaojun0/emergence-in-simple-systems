> **实验一 · 设计文档**
> 原始文件：`spectral_mlm_cpu/README.md`（架构、约束、梯度校验、复现步骤）。

---

# 用傅里叶谱混合在 CPU 上做 BERT 式掩码语言模型（纯 NumPy，无 autograd）

> **实验结果见 [`REPORT.md`](REPORT.md)**（含完整数据、对照与诚实的局限）。
> 摘要：能训起来，且比注意力在原始 SGD 下稳健得多；但本规模下能力只到
> **双向 4-gram 计数模型**的水平（1.230 vs 1.200），没有涌现 BERT 级能力。
> FNO 的通用逼近定理对此没有贡献，起作用的是"恒等初始化 + 良态参数化"。

问题：FNO 有通用逼近定理背书，那能不能**只用傅里叶变换当骨架**、在 CPU 上、
**不借助现代深度学习那一套**，做出 BERT 的效果？

## 0. 先把目标说清楚（否则就是自欺）

不可能复现 BERT 的 GLUE 分数：那需要 33 亿词预训练 + 下游微调 + 数百 TPU 日。
本目录做的是**可证伪的机制检验**，即 FNet/FNO 那个核心主张：

> 把自注意力换成（可学习的）傅里叶谱混合，语言建模能力是否还在？

为此设置三个 arm，**除了 token 混合层以外全部相同**：

| arm | 混合层 | 混合层参数量（T=96） |
|---|---|---|
| `fnet` | `y = Re(FFT(x))`，**零参数**（Lee-Thorp et al., NAACL 2022） | 0 |
| `spectral` | `y = Re(IFFT(H ⊙ FFT(x)))`，频域可学习复数滤波器（FNO / GFNet / AFNO 一族） | 28,224 |
| `attn` | 手写 4 头自注意力（对照组） | 111,744 |

## 1. 遵守的约束：什么被去掉了

- **没有** PyTorch / JAX / TensorFlow / autograd —— 所有梯度手工推导
- **没有** GPU、CUDA、混合精度
- **没有** Adam / AdamW —— 只有 SGD + momentum + 全局范数裁剪
- **没有** dropout / weight decay / EMA / LR schedule 花活（只有 warmup + 线性衰减）

全部 import 只有 `argparse, json, math, os, time, numpy`。

## 2. 架构

```
h = Embed(tokens) + sin/cos 位置编码        # 位置编码无参数，故可外推长度
for l in range(L):
    a  = LayerNorm(h)
    z  = Mix(a)                             # ← 唯一变量：fnet / spectral / attn
    h  = h + z
    h  = h + W2 · ReLU(W1 · LayerNorm(h))   # 通道混合（MLP）
logits = LayerNorm(h) · Embedᵀ              # 权重共享
loss   = 仅在 15% 被掩码位置上的交叉熵       # BERT 的 MLM 目标
```

`spectral` 的滤波器初始化为 `H=1`，即**开局等价于恒等混合**，必须自己学出结构。
`H` 只有在 `T_train` 范围内的频率被训练；为长度外推预留的高频保持初始恒等。

## 3. 手推反向传播中两个真正容易错的地方（本仓库已用数值梯度校验）

1. **FFT 的伴随算子带因子 T**。`np.fft.fft` 不含 `1/T`，`ifft` 含，所以
   `d/da = T · Re(IFFT(dz))`，漏掉 T 会让输入侧梯度整体缩小 T 倍
   （而滤波器梯度却是对的，极具迷惑性）。
2. **ReLU 的激活值与掩码不能混用**：`dW2` 需要激活值，链式回传需要 0/1 掩码。

校验方式：中心差分，判据用**全局相对误差**（绝对误差 / 全体梯度 RMS）——
单参数相对误差在梯度只有 1e-5 时会被浮点噪声淹没，没有意义。

```
$ python3 spectral_bert.py --gradcheck
[gradcheck:spectral] checked=184  |grad|rms=1.602e-02  max_abs_err=1.883e-09  global_rel=1.18e-07
[gradcheck:fnet]     checked=152  |grad|rms=2.167e-02  max_abs_err=1.590e-09  global_rel=7.34e-08
[gradcheck:attn]     checked=280  |grad|rms=1.832e-02  max_abs_err=6.033e-10  global_rel=3.29e-08
GRADCHECK: ALL PASS
```

## 4. 实验协议

- 语料：Tiny Shakespeare（1,115,394 字符，字符级，词表 66 = 65 字符 + [MASK]）
- 划分：前 95% 训练 / 后 5% 验证
- 模型：L=3, d=96, FFN=384, 4 头, T=96, batch=24
- 优化：SGD momentum 0.9，梯度裁剪 1.0，warmup 200，线性衰减到 5%
- **每个 arm 单独扫学习率**（不这么做就是用不公平的 lr 下结论）
- 指标：验证集 MLM 交叉熵 / 掩码 token 准确率、
  零样本长度外推（T=96/192/384/768）、学到的频谱形状、cloze 样例
- 零上下文参照线（必须报告，否则"loss 下降了"没有意义）：

| 预测器 | 交叉熵 | 准确率 |
|---|---|---|
| unigram（纯字频，零上下文） | 3.348 | 0.146 |
| bigram Markov | 2.536 | 0.271 |
| trigram Markov | 2.445 | 0.384 |
| 4-gram Markov | 2.425 | 0.466 |

## 5. 复现

```bash
cd spectral_mlm_cpu
python3 spectral_bert.py --gradcheck                      # 1) 校验手推反向传播
python3 spectral_bert.py --mix spectral --lr 0.8 --steps 6000 \
        --extrapolate 96 192 384 768 --outdir runs/main   # 2) 三个 arm 各跑一遍
python3 aggregate.py --md                                  # 3) 汇总成表
```

## 6. 文件

- `spectral_bert.py` —— 模型、手推反向传播、SGD、训练与评估（单文件）
- `debug_grad.py` —— 按参数组定位梯度错误的诊断脚本
- `aggregate.py` —— 汇总 `runs/*.final.json` 输出对照表与 `REPORT.md`
- `runs/` —— 每个 arm 的 `*.jsonl`（训练曲线）与 `*.final.json`（含外推、频谱、cloze）
