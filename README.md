# 简单系统的智能涌现 / Emergence of Intelligence in Simple Systems

> **仓库命题**
>
> 任何简单系统，只要规模足够大且内部存在足够丰富的相互作用，就有可能涌现出智能。
>
> *Any sufficiently large simple system with sufficiently rich internal
> interactions may give rise to intelligence.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-blue.svg)](docs/)

---

## 为什么要有这个仓库

上面那句话**作为文字是诱人的，作为科学命题是不合格的**——因为"有可能"无法被证伪。
它不能被证明错，因此也不能被证明对；它解释一切，因此不解释任何事。

这个仓库的唯一目的是**把这句话拆成可以被实验打死的若干条**，然后一条一条去打。
我们欢迎的是**否证**，不是又一个动人的隐喻。

### 拆解后的六条判据（本仓库的操作化定义）

一个系统要成为"可能涌现智能的简单系统"，必须同时满足：

| # | 判据 | 说明 | 不满足的后果 |
|---|---|---|---|
| 1 | **简单** | 原语可 O(N) 或 O(N log N) 计算，参数少或结构固定 | 不可规模化 |
| 2 | **可扩展** | 位置数 N × 通道数 d × 深度 L 能一起放大 | 停留在玩具 |
| 3 | **相互作用丰富** | 每步能做长程 / 全对全混合 | 退化为逐位置分类器 |
| 4 | **含非线性** | 能生成输入谱中不存在的新分量 | **纯线性算子堆叠会数学坍缩**（见下） |
| 5 | **良态可训练** | 能恒等初始化，梯度不炸，lr 窗口宽 | 学不到东西（见实验一） |
| 6 | **任务真的需要长程** | 数据的解必须依赖长程结构 | 模型会自发退回局部近似（见实验一） |

第 3、4 条是**理论约束**，第 5、6 条是**我们第一个实验用数据打出来的经验约束**。
第 6 条最容易被忽略，也是本仓库认为最值得强调的一点：
**算子空间里有长程通路 ≠ 模型会用长程通路。**

关于第 4 条的数学事实（这也是"纯傅里叶系统不可能涌现"的原因）：
归一化 DFT 满足 `F² = 翻转`、`F⁴ = I`，周期为 4 —— 纯傅里叶算子的任意复合都退化为
一次线性变换，**没有深度可言**，且输出谱 ⊆ 输入谱，不可能产生新结构。

---

## 实验一：把命题压缩成可证伪的形式（已完成）

**问题**：FNO 有通用逼近定理背书，那能不能只用傅里叶变换当骨架、在 CPU 上、
不借助现代深度学习那一套（无 autograd / 无 Adam / 无 GPU），做出 BERT 的效果？

**做法**：三个 arm **除 token 混合层外完全相同**，训练 BERT 的真实目标（15% 掩码 MLM）。
全部梯度手推并用中心差分校验（全局相对误差 ~1e-7）。

| arm | 混合层 | 活跃混合参数 |
|---|---|---|
| `fnet` | `y = Re(FFT(x))`，零参数 | 0 |
| `spectral` | `y = Re(IFFT(H ⊙ FFT(x)))`，可学习复数滤波器 | 28,224 |
| `attn` | 手写 4 头自注意力 | 111,744 |

**结果（等步数 3000 步）**：

| arm | 配置 | val 交叉熵 | 准确率 |
|---|---|---|---|
| spectral | lr1.2 | **1.398–1.486** | 0.578–0.588 |
| attn | lr0.2 / 输出零初始化 | 1.515–1.530 | 0.547–0.552 |
| fnet | lr1.2 | 2.478 | 0.313 |
| attn | lr0.8（短程最优） | **3.031–3.456（崩溃）** | 0.123–0.240 |
| —— | 双向 ±2 的 4-gram 计数模型 | **1.200** | **0.725** |

**四条结论**：

1. **可行性成立**：纯 NumPy + 手推梯度 + SGD + CPU 能训起傅里叶谱混合的 MLM，
   稳定超过所有 n-gram 参照线。
2. **稳健性上谱算子明显优于注意力**：spectral 在 lr 0.05→1.2 全程稳定；
   注意力是**双峰**的——短程调优出的 lr 会让它在长程崩塌（两个种子都崩）。
   关键变量是**初始化/条件数**，不是表达能力。
3. **可学习性才是分水岭**：冻结的 FNet 式零参数混合（2.25）远差于可学习滤波器（1.23）。
4. **能力上限很低，且没有涌现**：6000 步的 spectral（1.230 / 0.635）**仍不及**一个双向 ±2
   计数模型（1.200 / 0.725）。而且直接读取学到的卷积核发现：lag0 占 **93%**、
   ±1 占 5.3%、长程 >8 仅 **0.6%** —— **模型自发收敛到了局部模板**，
   完全没有使用参数化所允许的全局混合。

> 这正是判据 6 的实证版本：**任务不要求长程，再丰富的算子也会塌成局部。**
> 1.1MB 语料的字级统计只需要 ±2 窗口，模型就只买了 ±2 窗口的账。

完整数据、长度外推失败记录、混合核逐通道分析见 [`docs/03-experiment-report.md`](docs/03-experiment-report.md)。

---

## 实验二：把实验一放大 100×（已完成）

**问题**：实验一的结论是在 1.1 MB 语料、0.45 M 参数上得到的。把规模放大 100 倍之后，
判据 2（可扩展）能不能撑住？"模型自发塌成局部模板"（判据 6）会不会被规模推翻？

**做法**：语料换成 **100 MB enwik8**（byte 级，词表 206），训练量放大到每格 **41 M token**，
d/L/T 从 96/3/96 放大到 **384/6/512**（spectral 臂 9.54 M 参数）。新增一个
**严格局部**的 `stencil` 臂（循环卷积，|lag| ≤ 2，仅 11,520 个混合参数），
它同时是谱算子的严格局部子集——于是"局部核 vs 全局核"第一次成为受控比较。
为在不放弃"不用 autograd/GPU"这条约束的前提下放大，写了一个 PyTorch 移植版，
并用实验一自己的代码做逐参数梯度等价性检验（`global_rel` ≤ 2.2e-14）。

**结果**：

| 规模（enwik8, 41 M token） | stencil | spectral | attn | fnet |
|---|---|---|---|---|
| s1 (d96,L3,T128) | **1.0871 / 0.684** | 1.2243 / 0.652 | 3.1802 / 0.203（崩溃） | 2.3025 / 0.384 |
| s2 (d192,L4,T256) | 0.9958 / 0.723 | 1.1439 / 0.676 | **0.9395 / 0.736** | 2.2326 / 0.407 |
| s3 (d384,L6,T512) | **0.8263 / 0.768** | 1.0914 / 0.688 | 3.5 → 2.217 → 3.351（双峰崩溃） | 2.2864 / 0.375 |
| s3, attn lr0.2 对照 | — | — | **0.9218 / 0.738** | — |

**四条结论**：

1. **局部碾压全局，而且规模越大差距越大**：严格局部的 stencil 在每一个规模上都打败
   参数多 100 倍的全局谱算子（s3：0.8263 vs 1.0914，混合参数 11,520 vs 1,184,256）。
   这是开放问题 **Q2** 的强化版答案——不只是追平，是碾压。
2. **"谱算子比注意力稳健"在放大后反转，而且非单调**：s1 崩、s2 稳、s3（lr0.8）再崩，
   s3 只要把 lr 降到 0.2 就稳定在 0.9218。**双峰崩溃不是注意力的固有属性，而是
   "lr 没跟着规模调"的症状** —— 这是 **Q7** 的答案。
3. **核随规模变得更局部**：s1 lag0=0.79 → s3 lag0=0.99，峰值 lag 恒为 1，lag>8 < 0.2%。
   参数化允许 512 长度的全局 all-to-all 卷积，模型买的仍然是 ±1…±2 的窗口。
4. **判据 6 被一个干净的实验证实（本仓库最重要的一条）**：把任务要求的 lag 从 4 换成 64，
   谱核三层**全部精确长到 lag 64**（acc 84.8%，掩码上限 ≈85.5%），注意力在 lr=0.4 时
   达到 85.2%（第 0 层调成近似 one-hot 的"回看 64"头），而**严格局部的 stencil 结构性失败**。
   **决定"用不用长程"的是任务，不是算子。**

另外发现**实验一的手推反向传播里有一个真实的 bug**：LayerNorm 增益**前**的激活被当成
增益**后**的激活去算 MLP 第一层权重梯度；实验一的 `--gradcheck` 只在初始化点跑有限差分，
而该点 `g=1, b=0` 使 `bb ≡ xh`，bug 恰好不可见。四重独立证据（含中心差分：误差 4.1e-08
对差分噪声 8.0e-11）钉死了它。在放大尺度上该 bug 的影响（0.018 nats）**小于跑次噪声
（0.06 nats）**，所以它是方法论教训，不是会毁掉实验一结论的缺陷。

> **总判断没有变，而且更硬了**：把规模放大 100 倍，买到的是**更好的局部模板，不是涌现**。
> 谱算子直到最大的 s3（1.0914）**仍然没有越过**"看得见 ±2 邻居的计数表"（1.0666）。

完整数据、核分析、长度外推、跑次噪声分析见
[`docs/06-experiment-2-scaleup-report.md`](docs/06-experiment-2-scaleup-report.md)。

---

## 实验三：把果蝇全脑连接组当成算子（已完成）

**问题**：实验一、二用的都是**人造**的混合算子（傅里叶谱滤波、注意力、stencil）。
把一只果蝇的**真实接线**当作算子，能学会 4-gram 吗？它比随机接线更好吗？

**做法**：取 [FlyWire](https://zenodo.org/records/10676866) 全脑连接组 v783
（16,847,997 行、54,492,922 个突触、含神经递质预测），抽出突触总量最大的
**top-1000 神经元核心**（53,362 条边、54% 抑制性）当作一个**冻结的储层**：
`h_t = tanh(W(leak·h_{t-1} + (1−leak)·u_t))`，只有读出层被训练。四条配对对照：
度序列保持的双重边交换、拓扑不变但权重打乱、每个神经元入权重总和精确不变、以及只匹配
节点数与边数的 Erdős–Rényi 图。全部归一到同一谱半径。

**结果**（TinyShakespeare 字级下一 token 预测，3 种子，N=1000，MLP 读出）：

| 接线 | val loss | acc |
|---|---|---|
| **`er`（纯随机）** | **2.0279 ± 0.0263** | 0.4358 |
| `row_weight_shuffle` | **2.0271 ± 0.0144** | **0.4406** |
| `fly`（果蝇） | 2.0720 ± 0.0122 | 0.4254 |
| `degree_swap` | 2.0748 ± 0.0132 | 0.4217 |
| `shuffle_weights` | 2.0795 ± 0.0173 | 0.4126 |
| —— 3-gram markov / 4-gram markov | 2.0740 / **1.8285** | 0.3842 / 0.4648 |

**三条结论**：

1. **学不会 4-gram。** 最好的结果 2.0271 离 4-gram 的 1.8285 还差 **0.20 nats**，
   落在 **3-gram（2.0740）**水平。
2. **果蝇接线不比随机接线好。** 两种读出下 `er` 都优于果蝇；MLP 读出下差距 **0.044**，
   超过种子噪声 0.019。果蝇只赢了两条**破坏权重分配**的对照。
   这与独立先例 [flybook-git/flm](https://github.com/flybook-git/flm) 同向
   （其 README 自述配对对照"略好"，未证明果蝇解剖结构有优势）。
3. **机制是记忆长度。** 用同一线性探针测"能否从 h_t 解码 x_{t−k}"：
   **lag-3 果蝇 0.515 vs 随机 0.777**，lag-6 两者都掉回 unigram。
   4-gram 需要 lag 1–3，而果蝇恰好在最要紧的那一档先塌。

> **判据 6 现在有了第三种独立证据。** 实验一/二证明"人造的丰富算子不会自己买到长程"；
> 实验三证明**生物真实接线也不会**——至少在以"top-1000 hub 核心 + 固定随机输入投影 +
> 训练读出"这种方式接入时不会。**参数化里有长程通路 ≠ 系统真的用了长程。**

顺带产出：谱半径必须用 `rho(W)` 而非 `rho(|W|)`（后者会把动力学压进线性区，
实测 51% 神经元变死单元）；记忆与非线性是**权衡**而非越大越好；
连接表是 `(pre,post,neuropil)` 三元组，必须先合并突触数再取 log。

完整报告、机制诊断与诚实局限见
[`docs/07-experiment-3-fly-connectome.md`](docs/07-experiment-3-fly-connectome.md)。

---

## 仓库结构

```
docs/                                讨论与实验文档
  00-hypothesis.md                   命题拆解：六条判据与可证伪化
  01-literature-verification.md      文献核实记录（含被证伪/需修正的措辞）
  02-operator-taxonomy.md            候选算子族清单（矩阵/图/傅里叶/……共八族）
  03-experiment-report.md            实验一完整报告
  04-experiment-design.md            实验一设计文档（架构、约束、复现）
  05-findings-and-open-questions.md  反直觉发现与下一步实验清单
  06-experiment-2-scaleup-report.md  实验二完整报告（100× 放大 + Q1/Q2/Q3/Q7）
  07-experiment-3-fly-connectome.md  实验三完整报告（果蝇连接组当储层）
experiments/
  spectral-mlm-cpu/                  实验一代码（纯 NumPy，无 autograd）
    spectral_bert.py                 模型 + 手推反向传播 + SGD + MLM 训练
    debug_grad.py                    按参数组定位梯度错误的诊断脚本
    aggregate.py / plot.py           汇总与出图
    results/                         原始曲线、外推、cloze、对比图
  spectral-mlm-scaleup/              实验二代码（PyTorch GPU 移植 + 100× 放大）
    scripts/spectral_lm_torch.py     移植版：逐项保持实验一的骨架/目标/优化器
    scripts/run_stage*.py            Stage A 忠实复现 / Stage C ladder / Stage D 长程任务
    scripts/run_equiv.py             与实验一的逐参数梯度等价性检验
    scripts/check_ln_bug.py          用中心差分钉死实验一的 W1 梯度 bug
    scripts/check_ln_bug_independent.py  该 bug 的独立复核（不同方法 + 机器判据）
    scripts/verify_claims.py         从原始 run 文件复算报告里的 78 个数字
    runs/ logs/ figures/ ref/        原始运行数据、stdout、图、等价性参考 dump
  fly-connectome-reservoir/          实验三代码（连接组当冻结储层）
    scripts/graphs.py                FlyWire 解析、子图抽取、四类对照接线
    scripts/reservoir.py             储层扫描 + 线性/MLP 读出
    scripts/run_experiment.py        五条接线 × 两种读出 × 多种子
    scripts/probe_memory.py          记忆容量探针（解释果蝇为何更差）
    scripts/verify_graphs.py         对照图不变式校验（23 项）
    scripts/test_load_flywire.py     连接组朝向/符号校验（10 项）
    scripts/verify_claims.py         复算报告里的 63 个数字
    results/ figures/                派生子图（136 KB）、结果、三张图
```

## 复现实验一

```bash
cd experiments/spectral-mlm-cpu
python3 spectral_bert.py --gradcheck          # 梯度校验，必须先全过
bash run_main.sh 0 6000                       # 三臂主实验
bash run_followup.sh                          # 稳健性对照（零初始化 / seed1 / lr0.2）
python3 aggregate.py "results/main/*.final.json"
python3 plot.py
```

## 复现实验二

需要 PyTorch（GPU）跑 Stage B/C/D；Stage A 用实验一原始 NumPy 代码。

```bash
cd experiments/spectral-mlm-scaleup
python scripts/run_stageA.py main followup     # Stage A：原始规模的忠实复现（CPU）
python scripts/run_equiv.py                    # Stage B：数值等价性 10/10 PASS
python scripts/check_ln_bug.py                 # Stage B：有限差分钉死实验一的 bug
python scripts/check_ln_bug_independent.py     # Stage B：该 bug 的独立复核
python scripts/run_stageC.py ladder            # Stage C：s1/s2/s3 × 4 臂
python scripts/run_stageD.py                   # Stage D：长程合成任务
python scripts/plot.py                         # 出图
python scripts/verify_claims.py                # 复算报告里的 78 个数字
```

## 复现实验三

需要 `torch + scipy + pandas + pyarrow + matplotlib`。**812 MB 的连接组原表不必下载**——
派生的 136 KB 子图已纳入版本控制。

```bash
cd experiments/fly-connectome-reservoir
python scripts/verify_graphs.py --N 400        # 对照图不变式 23/23
python scripts/test_load_flywire.py            # 连接组朝向/符号 10/10
python scripts/run_experiment.py --subgraph results/fly_subgraph_N1000.npz \
    --mode causal --readout both --leaks 0.97 --input_scales 10 --seed 0 \
    --out results/causal_seed0.json
python scripts/summarize.py --inputs "results/causal_seed*.json" --out results/causal_summary.json
python scripts/probe_memory.py --subgraph results/fly_subgraph_N1000.npz \
    --leaks 0.97 --input_scales 10 --out results/probe_fly.json
python scripts/plot.py --results results/causal_summary.json \
    --subgraph results/fly_subgraph_N1000.npz
python scripts/verify_claims.py                # 复算报告里的 63 个数字
```

## 如何参与

我们最欢迎的不是"这个想法很深刻"，而是**能杀死某条判据的实验设计**。
请优先看 [`docs/05-findings-and-open-questions.md`](docs/05-findings-and-open-questions.md)
里列出的开放问题——每一条都写了具体的、可执行的否证协议。

## 许可

代码 MIT（见 [`LICENSE`](LICENSE)）；`docs/` 下的文字采用 CC BY 4.0。
