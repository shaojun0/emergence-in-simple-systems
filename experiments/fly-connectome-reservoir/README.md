# 实验三 · fly-connectome-reservoir

把 **FlyWire 果蝇全脑连接组**当成一个**冻结的储层**（reservoir）接进字符级语言模型，
只训练一个读出层，问两个问题：

1. 它能不能学会 **4-gram**（对标实验一/二里的 `4-gram markov` 参照线 **1.8285 / 0.4648**）？
2. **果蝇的具体接线是否优于同规模的随机接线？**

**答案：都不能 / 不是。** 3 种子、两种读出下最好的结果是 2.0271（离 4-gram 差 0.20 nats，
落在 3-gram 的 2.0740 水平），而且**结构完全随机的 Erdős–Rényi 图反而最好**
（MLP 读出：`er` 2.0279 vs 果蝇 2.0720，差距 0.044 > 种子噪声 0.019）。
机制是**记忆长度**：lag-3 可解码性果蝇 0.515 vs 随机 0.777。

| 接线（MLP 读出，3 seeds） | val loss | acc |
|---|---|---|
| `er`（只匹配节点数/边数） | **2.0279 ± 0.0263** | 0.4358 |
| `row_weight_shuffle` | **2.0271 ± 0.0144** | **0.4406** |
| `fly`（果蝇） | 2.0720 ± 0.0122 | 0.4254 |
| `degree_swap` | 2.0748 ± 0.0132 | 0.4217 |
| `shuffle_weights` | 2.0795 ± 0.0173 | 0.4126 |

- **完整报告**：[`../../docs/07-experiment-3-fly-connectome.md`](../../docs/07-experiment-3-fly-connectome.md)
- **实验一/二**：[`../spectral-mlm-cpu/`](../spectral-mlm-cpu/) · [`../spectral-mlm-scaleup/`](../spectral-mlm-scaleup/)

> ⚠️ **先纠正一个前提**：连接组**不是**「预训练权重」。FlyWire 给的是**结构连接**
> （哪些神经元连哪些、突触数、以及神经递质预测），没有"训练好的果蝇模型"可以加载。
> 本实验把它当作**固定的循环动力系统**来用——这才是"接线"能提供的唯一东西。

## 为什么用储层

实验一/二的所有混合算子（spectral / stencil / attention）都是**位置→位置**的映射，
而且平移等变。连接组是**神经元→神经元**的图，既没有"第 t 个位置"，也不满足平移等变。
直接把邻接矩阵当成 T×T 混合矩阵会破坏平移等变性，而 4-gram 统计本身是平移不变的——
那样的失败是设计错误，不是科学结论。

储层框架绕开了这个问题：接线只负责**提供动力学**，token 通过一个固定的随机输入投影
驱动它，读出层再去解码。

```
h_t = tanh( W @ (leak * h_{t-1} + (1 - leak) * u_t) ),   u_t = U[:, tok_t]
logits_t = h_t @ Wout + b
```

`W`（连接组）和 `U`（输入投影）**全程冻结**，只训练 `Wout, b`。所以任何两条接线之间的
差异都只能归因于接线本身，不能归因于优化器。

## 五条接线（全部归一到同一谱半径）

| 名字 | 保留什么 | 破坏什么 |
|---|---|---|
| `fly` | 真实连接组 | — |
| `degree_swap` | 无权重入/出度序列、权重多重集 | 具体拓扑（双重边交换） |
| `row_weight_shuffle` | 拓扑**完全不变**、每个神经元入权重总和**精确不变** | "哪条输入有多强"的分配 |
| `shuffle_weights` | 拓扑**完全不变** | 权重的全局分配（行和不再保持） |
| `er` | 只有节点数与边数 | 其他一切 |

全部按 `rho(W) = radius` 重新缩放，使**动力学可比、只有接线不同**。
不变式由 `scripts/verify_graphs.py` 逐条机器校验（23/23 通过）。

> **注意**：缩放必须用 `rho(W)`（W 的最大特征值模），**不能**用 `rho(|W|)`。
> 带符号矩阵行归一化后行和为 1，但 `rho(|W|)` 可达 ~10——按它缩放会把动力学压进
> 线性区（实测 51% 神经元落到 |h|<0.01）。这是本项目踩到并修掉的一个真陷阱。

## 复核顺序

```bash
# 0) 环境：需要 torch + scipy + pandas + pyarrow + matplotlib
#    本机用 D:\Anaconda\envs\bert\python.exe (torch 2.0.1+cu117, py3.10)

# 1) 先验证对照图真的保持了它声称的性质（必须先全过）
python scripts/verify_graphs.py --N 400                # 23/23
python scripts/test_load_flywire.py                    # 10/10（朝向与符号）

# 2) 取子图（只需一次；产物 136 KB，已纳入版本控制，因此不必下载 812 MB）
python scripts/export_subgraph.py --connectome <feather> --N 1000

# 3) 主实验：五条接线 × 两种读出 × 3 个种子（约 20 分钟）
python scripts/run_experiment.py --subgraph results/fly_subgraph_N1000.npz \
    --mode causal --readout both --leaks 0.97 --input_scales 10 \
    --fit_positions 200000 --epochs 60 --seed 0 --out results/causal_seed0.json
python scripts/summarize.py --inputs "results/causal_seed*.json" --out results/causal_summary.json

# 4) 机制诊断：记忆容量探针（说明果蝇为什么更差）
python scripts/probe_memory.py --subgraph results/fly_subgraph_N1000.npz \
    --leaks 0.97 --input_scales 10 --out results/probe_fly.json

# 5) 出图 / 复算报告里的每个数字
python scripts/plot.py --results results/causal_summary.json --subgraph results/fly_subgraph_N1000.npz
python scripts/verify_claims.py                        # 63/63
```

## 目录

```
scripts/flydata.py        语料 / 切分 / 80-10-10 掩码，与实验一逐位对齐；参照线复用实验二的 count_baselines.py
scripts/graphs.py         FlyWire 解析、子图抽取、四类对照、谱半径归一化、子图 npz 存取
scripts/reservoir.py      储层扫描（分块并行）、线性/MLP 读出、指标
scripts/run_experiment.py 主实验（causal / mlm × fly + 对照，leak×input_scale 网格）
scripts/probe_memory.py   记忆容量探针 + 状态统计（memory-nonlinearity 权衡）
scripts/export_subgraph.py 把 812 MB 的连接表压成 136 KB 的派生 npz（可提交）
scripts/summarize.py      多种子合并成 mean±sd，并算出"果蝇 vs 对照"的差值
scripts/verify_graphs.py  对照图不变式校验（23 项）
scripts/test_load_flywire.py 连接组读取的朝向/符号校验（合成表，10 项）
scripts/verify_claims.py  从原始结果复算报告里的 63 个数字
scripts/plot.py           三张图：结果、记忆曲线、接线统计
results/                  子图 npz、每种子结果、汇总、记忆探针、合成图调参
figures/                  三张图
```

## 数据

`proofread_connections_783.feather`（812 MB，FlyWire v783，16,847,997 行 / 54,492,922 突触）：
<https://zenodo.org/records/10676866> · 用 `../spectral-mlm-scaleup/scripts/fetch.py` 下载
（**走本地代理快约 10 倍**，且它可断点续传）。

原表未纳入版本控制，但**派生的 136 KB 子图 `results/fly_subgraph_N1000.npz` 已纳入**，
所以除重新下载原表以外的全部步骤都能在仓库内复现。
