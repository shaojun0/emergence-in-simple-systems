# 实验二 · spectral-mlm-scaleup

把实验一（`../spectral-mlm-cpu/`）**放大 100 倍**重跑，并补上实验一留下的四个开放问题
（Q1 / Q2 / Q3 / Q7）与判据 2、判据 6 的正面对撞。

- **完整报告**：[`../../docs/06-experiment-2-scaleup-report.md`](../../docs/06-experiment-2-scaleup-report.md)
- **结果汇总**：`RESULTS.md`（由 `scripts/aggregate.py` 生成）
- **实验一**：[`../spectral-mlm-cpu/`](../spectral-mlm-cpu/) · [`../../docs/03-experiment-report.md`](../../docs/03-experiment-report.md)

## 放大规格

| 项目 | 实验一 | 实验二 |
|---|---|---|
| 语料 | Tiny Shakespeare, 1.1 MB, 词表 66 | enwik8, **100 MB**, 词表 206（byte 级） |
| d / L / T | 96 / 3 / 96 | 96/3/128（s1）、192/4/256（s2）、384/6/512（s3） |
| 每格训练量 | ~1.2 M token | **41 M token**（10000 步 × 4096） |
| spectral 臂参数量 | 0.45 M | 0.54 M / 2.01 M / 9.54 M |
| 混合层 | fnet / spectral / attn | 同上 **+ stencil**（严格局部，|lag| ≤ 2） |
| 优化器 | SGD+momentum 0.9, clip 1.0, warmup 200, 线性衰减到 5% | **完全相同** |

实验一的设计约束是"不用 PyTorch / autograd / GPU"，是**第一等公民**。为了既保留这条约束的
可证伪性、又能放大 100×，做法是写一个 PyTorch 移植版并用实验一自己的代码做**逐参数梯度等价性
检验**（`scripts/run_equiv.py`，10/10 PASS，`global_rel` ≤ 2.2e-14）。

## 四条主要结论（详见报告）

1. **局部碾压全局**：严格局部 stencil 在每个规模上都打败参数多 100 倍的全局谱算子
   （s3：0.8263 vs 1.0914，混合参数 11,520 vs 1,184,256）。这是实验一 Q2 的强化版答案。
2. **"谱算子比注意力稳健"反转且非单调**：s1 崩、s2 稳、s3（lr0.8）再崩，s3 降到 lr0.2 即稳。
   实验一 Q7 的答案是"最优 lr 随规模漂移"（§5.5）。
3. **核随规模变得更局部**：s1 lag0=0.79 → s3 lag0=0.99，峰值 lag 恒为 1，lag>8 < 0.2%。
4. **任务要求长程时算子会用长程**：把任务从 lag 4 换成 lag 64，谱核三层全部精确长到 lag 64
   （acc 84.8%，掩码上限 ≈85.5%），注意力 lr=0.4 时 85.2%，而严格局部 stencil **结构性失败**。
   **决定"用不用长程"的是任务，不是算子。**

另外发现**实验一手推反向传播里有一个真实的 bug**（LayerNorm 增益前激活被当成增益后激活算
`W1` 梯度；实验一的梯度校验只在初始化点跑，而该点 `g=1,b=0` 使 bug 不可见）。
见报告 §3.2。

## 复现

```powershell
cd experiments/spectral-mlm-scaleup

# Stage A：实验一未修改的 NumPy 代码，原始规模忠实复现（CPU）
python scripts\run_stageA.py main followup

# Stage B：GPU 移植的数值等价性 + 实验一 bug 的证据
python scripts\run_equiv.py            # 10/10 PASS
python scripts\check_trajectory.py     # 3/3 PASS，150 步 1e-15
python scripts\check_ln_bug.py         # 有限差分钉死实验一的 bug

# Stage C：放大 100×（GPU）
python scripts\run_stageC.py lr        # 学习率扫描
python scripts\run_stageC.py ladder    # s1/s2/s3 × 4 臂
python scripts\run_stageC.py fair      # 注意力 lr/零初始化 对照
python scripts\run_stageC.py extra     # fnet/gated@s3 + LN bug 对照

# Stage D：长程任务（实验一 Q1）
python scripts\make_synthetic.py --task periodic --L 64 --chars 4000000 --out <path>
python scripts\run_stageD.py
python scripts\run_stageD2.py

# 汇总、出图、复算报告里的每个数字
python scripts\aggregate.py "runs/*/*.final.json" --md
python scripts\plot.py
python scripts\verify_claims.py        # VERIFY: 78/78 claims match
```

## 目录

```
scripts\spectral_lm_torch.py   PyTorch GPU 移植版（逐项保持实验一的骨架/目标/优化器/数据管线）
scripts\run_stageA.py          Stage A：原始 NumPy 代码的忠实复现
scripts\run_equiv.py           Stage B：移植版与实验一的逐参数梯度等价性
scripts\check_trajectory.py    Stage B：150 步完整训练轨迹等价性
scripts\check_ln_bug.py        Stage B：用中心差分钉死实验一的 W1 梯度 bug
scripts\run_stageC.py          Stage C：lr 扫描 / scale ladder / fair / extra
scripts\run_stageD.py          Stage D：p4 / p64 / double 长程合成任务
scripts\run_stageD2.py         Stage D2：p64 上的注意力 lr 扫描
scripts\make_synthetic.py      合成语料生成
scripts\count_baselines*.py    n-gram / 双向 ±k 计数参照线（含掩码下的"公平基线"）
scripts\aggregate.py           汇总 runs/ 下的 final.json 为对照表与 RESULTS.md
scripts\plot.py                出图（fig_repro / fig_lr / fig_ladder / fig_kernel）
scripts\verify_claims.py       从原始 run 文件复算报告里引用的 78 个数字
scripts\analyze_numpy_weights.py / inspect_block0.py / localise_divergence.py / diag_trajectory.py
                               核分析与数值分歧定位
runs\                          每次运行的 *.jsonl 曲线 + *.final.json（含诊断/外推/cloze）
logs\                          原始 stdout（Stage A 曲线的权威来源，见下）
figures\                       四张图
ref\                           等价性检验用的 NumPy 参考 dump
RESULTS.md                     aggregate.py 的汇总输出
```

## 两个已知坑

1. **Stage A 的 `tag` 不含 lr**（实验一遗留的工程缺陷，见
   [`docs/03` §8](../../docs/03-experiment-report.md)）：`attn lr0.2` 与 `attn lr0.8` 会写同一个
   `runs/main/attn_T96_d96_L3_s0.jsonl` 而互相覆盖。**Stage A 的逐步骤曲线以 `logs/A_*.log`
   为准**，`scripts/plot.py` 的 `fig_repro` 即解析日志。
2. **`*.weights.npz` 未纳入版本控制**（约 165 MB），与实验一同例；需要时用 `--save_weights`
   重新生成，核分析脚本会读取它们。
