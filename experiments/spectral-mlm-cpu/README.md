# 实验一 · spectral-mlm-cpu

用**纯 NumPy**（无 PyTorch / 无 autograd / 无 GPU / 无 Adam）实现的 BERT 式掩码语言模型，
三个 arm 除 token 混合层外完全相同：

| arm | 混合层 | 活跃混合参数 |
|---|---|---|
| `fnet` | `y = Re(FFT(x))`，零参数 | 0 |
| `spectral` | `y = Re(IFFT(H ⊙ FFT(x)))`，可学习复数滤波器 | 28,224 |
| `attn` | 手写 4 头自注意力 | 111,744 |

- **完整报告**：`../../docs/03-experiment-report.md`
- **设计文档**：`../../docs/04-experiment-design.md`
- **结果汇总**：`results/SUMMARY.md`

## 复现

```bash
python3 spectral_bert.py --gradcheck          # 手推梯度的中心差分校验，必须先全过
bash run_main.sh 0 6000                       # 三臂主实验（每臂用自己最优 lr）
bash run_followup.sh                          # 稳健性对照：wo_zero / seed1 / lr0.2
python3 aggregate.py "results/*/*.final.json"
python3 plot.py                               # 生成 results/comparison.png
```

## 目录

```
spectral_bert.py   模型 + 手推反向传播 + SGD + MLM 训练（单文件，含 --gradcheck）
debug_grad.py      按参数组定位梯度错误的诊断脚本
aggregate.py       汇总 results/ 下的 final.json 为对照表
plot.py            四联图：损失、准确率、长度外推、学到的滤波器幅度
run_main.sh        三臂主实验
run_followup.sh    稳健性对照
results/           原始曲线（jsonl）、外推/cloze/频谱（final.json）、对比图、SUMMARY.md
tinyshakespeare.txt  语料（1,115,394 字符，来自 Karpathy 的 char-rnn 仓库）
```

> 3.6MB 的模型权重 `*.weights.npz` 未纳入版本控制；用 `--save_weights` 重新生成。
