# 01 · 文献核实记录

本仓库的规则：**任何写进文档的技术论断都必须能指到原始出处，并记录核实结论与限定条件。**
本文件是核实过程的留痕，包括被修正的措辞和被否定的说法。

---

## A. 三条核心论断的核实

### A1. FNet：无参数 FFT 替换注意力 —— ✅ 成立（措辞需修正）

- **出处**：Lee-Thorp, Ainslie, Eckstein, Ontañón, *FNet: Mixing Tokens with Fourier
  Transforms*, [arXiv:2105.03824](https://arxiv.org/abs/2105.03824)；
  发表于 NAACL 2022, pp. 4296–4313，获该届 **Best Efficient NLP Paper**
  （[ACL Anthology](https://aclanthology.org/2022.naacl-main.319/)）。
- **摘要原文**：

  > replacing the self-attention sublayer in a Transformer encoder with a standard,
  > **unparameterized Fourier Transform** achieves **92-97% of the accuracy of BERT**
  > counterparts on the GLUE benchmark, but trains **80% faster on GPUs and 70% faster
  > on TPUs** at standard 512 input lengths.

- **核实结论**：成立。
- **需要修正的措辞**：初稿写作"保留约九成 GLUE 精度"，原文实为 **92–97%**，偏低。
- **限定条件**（两个容易误读的点）：
  1. 这是 **encoder-only、BERT 式掩码语言模型**，不是解码器 LLM。
  2. "无参数"仅指那个 2D FFT 混合层本身；FFN、embedding、LayerNorm 仍然有参数且需训练。

### A2. AFNO / FourCastNet —— ✅ 成立（需补上下文）

- **AFNO 出处**：Guibas, Mardani, Li, Tao, Anandkumar, Catanzaro,
  *Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers*,
  [arXiv:2111.13587](https://arxiv.org/abs/2111.13587)。
  摘要原文列出三处关键改动：

  > imposing a **block-diagonal structure on the channel mixing weights**, adaptively
  > sharing weights across tokens, and **sparsifying the frequency modes via
  > soft-thresholding and shrinkage**.

- **FourCastNet 出处**：Pathak 等（含 Zongyi Li、Anandkumar）,
  *FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive
  Fourier Neural Operators*, [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)。
  摘要给出：**0.25° 分辨率**、短期大尺度变量**比肩 ECMWF IFS**、降水等细尺度变量
  **优于 IFS**、**一周预报 < 2 秒**。另有 HPC 版本 arXiv:2208.05419（SC22）。
- **核实结论**：成立。"AFNO 支撑 FourCastNet"因果也对——AFNO 是骨干里的 token mixer。
- **需补的上下文**：FourCastNet 是 2022 年初的工作，随后 2023 年的 GraphCast、
  Pangu-Weather 等在多数指标上超过它。准确说法是"**它支撑了首个有影响力的
  数据驱动全球天气预报模型之一**"，而非"当前最强"。

### A3. FNO 的通用逼近定理 —— ✅ 成立（但有限定，且与"能训练"无关）

- **出处**：Kovachki, Lanthaler, Mishra, *On universal approximation and error bounds
  for Fourier Neural Operators*, [arXiv:2107.07562](https://arxiv.org/abs/2107.07562)，
  正式发表于 **JMLR 22 (2021) 1–76**。摘要原文：

  > We prove that FNOs are **universal**, in the sense that they can approximate
  > **any continuous operator** to desired accuracy. … error bounds … only increases
  > **sub(log)-linearly** in terms of the reciprocal of the error.

- **FNO 本身**：Li 等, *Fourier Neural Operator for Parametric Partial Differential
  Equations*, arXiv:2010.08895（ICLR 2021）；上述定理是后续数学论文补的严格证明。
- **核实结论**：定理成立，但必须交代限定条件，否则会被过度解读：
  定理的成立前提是**取足够多的频率模态与足够的宽度/深度**。当模态数被固定、
  且算子含强局部化或非光滑结构时，纯谱截断会失效——这正推动了后续
  Local Neural Operator 一类工作（如 ICML 2024, *Neural Operators with Localized
  Integral and Differential Kernels*）。
- **本仓库特别声明**：定理是**表达能力**陈述，**不蕴含可训练性**。
  实验一表明，决定成败的是初始化与条件数，而不是定理。
  "有定理背书"与"能训得动"是两件独立的事。

---

## B. 算子族清单中引用的条目核实

| 论断 | 出处 | 结论 |
|---|---|---|
| 热带半环（max-plus）注意力 | *Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms*, [arXiv:2505.17190](https://ar5iv.labs.arxiv.org/html/2505.17190) | ✅ 存在 |
| Lenia（连续元胞自动机） | Bert Chan, *Lenia — Biology of Artificial Life*, [arXiv:1812.05433](https://arxiv.org/pdf/1812.05433v2) | ✅ 存在 |
| 现代 Hopfield = 注意力 | Ramsauer 等, *Hopfield Networks is All You Need*, [ICLR 2021](https://mlanthology.org/iclr/2021/ramsauer2021iclr-hopfield/) | ✅ 存在 |
| Growing Neural CA | Mordvintsev 等, Distill 2020（多份独立复现指向同一工作） | ✅ 存在 |
| MERA 与重整化群 / 深度网络的对应 | Vidal 2007 提出 MERA；与 RG、深度表示的对应有独立文献（[INSPIRE](https://inspirehep.net/literature/2728120)） | ✅ 存在，但"对应"是研究program而非已证定理 |
| 蝶形/ Monarch 稀疏可学变换 | Dao 等, *Monarch*、*Kaleidoscope*（本仓库前序讨论中已核） | ✅ 存在 |

---

## C. 本次核实中**未被采信**或**被降级**的说法

诚实记录我们放弃了哪些表述：

1. ~~"FNet 达到 BERT 九成精度"~~ → 应为 92–97%，且仅 encoder。
2. ~~"AFNO 是当前最强天气预报模型"~~ → FourCastNet 已被 2023 年的工作超越。
3. ~~"FNO 有通用逼近定理，所以什么算子都能学好"~~ → 定理需"足够多模态"前提，
   固定模态下有已知失效场景；且定理与可训练性无关。
4. ~~"傅里叶参数化天然支持长度外推"~~ → 实验一实测**否证**：
   在 T=96 训练、零样本外推到 T=768，spectral 从 1.27 退化到 3.94，fnet 从 2.34 到 5.15。
   两个 arm 都灾难性失败。
5. ~~"纯傅里叶系统可以涌现智能"（本次讨论早期的直觉）~~ → 数学上**否证**：
   归一化 DFT 满足 `F⁴ = I`，纯算子的任意深度复合退化为一次线性变换，无深度可言。
   必须插入非线性（判据 4）。

---

## D. 核实方法

- 一律读**原始出处**（arXiv 摘要页 / ACL Anthology / JMLR），不依赖二手转述。
- 记录**原文引语**而非概括，便于后续复核。
- 逐条标注**判定**（成立 / 需修正 / 否定）与**限定条件**。
- 对无法确认的条目，写"未能核实"而不是模糊表述；本文件目前没有此类条目。
