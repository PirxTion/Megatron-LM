# 课时 1：为什么需要 Megatron-LM

Date: 2026-03-25
Source: Megatron-LM codebase + Megatron 系列论文

---

## 学习目标

完成本课时后，你应该能够：

1. 解释大模型训练面临的三面"墙"，以及为什么单卡训练不够
2. 用自己的话描述 DP、TP、PP、CP、EP 五种并行策略分别解决什么问题
3. 理解 Megatron Core 和 Megatron Training 两个组件的定位差异
4. 对 Megatron-LM 代码仓库的顶层结构建立心智模型

---

## 第一节：大模型训练的三面墙

### 1.1 显存墙（Memory Wall）

训练一个 Transformer 模型，GPU 显存需要承载四类数据：

| 类别 | 内容 | 量级估算（以 70B 参数为例） |
|------|------|---------------------------|
| **模型参数** | 权重矩阵 W | 70B × 2 bytes (BF16) = **140 GB** |
| **优化器状态** | Adam 的 m, v（FP32） | 70B × 4 × 2 = **560 GB** |
| **梯度** | 与参数同形状 | 70B × 2 bytes = **140 GB** |
| **激活值** | 前向传播的中间结果 | 随 batch size 和序列长度变化，可达 **数百 GB** |

#### 激活值显存的计算方法

激活值是前向传播中需要保存给反向传播使用的中间结果。要理解"需要保存什么"，需要知道每个操作的反向传播需要什么输入。

**符号约定：** s=序列长度, b=batch, h=hidden_size, a=num_heads, d=h/a (head_dim)。所有张量默认 BF16（2 bytes/元素），dropout mask 为 1 byte/元素。

##### Transformer 层完整前向传播

```
输入 x: [s, b, h]
│
├── ① LayerNorm1 ──────────── x_norm: [s, b, h]
│
├── ② QKV 线性投影 ─────────── Q, K, V: 各 [s, b, h]
│     (三个独立线性层，或一个大的 [h, 3h] 投影后拆分)
│
├── ③ Reshape for multi-head ─ Q, K, V: 各 [s, b, a, d]
│
├── ④ Attention scores ─────── S = QKᵀ/√d: [b, a, s, s]    ← O(s²) 显存！
│
├── ⑤ Softmax ─────────────── P = softmax(S): [b, a, s, s]
│
├── ⑥ Attention dropout ───── P_drop = dropout(P): [b, a, s, s]
│
├── ⑦ Context ─────────────── C = P_drop × V: [s, b, h]
│     (reshape 回 [s, b, h])
│
├── ⑧ Output projection ──── attn_out = C·Wₒ: [s, b, h]
│
├── ⑨ Dropout + 残差 ─────── y = x + dropout(attn_out): [s, b, h]
│
├── ⑩ LayerNorm2 ──────────── y_norm: [s, b, h]
│
├── ⑪ MLP 第一层 ──────────── m1 = y_norm·W₁: [s, b, 4h]
│
├── ⑫ GeLU 激活 ──────────── m2 = GeLU(m1): [s, b, 4h]
│
├── ⑬ MLP 第二层 ──────────── m3 = m2·W₂: [s, b, h]
│
└── ⑭ Dropout + 残差 ─────── z = y + dropout(m3): [s, b, h]
                               ↓
                          输出 z: [s, b, h] → 下一层的输入
```

##### 每步的求导公式与需要保存的激活

**核心原则：** 对于 $Y = f(X)$，反向传播计算 $\frac{\partial \mathcal{L}}{\partial X}$ 时需要什么，就保存什么。

| 步骤 | 操作 | 求导公式 | 需要保存 | 大小 (bytes) |
|------|------|---------|---------|-------------|
| ① | $\mathbf{x}_{\text{norm}} = \text{LayerNorm}(\mathbf{x})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ 需要 $\mathbf{x}$ 来计算 $\mu, \sigma^2$ | **$\mathbf{x}$**（LN1 输入） | $2sbh$ |
| ② | $Q = \mathbf{x}_{\text{norm}} W_Q$（K, V 同理） | $\frac{\partial \mathcal{L}}{\partial W_Q} = \mathbf{x}_{\text{norm}}^\top \frac{\partial \mathcal{L}}{\partial Q}$ → 需要输入 | **$\mathbf{x}_{\text{norm}}$**（QKV 投影输入） | $2sbh$ |
|  |  | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{norm}}} = \frac{\partial \mathcal{L}}{\partial Q} W_Q^\top$ → 需要 $W$（已有） | **$Q, K, V$**（④⑦的输入，在此产出并保存） | $6sbh$ |
| ③ | Q, K, V reshape | 无参数，无额外存储 | — | 0 |
| ④ | $S = \frac{QK^\top}{\sqrt{d}}$ | $\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot \frac{K}{\sqrt{d}}$ → 需要 $K$ | $Q, K$ 已在②中保存 | 0 |
|  |  | $\frac{\partial \mathcal{L}}{\partial K} = \frac{\partial \mathcal{L}}{\partial S} \cdot \frac{Q}{\sqrt{d}}$ → 需要 $Q$ | | |
| ⑤ | $P = \text{softmax}(S)$ | $\frac{\partial \mathcal{L}}{\partial S_i} = P_i \left( \frac{\partial \mathcal{L}}{\partial P_i} - \sum_j P_j \frac{\partial \mathcal{L}}{\partial P_j} \right)$ → 需要 $P$ | **$P$**（softmax 输出） | $2bas^2$ |
| ⑥ | $P_{\text{drop}} = \text{dropout}(P, \text{mask})$ | $\frac{\partial \mathcal{L}}{\partial P} = \frac{\partial \mathcal{L}}{\partial P_{\text{drop}}} \cdot \frac{\text{mask}}{1-p}$ → 需要 mask | **mask** | $bas^2$ |
| ⑦ | $C = P_{\text{drop}} \cdot V$ | $\frac{\partial \mathcal{L}}{\partial P_{\text{drop}}} = \frac{\partial \mathcal{L}}{\partial C} \cdot V^\top$ → 需要 $V$ | $V$ 已在②中保存 | 0 |
|  |  | $\frac{\partial \mathcal{L}}{\partial V} = P_{\text{drop}}^\top \frac{\partial \mathcal{L}}{\partial C}$ → 需要 $P_{\text{drop}}$ | $P_{\text{drop}}$ 可从 $P$ + mask 恢复 | 0 |
| ⑧ | $\mathbf{a} = C \cdot W_o$ | $\frac{\partial \mathcal{L}}{\partial W_o} = C^\top \frac{\partial \mathcal{L}}{\partial \mathbf{a}}$ → 需要 $C$ | **$C$**（OProj 输入） | $2sbh$ |
| ⑨ | $\mathbf{y} = \mathbf{x} + \text{dropout}(\mathbf{a})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$（直通）; dropout 需要 mask | **dropout mask** | $sbh$ |
| ⑩ | $\mathbf{y}_{\text{norm}} = \text{LayerNorm}(\mathbf{y})$ | 同①，需要 $\mathbf{y}$ | **$\mathbf{y}$**（LN2 输入） | $2sbh$ |
| ⑪ | $\mathbf{m}_1 = \mathbf{y}_{\text{norm}} W_1$ | $\frac{\partial \mathcal{L}}{\partial W_1} = \mathbf{y}_{\text{norm}}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{m}_1}$ → 需要 $\mathbf{y}_{\text{norm}}$ | **$\mathbf{y}_{\text{norm}}$**（$W_1$ 输入） | $2sbh$ |
| ⑫ | $\mathbf{m}_2 = \text{GeLU}(\mathbf{m}_1)$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{m}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{m}_2} \odot \text{GeLU}'(\mathbf{m}_1)$ | **$\mathbf{m}_1$**（GeLU 输入） | $2 \cdot sb \cdot 4h = 8sbh$ |
|  |  | $\text{GeLU}'(x) = \Phi(x) + x \cdot \phi(x)$，依赖原始输入 $x$ | 不能从 $\mathbf{m}_2$ 反推 $\mathbf{m}_1$！ | |
| ⑬ | $\mathbf{m}_3 = \mathbf{m}_2 W_2$ | $\frac{\partial \mathcal{L}}{\partial W_2} = \mathbf{m}_2^\top \frac{\partial \mathcal{L}}{\partial \mathbf{m}_3}$ → 需要 $\mathbf{m}_2$ | **$\mathbf{m}_2$**（$W_2$ 输入） | $2 \cdot sb \cdot 4h = 8sbh$ |
| ⑭ | $\mathbf{z} = \mathbf{y} + \text{dropout}(\mathbf{m}_3)$ | 同⑨ | **dropout mask** | $sbh$ |

##### 汇总

**线性部分**（与 $sbh$ 成正比）：

| 保存的激活 | 来自步骤 | 大小 (bytes) |
|-----------|---------|-------------|
| $\mathbf{x}$（LN1 输入） | ① | $2sbh$ |
| $\mathbf{x}_{\text{norm}}$（QKV 投影输入） | ② | $2sbh$ |
| $Q, K, V$（attention 计算输入） | ④⑦ | $6sbh$ |
| $C$（output projection 输入） | ⑧ | $2sbh$ |
| dropout mask | ⑨ | $sbh$ |
| $\mathbf{y}$（LN2 输入） | ⑩ | $2sbh$ |
| $\mathbf{y}_{\text{norm}}$（MLP $W_1$ 输入） | ⑪ | $2sbh$ |
| $\mathbf{m}_1$（GeLU 输入，$4h$ 维） | ⑫ | $8sbh$ |
| $\mathbf{m}_2$（MLP $W_2$ 输入，$4h$ 维） | ⑬ | $8sbh$ |
| dropout mask | ⑭ | $sbh$ |
| **小计** | | **$\sim 34sbh$** |

**Attention scores**（与 $s^2$ 成正比）：

| 保存的激活 | 来自步骤 | 大小 (bytes) |
|-----------|---------|-------------|
| $P$（softmax 输出） | ⑤ | $2bas^2$ |
| attention dropout mask | ⑥ | $bas^2$ |
| **小计** | | **$\sim 3bas^2$** |

> **每层激活 $\approx 34sbh + 3bas^2$ bytes**

注：一些文献中使用 $5bas^2$ 系数，取决于是否将 attention scores $S$ 本身也保存（额外 $2bas^2$），实现细节有所不同。

三种优化策略对应不同的激活显存：

| 策略 | 保留的部分 | 每层显存 | 代价 |
|------|-----------|---------|------|
| **无优化** | 全部（$34sbh + 3 \sim 5 \cdot bas^2$） | 最大 | 无 |
| **选择性重计算** | 去掉 attention scores（$\sim 34sbh$） | 中等 | 反向时重算 attention |
| **全重计算** | 只存层输入（$2sbh$） | 最小 | 反向时重算整层，慢 30-40% |

以 70B 模型（$s=4096, b=4, h=8192, a=64, L=80$ 层）为例：
- 无优化：每层 $\sim$ 70 GB → 80 层 **$\sim$ 5600 GB**（attention scores 的 $s^2$ 项主导）
- 选择性重计算：每层 $\sim$ 9 GB → 80 层 **$\sim$ 720 GB**
- 全重计算：每层 $\sim$ 0.27 GB → 80 层 **$\sim$ 21 GB**

**关键洞察：** 一块 H100 GPU 只有 80 GB 显存。即便只看模型参数（140 GB），单卡也放不下一个 70B 模型。而加上优化器状态，总需求超过 **840 GB**，需要至少 11 块 H100。

### 1.2 算力墙（Compute Wall）

- 训练 LLaMA-3 70B 需要约 **1.7M GPU-hours**（H100）
- 单卡训练需要约 **194 年**
- 即使 1024 块 H100，也需要约 **69 天**

算力需求随模型规模超线性增长，分布式训练不是"优化"，而是"必须"。

### 1.3 通信墙（Communication Wall）

把模型分散到多块 GPU 后，GPU 之间需要频繁交换数据：

- **梯度同步**（数据并行）：每个训练步都要 allreduce 所有梯度
- **激活值传输**（流水线并行）：相邻 stage 之间传递中间结果
- **权重切片通信**（张量并行）：矩阵乘法的局部结果需要 allreduce/allgather

通信开销可能占训练时间的 **30-50%**。Megatron 的核心技术贡献之一就是将通信隐藏在计算之下（通信/计算重叠）。

### 1.4 思考题

> 如果你有 64 块 H100（80GB），需要训练一个 175B 参数的模型，仅模型参数 + 优化器状态就需要多少 GB？
> 你至少需要多少块卡才能装下？（假设 BF16 训练 + Adam 优化器）

混合精度训练（BF16 + Adam）每个参数需要 ~18 bytes：

| 组件 | 精度 | 计算 | 大小 |
|------|------|------|------|
| BF16 模型参数（前向/反向用） | BF16 | 175B × 2 | 350 GB |
| FP32 master weights（优化器更新用） | FP32 | 175B × 4 | 700 GB |
| FP32 梯度 | FP32 | 175B × 4 | 700 GB |
| Adam m + v | FP32 | 175B × 4 × 2 | 1400 GB |
| **合计** | | 175B × 18 bytes | **3150 GB** |

至少需要 3150 / 80 ≈ **40 块 H100**（不含 activations）。

关键点：混合精度训练需要同时维护 BF16 权重（用于前向/反向）和 FP32 master weights（用于 Adam 更新），更新完后 cast 回 BF16。

---

## 第二节：五种并行策略直觉

Megatron 实现了五个维度的并行，每个维度解决不同的瓶颈。以下用餐厅厨房的类比来建立直觉：

### 2.1 数据并行（Data Parallelism, DP）

```
             ┌─── GPU 0: 完整模型副本 ←── Batch 0
  数据集 ───├─── GPU 1: 完整模型副本 ←── Batch 1
             └─── GPU 2: 完整模型副本 ←── Batch 2
                         ↓ allreduce 梯度
```

**类比：** 3 个厨师各做一份完整的菜（同一菜谱），但处理不同的食材批次。做完后交流心得（梯度同步）。

**解决的问题：** 加速训练（throughput），每块 GPU 处理不同的数据子集
**限制：** 每块 GPU 必须装下完整模型 → 对超大模型无效
**代码位置：** `megatron/core/distributed/`

### 2.2 张量并行（Tensor Parallelism, TP）

```
  一个线性层 Y = XW，W 的形状为 [H, 4H]：

  GPU 0: W₁ = W[:, 0:2H]  →  Y₁ = X · W₁
  GPU 1: W₂ = W[:, 2H:4H] →  Y₂ = X · W₂
                                ↓ AllGather
                          Y = [Y₁, Y₂]
```

**类比：** 一道菜太复杂，一个厨师忙不过来。把菜分成两半，两个厨师各做一半，最后拼到一个盘子里。

**解决的问题：** 单层太大，放不进一块 GPU 的显存
**特点：** 通信频繁（每个层都要通信），适合节点内高带宽 NVLink 连接
**代码位置：** `megatron/core/tensor_parallel/`

### 2.3 流水线并行（Pipeline Parallelism, PP）

```
  模型有 16 层：

  GPU 0: Layer  1-4   (Stage 0)
  GPU 1: Layer  5-8   (Stage 1)
  GPU 2: Layer  9-12  (Stage 2)
  GPU 3: Layer 13-16  (Stage 3)
        ↓ P2P send/recv
```

**类比：** 流水线工厂。每个工人负责加工的一个工序，半成品从一个工位传到下一个。

**解决的问题：** 模型太深（层数太多），垂直切分到不同 GPU
**挑战：** "气泡"（pipeline bubble）—— 前面的 stage 在等后面的 stage 做完
**代码位置：** `megatron/core/pipeline_parallel/`

### 2.4 上下文并行（Context Parallelism, CP）

```
  输入序列长度 128K tokens：

  GPU 0: tokens  0-32K
  GPU 1: tokens  32K-64K
  GPU 2: tokens  64K-96K
  GPU 3: tokens  96K-128K
        ↓ Ring Attention 通信 KV
```

**类比：** 一本很长的书，每个人读一章，但讨论理解时需要知道其他章节的内容（注意力的 KV 交换）。

**解决的问题：** 长序列的 attention 显存 O(n²) 瓶颈
**代码位置：** `megatron/core/pipeline_parallel/hybrid_cp_schedule.py`

### 2.5 专家并行（Expert Parallelism, EP）

```
  MoE 层有 64 个专家：

  GPU 0: Expert  0-15    ←── Router 分发 token
  GPU 1: Expert 16-31    ←── AllToAll 通信
  GPU 2: Expert 32-47
  GPU 3: Expert 48-63
```

**类比：** 一家医院，有 64 个专科医生。不是每个病人都要看所有科室，分诊台（Router）把病人指向合适的专家。

**解决的问题：** MoE 模型的专家数量太多，无法全放在一块 GPU 上
**代码位置：** `megatron/core/transformer/moe/`

### 2.6 五种并行的组合

在实际训练中，五种并行同时使用。GPU 总数 = TP × PP × DP × CP × EP。

以 Megatron 源码中的例子为参考（`parallel_state.py` 第 692-706 行）：

> 假设有 16 个 GPU (g0...g15)，TP=2，PP=4，则 DP=16/(2×4)=2，进程组划分如下：
>
> - 8 个 TP 组：[g0,g1], [g2,g3], [g4,g5], [g6,g7], [g8,g9], [g10,g11], [g12,g13], [g14,g15]
> - 4 个 PP 组：[g0,g4,g8,g12], [g1,g5,g9,g13], [g2,g6,g10,g14], [g3,g7,g11,g15]
> - 8 个 DP 组：[g0,g2], [g1,g3], [g4,g6], [g5,g7], [g8,g10], [g9,g11], [g12,g14], [g13,g15]

**理解"组大小"与"组数量"：** 上面的例子中 DP=2 但有 8 个 DP 组，初看可能困惑。关键在于：**DP=2 是每组的大小（size），不是组的数量（count）**。

把 16 块 GPU 想象成一个 3D 网格 `[TP=2, DP=2, PP=4]`：

```
                PP stage 0    PP stage 1    PP stage 2    PP stage 3
              ┌─────────────┬─────────────┬─────────────┬─────────────┐
DP replica 0  │ g0,g1 (TP组) │ g4,g5 (TP组) │ g8,g9 (TP组) │ g12,g13(TP组)│
DP replica 1  │ g2,g3 (TP组) │ g6,g7 (TP组) │ g10,g11(TP组)│ g14,g15(TP组)│
              └─────────────┴─────────────┴─────────────┴─────────────┘
```

每个格子是一个 TP 组（2 块 GPU 共同持有同一层的切片）。DP 组把**同一位置（同 TP rank、同 PP stage）但不同 data replica** 的 GPU 连在一起——它们持有相同参数，处理不同数据，需要 allreduce 梯度。

从上图竖着看（同列同位置）就是 DP 组：

| DP 组 | 成员 | 含义 |
|-------|------|------|
| 1 | [g0, g2] | PP stage 0, TP rank 0 的两个 replica |
| 2 | [g1, g3] | PP stage 0, TP rank 1 的两个 replica |
| 3 | [g4, g6] | PP stage 1, TP rank 0 的两个 replica |
| ... | ... | ... |
| 8 | [g13, g15] | PP stage 3, TP rank 1 的两个 replica |

通用规律——对任意并行维度 X：

> **组大小 = X，组数量 = 总 GPU 数 / X**

验证：TP 组大小 2，数量 16/2=8 ✓ | PP 组大小 4，数量 16/4=4 ✓ | DP 组大小 2，数量 16/2=8 ✓

**思考题：** 为什么 TP 组总是相邻的 GPU（g0,g1 而不是 g0,g8）？提示：考虑 NVLink vs InfiniBand 的带宽差异。

**答：** 因为 Megatron 的 GPU 编号排布顺序是 **TP（最内）→ DP → PP（最外）**，编号先填满同一 PP stage 的所有 DP replica，再进入下一个 PP stage。这一排布的原因是三种并行对带宽需求不同：

| 维度 | 通信模式 | 频率 | 排布位置 | 原因 |
|------|---------|------|---------|------|
| **TP** | AllReduce/AllGather（每层都做） | 最高 | 最内层：相邻 GPU | 需要 NVLink 高带宽 |
| **DP** | AllReduce 梯度（每步一次） | 中等 | 中间层 | 带宽需求适中 |
| **PP** | P2P send/recv（stage 边界） | 最低 | 最外层 | 可以跨节点，带宽要求低 |

通信越频繁的维度，GPU 编号越紧凑，物理距离越近。

以 TP=4, DP=4, PP=2（共 32 GPU）为例，编号展开是：

```
g0-g3:   TP组, DP=0, PP=0  ┐
g4-g7:   TP组, DP=1, PP=0  │ 先填满 PP stage 0
g8-g11:  TP组, DP=2, PP=0  │ 的所有 DP replica
g12-g15: TP组, DP=3, PP=0  ┘
g16-g19: TP组, DP=0, PP=1  ┐
g20-g23: TP组, DP=1, PP=1  │ 再填 PP stage 1
g24-g27: TP组, DP=2, PP=1  │
g28-g31: TP组, DP=3, PP=1  ┘
```

所以 g0→g4 是 DP 关系（同 stage，不同 replica），g0→g16 才是 PP 关系（同 replica，不同 stage）。

---

## 第三节：Megatron 项目的两个组件

### 3.1 Megatron Core（`megatron/core/`）

**定位：** 可组合的积木库（composable library）

提供 GPU 优化的底层构建块，让你可以组装自己的训练框架。类似于 PyTorch 之于深度学习 —— 提供原子操作，不限定使用方式。

主要模块：

| 目录 | 职责 |
|------|------|
| `parallel_state.py` | 并行拓扑的"大脑"——管理所有进程组（详见下方） |
| `tensor_parallel/` | 张量并行的层实现（ColumnParallelLinear 等） |
| `pipeline_parallel/` | 流水线调度（1F1B 等） |
| `distributed/` | 数据并行（DDP、FSDP） |
| `transformer/` | Transformer 构建块（Attention, MLP, Layer, Block） |
| `models/` | 完整模型定义（GPT, BERT, T5, Mamba, VLM） |
| `optimizer/` | 分布式优化器 |
| `datasets/` | 数据管线 |
| `dist_checkpointing/` | 分布式 checkpoint（支持 resharding） |

#### `parallel_state.py` 全局变量详解

`parallel_state.py` 前 140 行定义了 40+ 个全局变量，分为 5 类：

**① 基础并行组**（5 个维度，各一个 NCCL 进程组）

| 变量 | 对应维度 | 用途 |
|------|---------|------|
| `_TENSOR_MODEL_PARALLEL_GROUP` | TP | 层内张量切分通信 |
| `_PIPELINE_MODEL_PARALLEL_GROUP` | PP | 层间流水线 P2P 通信 |
| `_DATA_PARALLEL_GROUP` | DP | 梯度 allreduce |
| `_CONTEXT_PARALLEL_GROUP` | CP | 序列切分的 KV 交换 |
| `_EXPERT_MODEL_PARALLEL_GROUP` | EP | 专家分发的 AllToAll |

**② 组合并行组**（跨维度操作需要，单个维度的组不够用）

| 变量 | 组合维度 | 为什么需要 |
|------|---------|-----------|
| `_MODEL_PARALLEL_GROUP` | TP × PP | 标识整个模型并行域 |
| `_TENSOR_AND_DATA_PARALLEL_GROUP` | TP × DP | FP8 需要跨 TP+DP 统计 tensor 分布来做 scaling |
| `_DATA_PARALLEL_GROUP_WITH_CP` | DP × CP | CP 各 rank 看到不同 token → 梯度不同 → 梯度同步必须包含 CP |
| `_TENSOR_AND_CONTEXT_PARALLEL_GROUP` | TP × CP | |
| `_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP` | TP × DP × CP | FP8 + CP 场景 |
| `_HYBRID_DP_CP_GROUPS` | DP + CP 混合 | 混合调度（dict 类型） |

**③ 专家并行组**（MoE 特有，EP 引入额外的 TP/DP 维度）

| 变量 | 含义 |
|------|------|
| `_EXPERT_TENSOR_PARALLEL_GROUP` | 单个专家内部的张量并行 |
| `_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP` | 专家 TP × EP |
| `_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP` | 专家 TP × EP × PP |
| `_EXPERT_DATA_PARALLEL_GROUP` | 专家 DP（持有相同专家副本的 rank 间同步梯度） |
| `_INTRA/INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP` | 部分专家 DP（梯度累积优化） |

为什么专家需要独立的 DP 组？因为 EP 改变了"谁持有相同权重"的映射——非 MoE 层的 DP 组和 MoE 层的 DP 组成员不同。

**④ 全局 rank 列表**（local rank → global rank 的映射表）

| 变量 | 用途 |
|------|------|
| `_PIPELINE_GLOBAL_RANKS` | 本 PP 组各 stage 的全局 rank |
| `_DATA_PARALLEL_GLOBAL_RANKS` | 本 DP 组各成员的全局 rank |
| `_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS` | 本 TP 组各成员的全局 rank |
| `_CONTEXT_PARALLEL_GLOBAL_RANKS` | 本 CP 组各成员的全局 rank |
| `_EMBEDDING_GLOBAL_RANKS` | 持有 embedding 副本的全局 rank |
| ... | |

为什么需要？NCCL 的 `broadcast(src=?)` 等 API 要求填**全局 rank**，而代码逻辑用的是 local rank（如"PP stage 0"）。列表提供直接查表：

```python
# g5 的 PP 组是 [g1, g5, g9, g13]
_PIPELINE_GLOBAL_RANKS = [1, 5, 9, 13]
_PIPELINE_GLOBAL_RANKS[0]  # → 1，即 stage 0 的全局 rank
```

**⑤ 动态覆盖值 + 特殊用途**

| 变量 | 用途 |
|------|------|
| `_MPU_*_WORLD_SIZE / _RANK` | 运行时覆盖并行大小/rank，不重建进程组 |
| `_VIRTUAL_PIPELINE_MODEL_PARALLEL_*` | Virtual PP（interleaved 1F1B 调度） |
| `_EMBEDDING_GROUP` | PP 首尾 stage 同步 embedding 权重 |
| `_*_GLOO` 后缀变量 | CPU 端通信后端（不需要 GPU 的集合操作） |
| `_GLOBAL_MEMORY_BUFFER` | 预分配内存缓冲区，避免反复 malloc |

`_MPU_*` 的典型场景：用 TP=8 训练，推理只有 1 卡。设置 `_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1` 后，所有查询 TP 大小的代码都返回 1，无需改动其他代码：

```python
def get_tensor_model_parallel_world_size():
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE  # ← 优先用覆盖值
    return torch.distributed.get_world_size(group=_TENSOR_MODEL_PARALLEL_GROUP)
```

### 3.2 Megatron Training（`megatron/training/`）

**定位：** 参考训练实现（reference implementation）

基于 Megatron Core 搭建的一套完整训练脚本，包括参数解析、训练循环、日志记录等。适合研究团队快速实验。

关键文件：

| 文件 | 职责 | 规模 |
|------|------|------|
| `training.py` | 主训练循环（`pretrain()` 函数） | ~156 KB |
| `arguments.py` | 命令行参数解析（50+ 参数组） | ~183 KB |
| `initialize.py` | 分布式环境初始化 | ~582 行 |
| `checkpointing.py` | Checkpoint 保存/加载 | — |

### 3.3 两者的关系

```
┌─────────────────────────────────────────────────┐
│  pretrain_gpt.py / pretrain_bert.py / ...       │  ← 入口脚本
├─────────────────────────────────────────────────┤
│  Megatron Training (megatron/training/)         │  ← 训练管道
│    training loop, argument parsing, logging     │
├─────────────────────────────────────────────────┤
│  Megatron Core (megatron/core/)                 │  ← 底层积木
│    parallelism, models, optimizer, datasets     │
├─────────────────────────────────────────────────┤
│  PyTorch / NCCL / CUDA                          │  ← 基础设施
└─────────────────────────────────────────────────┘
```

**设计哲学：** Megatron Core 是无状态的积木，不假设你用什么训练框架。NVIDIA 的 NeMo 框架和其他第三方框架都可以使用 Megatron Core 而不依赖 Megatron Training。

---

## 第四节：代码仓库顶层地图

```
Megatron-LM/
├── megatron/
│   ├── core/                 # 【核心库】GPU 优化的分布式训练积木
│   │   ├── parallel_state.py # ★ 最重要的单文件：并行拓扑管理
│   │   ├── models/           # 模型定义（GPT, BERT, T5, Mamba, VLM...）
│   │   ├── transformer/      # Transformer 构建块 + MoE
│   │   ├── tensor_parallel/  # 张量并行实现
│   │   ├── pipeline_parallel/# 流水线并行调度
│   │   ├── distributed/      # 数据并行（DDP, FSDP）
│   │   ├── optimizer/        # 分布式优化器
│   │   ├── datasets/         # 数据加载管线
│   │   ├── dist_checkpointing/ # 分布式 checkpoint
│   │   └── inference/        # 推理引擎
│   ├── training/             # 【训练框架】参考训练实现
│   ├── legacy/               # 旧版代码（不建议学习）
│   ├── post_training/        # 后训练（量化、蒸馏、剪枝）
│   └── rl/                   # RLHF 训练
│
├── pretrain_gpt.py           # GPT 训练入口 ← 课时 2 精读的起点
├── pretrain_bert.py          # BERT 训练入口
├── pretrain_mamba.py         # Mamba SSM 训练入口
├── pretrain_vlm.py           # Vision-Language 训练入口
├── train_rl.py               # RLHF 训练入口
├── model_provider.py         # 模型工厂
├── gpt_builders.py           # GPT 模型构建器
│
├── tools/                    # 工具脚本（数据预处理、推理服务器等）
├── tests/                    # 测试套件
├── examples/                 # 训练示例配置
└── docs/                     # 文档
```

### 关键数字感受

| 指标 | 数值 |
|------|------|
| `parallel_state.py` | **2192 行**，管理 40+ 全局进程组变量 |
| `training.py` | **~156 KB**，完整训练循环 |
| `arguments.py` | **~183 KB**，50+ 参数组 |
| 支持的模型 | GPT, BERT, T5, Mamba, Vision, VLM, MIMO, HuggingFace |
| 最大验证规模 | **462B 参数，6144 H100 GPU** |
| 最高 MFU | **47-48%**（H100 集群） |

---

## 第五节：一次 GPT 训练的完整调用链（预览）

这里先建立全局印象，课时 2 会详细精读每个文件。

打开 `pretrain_gpt.py`，从 `__main__` 入口开始：

```python
# pretrain_gpt.py 第 399-421 行（简化）
if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,   # 数据集构建函数
        partial(model_provider, gpt_builder), # 模型构建函数
        ModelType.encoder_or_decoder,         # 模型类型
        forward_step,                         # 前向步骤函数
    )
```

`pretrain()` 函数（在 `megatron/training/training.py` 中）做了以下事情：

```
pretrain()
  ├── 1. initialize_megatron()          # 解析参数、初始化分布式环境
  │     ├── 设置随机种子
  │     ├── 初始化 NCCL 进程组         ← parallel_state.initialize_model_parallel()
  │     └── 设置日志
  │
  ├── 2. model_provider()               # 构建模型
  │     └── gpt_builder()              ← 根据配置选择 layer spec
  │
  ├── 3. get_megatron_optimizer()       # 创建分布式优化器
  │
  ├── 4. load_checkpoint()              # 加载 checkpoint（如有）
  │
  └── 5. while not done:                # 训练循环
        ├── train_step()
        │   ├── forward_backward_func() ← PP 调度器（1F1B 等）
        │   ├── finalize_model_grads()  ← 梯度同步（DP allreduce）
        │   └── optimizer.step()        ← 参数更新
        ├── log_metrics()
        ├── evaluate()                  ← 定期验证
        └── save_checkpoint()           ← 定期保存
```

**要点：** 整个框架的设计是"函数注入"模式 —— `pretrain()` 接收用户提供的 `model_provider`、`forward_step`、`datasets_provider` 函数，自己只负责训练循环的编排。

---

## 第六节：论文导读

### 必读论文

**[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)** (Shoeybi et al., 2020)

这是 Megatron 的奠基论文，核心贡献：

1. **张量并行的具体方案**：如何将 MLP 和 Self-Attention 分布到多块 GPU
   - MLP：第一层列切分 → 第二层行切分 → 只需一次 allreduce
   - Attention：每个 head 天然独立 → 按 head 切分
2. **性能数据**：在 512 块 V100 上训练 8.3B 参数模型，达到 76% 的 scaling efficiency
3. **工程洞察**：TP 应该在节点内使用（NVLink 高带宽），PP 在节点间使用

### 推荐论文

| 论文 | 与本课时的关系 |
|------|--------------|
| *Efficient Large-Scale Language Model Training on GPU Clusters* (Narayanan et al., 2021) | PP + TP 组合，1F1B 调度 |
| *GPipe: Efficient Training of Giant Neural Networks* (Huang et al., 2019) | 流水线并行的最早形式 |
| *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020) | 分布式优化器的替代方案，帮助理解 Megatron DistributedOptimizer |

---

## 第七节：动手环节

### 练习 1：显存估算

计算以下模型配置的显存需求（BF16 训练 + Adam 优化器）：

| 参数 | 值 |
|------|------|
| 参数量 | 13B |
| 序列长度 | 4096 |
| Batch size per GPU | 4 |
| Hidden size | 5120 |
| Layers | 40 |

需要回答：

**(a) 模型参数占多少 GB？**

13B × 2 bytes (BF16) = **26 GB**

**(b) 优化器状态占多少 GB？**

| 组件 | 计算 | 大小 |
|------|------|------|
| FP32 master weights | 13B × 4 | 52 GB |
| Adam m (FP32) | 13B × 4 | 52 GB |
| Adam v (FP32) | 13B × 4 | 52 GB |
| **合计** | 13B × 12 | **156 GB** |

**(c) 至少需要多少块 80GB H100？**

梯度：13B × 4 bytes (FP32) = 52 GB

激活值取决于优化策略（假设 num_heads=40）：

| 场景 | 每层 | 40 层总计 |
|------|------|----------|
| 无优化（保留全部 + attention scores） | ~16 GB | ~651 GB |
| 选择性重计算（重算 attention，保留其他） | ~2.9 GB | ~114 GB |
| 全重计算（只存层输入：s×b×h×2） | ~0.17 GB | ~6.7 GB |

按实际常用的选择性重计算估算：

| 组件 | 大小 |
|------|------|
| BF16 模型参数 | 26 GB |
| 优化器状态 | 156 GB |
| FP32 梯度 | 52 GB |
| 激活值（选择性重计算） | ~114 GB |
| **合计** | **~348 GB** |

至少需要 ⌈348 / 80⌉ = **5 块 H100**

### 练习 2：进程组划分

给定 32 块 GPU，TP=4，PP=2，CP=1，EP=1：

**推导过程：**

Megatron 的 GPU 排布顺序（从内到外）：TP → DP → PP

**(a) DP = 32 / (TP × PP × CP × EP) = 32 / (4×2×1×1) = 4**

各维度在 rank 编号中的步长：
- TP stride = 1（最内层，相邻 GPU）
- DP stride = TP = 4
- PP stride = TP × DP = 4 × 4 = 16

画成 3D 网格（每格 = 一个 TP 组 = 4 块 GPU）：

```
              PP stage 0          PP stage 1
            ┌───────────────┐  ┌───────────────┐
DP replica 0│ g0  g1  g2  g3│  │g16 g17 g18 g19│
DP replica 1│ g4  g5  g6  g7│  │g20 g21 g22 g23│
DP replica 2│ g8  g9  g10 g11│  │g24 g25 g26 g27│
DP replica 3│g12 g13 g14 g15│  │g28 g29 g30 g31│
            └───────────────┘  └───────────────┘
```

**(b) TP 组：横着读每一行的 4 块 GPU**

组大小=4，组数量=32/4=**8 个**

[g0,g1,g2,g3], [g4,g5,g6,g7], [g8,g9,g10,g11], ... ✓

**(c) PP 组：同一 TP rank、同一 DP replica，跨 PP stage（左右对应位置）**

组大小=2，组数量=32/2=**16 个**

[g0, g16], [g1, g17], [g4, g20], [g5, g21], ...

~~[g0,g4]~~ 是错的——g0 和 g4 都在 PP stage 0，只是不同 DP replica。

**(d) DP 组：同一 TP rank、同一 PP stage，跨 DP replica（竖着读同一列）**

组大小=4，组数量=32/4=**8 个**

[g0, g4, g8, g12], [g1, g5, g9, g13], [g16, g20, g24, g28], [g17, g21, g25, g29], ...

~~[g0,g8]~~ 不完整——漏了 g4 和 g12，完整的是 [g0, g4, g8, g12]。

### 练习 3：代码探索

浏览以下文件，回答问题（不需要逐行阅读，只需找到关键信息）：

**1. 打开 `megatron/core/parallel_state.py`，数一数前 140 行定义了多少个全局变量（以 `_` 开头的大写变量）。这些变量分别属于哪几类并行维度？**

共 **49 个**全局变量（`grep "^_[A-Z].*= None"` 计数），分为 5 类（详见第三节 `parallel_state.py` 全局变量详解）：

| 类别 | 数量 | 示例 |
|------|------|------|
| 基础并行组 | 7 | `_TENSOR_MODEL_PARALLEL_GROUP`, `_DATA_PARALLEL_GROUP` |
| 组合并行组 | 10 | `_DATA_PARALLEL_GROUP_WITH_CP`, `_TENSOR_AND_DATA_PARALLEL_GROUP` |
| 专家并行组 | 9 | `_EXPERT_MODEL_PARALLEL_GROUP`, `_EXPERT_DATA_PARALLEL_GROUP` |
| 全局 rank 列表 | 9 | `_PIPELINE_GLOBAL_RANKS`, `_EMBEDDING_GLOBAL_RANKS` |
| 动态覆盖值 + 特殊 | 14 | `_MPU_*`, `_VIRTUAL_PIPELINE_*`, `_GLOBAL_MEMORY_BUFFER` |

**2. 打开 `pretrain_gpt.py`，找到 `__main__` 块（第 399 行），`pretrain()` 函数接收了哪几个参数？每个参数的职责是什么？**

```python
# pretrain_gpt.py 第 412-421 行
pretrain(
    train_valid_test_datasets_provider,            # ① 数据集构建函数
    partial(model_provider, gpt_builder),           # ② 模型构建函数（用 partial 绑定了 gpt_builder）
    ModelType.encoder_or_decoder,                   # ③ 模型类型标识
    forward_step,                                   # ④ 单步前向函数
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},  # ⑤ 默认参数覆盖
    extra_args_provider=add_modelopt_args,          # ⑥ 额外命令行参数注册器
    store=store,                                    # ⑦ 进程内重启的共享 store
    get_embedding_ranks=get_embedding_ranks,        # ⑧ 计算哪些 rank 需要持有 embedding
)
```

这就是"函数注入"设计模式——`pretrain()` 只负责训练循环的编排，具体的模型、数据、前向逻辑全由调用者注入。

**3. 打开 `model_provider.py`，理解 `model_provider()` 函数的签名。`pre_process` 和 `post_process` 参数是做什么用的？**

```python
# model_provider.py 第 24-26 行
def model_provider(
    model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, ...
)
```

`pre_process` 和 `post_process` 控制 **PP 切分时模型的首尾行为**：

| 参数 | 含义 | True 的情况 |
|------|------|------------|
| `pre_process` | 是否计算 embedding 层 | 当前 rank 在 **PP stage 0**（流水线第一段） |
| `post_process` | 是否计算 output head（logits/loss） | 当前 rank 在 **PP 最后一个 stage** |

假设 PP=4，模型被切成 4 段：

```
Stage 0: pre_process=True,  post_process=False  → Embedding + Layer 1-8
Stage 1: pre_process=False, post_process=False  → Layer 9-16
Stage 2: pre_process=False, post_process=False  → Layer 17-24
Stage 3: pre_process=False, post_process=True   → Layer 25-32 + LM Head
```

中间 stage 既不需要 embedding 也不需要 output head，所以两个都是 False，节省显存。

---

## 本课时小结

| 概念 | 一句话总结 |
|------|-----------|
| **显存墙** | 模型参数 + 优化器状态 + 激活值 >> 单卡显存 |
| **算力墙** | 大模型训练需要数千 GPU-年，必须并行 |
| **通信墙** | 多卡协作的通信开销可占 30-50%，需要精心隐藏 |
| **DP** | 数据切分，模型复制 → 提高吞吐量 |
| **TP** | 层内切分 → 解决单层放不下的问题 |
| **PP** | 层间切分 → 解决模型太深的问题 |
| **CP** | 序列切分 → 解决长序列 attention 的显存问题 |
| **EP** | 专家切分 → 解决 MoE 专家太多的问题 |
| **Megatron Core** | 可组合的 GPU 优化积木库 |
| **Megatron Training** | 基于 Core 的参考训练实现 |

---

## 下一课时预告

**课时 2：代码地图与运行流程** —— 我们将从 `pretrain_gpt.py` 出发，逐函数跟踪一次 GPT 训练从启动到第一个 train_step 完成的完整路径，建立对代码流程的详细理解。
