# Learn: Megatron-LM 代码精读课程大纲（15 课时）

Date: 2026-03-25
Source: Megatron-LM codebase (NVIDIA/Megatron-LM)

## 课程概览

由浅入深、由宏观到微观，覆盖 Megatron-LM 分布式训练框架的核心设计与实现。

---

## 第一阶段：全局视角（课时 1-3）

### 课时 1：为什么需要 Megatron-LM

**目标：** 理解分布式训练的核心挑战，以及 Megatron 的设计哲学

- 单卡训练的瓶颈：显存墙、算力墙、通信墙
- 4 种基本并行策略的直觉：DP / TP / PP / EP 各自解决什么问题
- Megatron Core vs Megatron Training 的定位差异
- 论文导读：*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2020)

**阅读材料：** `README.md`，Megatron-LM 系列论文（2020, 2021, 2024）

---

### 课时 2：代码地图与运行流程

**目标：** 建立代码全局心智模型，能跟踪一次训练从启动到结束的完整路径

- 顶层目录结构：`megatron/core/` vs `megatron/training/` vs `examples/`
- 一次 GPT 训练的完整调用链：

```
pretrain_gpt.py
  → model_provider() + gpt_builders.py    # 构建模型
  → initialize_megatron()                  # 初始化分布式环境
  → pretrain()                             # 进入训练循环
```

- 动手：阅读 `pretrain_gpt.py`（17KB），跟踪每个函数调用
- 理解 `model_provider.py` 如何作为模型工厂

**核心文件：** `pretrain_gpt.py`, `model_provider.py`, `gpt_builders.py`, `megatron/training/training.py`（前 100 行）

---

### 课时 3：并行状态管理 —— 框架的基石

**目标：** 理解 `parallel_state.py` 如何用进程组编织多维并行拓扑

- NCCL 进程组基础：什么是 group、rank、world_size
- Megatron 的多维并行网格：如何将 N 块 GPU 划分为 TP × PP × DP × CP × EP 的网格
- 核心函数精读：
  - `initialize_model_parallel_group()` — 进程组创建
  - `get_tensor_model_parallel_world_size/rank()` — 状态查询
  - `is_pipeline_first_stage()` / `is_pipeline_last_stage()` — 位置判断
- 理解 40+ 全局变量的设计取舍：为什么不用 class 封装

**核心文件：** `megatron/core/parallel_state.py`（2192 行，~107KB）

**练习：** 给定 16 GPU、TP=2、PP=4、DP=2 的配置，手画进程组划分

---

## 第二阶段：并行策略深入（课时 4-7）

### 课时 4：数据并行 —— 梯度同步的艺术

**目标：** 理解 Megatron 如何实现高效的数据并行通信

- 从 PyTorch DDP 到 Megatron 的 `DistributedDataParallel`
- 梯度桶（bucket）策略：`param_and_grad_buffer.py` 如何将参数打包以优化通信
- 通信/计算重叠：为什么先完成的层可以提前 allreduce
- FSDP 集成：`torch_fully_sharded_data_parallel.py`

**核心文件：**
- `megatron/core/distributed/distributed_data_parallel.py`（30KB）
- `megatron/core/distributed/param_and_grad_buffer.py`（59KB）
- `megatron/core/distributed/finalize_model_grads.py`（21KB）

---

### 课时 5：张量并行 —— 层内切分

**目标：** 理解如何将单个矩阵乘法分布到多个 GPU 上

- 列切分 vs 行切分的数学推导：
  - `ColumnParallelLinear`：按输出维度切分 → AllGather 收集激活
  - `RowParallelLinear`：按输入维度切分 → AllReduce 汇总结果
- `VocabParallelEmbedding`：词表切分的特殊处理
- Sequence Parallelism：在非 TP 算子上切分序列维度以节省显存
- 通信原语精读：`mappings.py` 中的 `_CopyToModelParallelRegion`, `_ReduceFromModelParallelRegion`
- Forward 里做 gather/scatter，backward 自动对偶的设计

**核心文件：**
- `megatron/core/tensor_parallel/layers.py`（50KB）— ColumnParallelLinear, RowParallelLinear
- `megatron/core/tensor_parallel/mappings.py`（20KB）— 通信原语
- `megatron/core/tensor_parallel/cross_entropy.py`（9KB）— 词表并行下的交叉熵

**练习：** 画出一个 2-layer MLP 在 TP=2 下的前向传播数据流图

---

### 课时 6：流水线并行 —— 层间切分

**目标：** 理解流水线调度如何最小化气泡开销

- 朴素流水线的问题：气泡比例 = (PP-1) / (PP-1+M)
- GPipe vs 1F1B 调度的对比
- 三种调度函数精读（由简到难）：
  1. `forward_backward_no_pipelining()` — 基线，无流水线
  2. `forward_backward_pipelining_without_interleaving()` — 标准 1F1B
  3. `forward_backward_pipelining_with_interleaving()` — 虚拟流水线（VP），多块交替执行
- P2P 通信：`p2p_communication.py` 中的 `send_forward()`, `recv_forward()` 等
- Microbatch 的概念：为什么要把 batch 拆成小块

**核心文件：**
- `megatron/core/pipeline_parallel/schedules.py`（100KB）
- `megatron/core/pipeline_parallel/p2p_communication.py`（25KB）

**练习：** 绘制 PP=4、microbatch=8 的 1F1B 时间线图

---

### 课时 7：上下文并行与专家并行

**目标：** 理解两种特殊场景下的并行策略

**Part A — Context Parallelism (CP)：**
- 长序列场景的挑战：attention 的 O(n²) 显存
- 序列维度切分：每个 rank 处理序列的一部分
- 层次化 CP：`hybrid_cp_schedule.py`
- 与 Ring Attention 的关系

**Part B — Expert Parallelism (EP)：**
- MoE 基础：Router 选择 Top-K 个专家
- `router.py`：`TopKRouter` 的路由决策
- `token_dispatcher.py`：AllToAll 通信将 token 分发到持有对应专家的 GPU
- `moe_layer.py`：编排 router → dispatch → expert compute → combine
- 负载均衡：辅助损失（auxiliary loss）防止专家坍缩
- Expert Tensor Parallelism：大专家还可以再切分

**核心文件：**
- `megatron/core/pipeline_parallel/hybrid_cp_schedule.py`（28KB）
- `megatron/core/transformer/moe/router.py`（33KB）
- `megatron/core/transformer/moe/token_dispatcher.py`（65KB）
- `megatron/core/transformer/moe/moe_layer.py`（25KB）

---

## 第三阶段：模型构建（课时 8-10）

### 课时 8：配置系统 —— 200+ 参数的管理

**目标：** 理解 Megatron 如何用分层配置驱动整个框架

- `ModelParallelConfig`（`model_parallel_config.py`）：并行策略参数基类
- `TransformerConfig`（继承自 `ModelParallelConfig`）：模型架构 + 训练 + 并行的 200+ 参数
  - 模型参数：`num_layers`, `hidden_size`, `num_attention_heads`, `ffn_hidden_size`
  - 精度参数：`fp16`, `bf16`, `fp8`
  - 并行参数：`tensor_model_parallel_size`, `pipeline_model_parallel_size`
- `__post_init__` 中的复杂验证逻辑
- 命令行参数系统：`megatron/training/arguments.py`（183KB）的 50+ 参数组

**核心文件：**
- `megatron/core/transformer/transformer_config.py`
- `megatron/core/model_parallel_config.py`
- `megatron/training/arguments.py`（参考性阅读，不需逐行）

---

### 课时 9：ModuleSpec —— 声明式模型构建

**目标：** 理解 Megatron 如何解耦模型架构定义与底层实现

- `ModuleSpec` dataclass：声明"这一层用什么类、什么参数"
- `build_module()` 函数：从 spec 实例化模块
- 为什么需要这层抽象：
  - 同一模型架构可以切换后端（PyTorch local → Transformer Engine → 推理优化）
  - 按需组合功能（MoE、MLA 等）
- `gpt_layer_specs.py` 精读：
  - `get_gpt_layer_with_transformer_engine_spec()` — TE 优化版
  - `get_gpt_layer_local_spec()` — 纯 PyTorch 版
  - 对比两者的 spec 定义差异

**核心文件：**
- `megatron/core/transformer/spec_utils.py`（128 行 — 简短但关键）
- `megatron/core/models/gpt/gpt_layer_specs.py`（770 行）

---

### 课时 10：Transformer 内部结构

**目标：** 逐层理解 TransformerBlock → TransformerLayer → Attention/MLP 的组装

- **TransformerBlock**（`transformer_block.py`）：
  - 管理 N 层 TransformerLayer 的堆叠
  - 处理 PP 切分：只实例化属于本 stage 的层
- **TransformerLayer**（`transformer_layer.py`）：
  - 标准结构：LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
  - Pre-norm vs Post-norm 的选择
- **Attention**（`attention.py`）：
  - QKV 投影（可以是 TP 切分的 ColumnParallelLinear）
  - 多种后端：local dot-product、FlashAttention、Transformer Engine
  - Multi-Latent Attention (MLA) 支持
  - RoPE 位置编码的集成
- **MLP**（`mlp.py`）：
  - SwiGLU / GeLU 门控 MLP
  - 与 TP 的配合：第一个线性层列切分，第二个行切分
- **GPTModel**（`gpt_model.py`）：
  - 完整模型 = Embedding + TransformerBlock + Output head
  - `pre_process` / `post_process` 标志控制 PP 边界行为

**核心文件：**
- `megatron/core/transformer/transformer_block.py`
- `megatron/core/transformer/transformer_layer.py`
- `megatron/core/transformer/attention.py`
- `megatron/core/transformer/mlp.py`
- `megatron/core/models/gpt/gpt_model.py`（846 行）

---

### 课时 10.5：CUDA Graph —— 消除 kernel launch 开销

**目标：** 理解 Megatron 如何用 CUDA Graph 将多次 kernel launch 合并为一次 graph replay

- CUDA Graph 基础：为什么 kernel launch overhead 在大规模训练中成为瓶颈
- Megatron 的两种实现（`cuda_graph_impl`）：
  - `"local"`：Megatron 自己的 `CudaGraphManager` + `_CudaGraphRunner`
  - `"transformer_engine"`：通过 TE 的 `make_graphed_callables()` 捕获
- **捕获粒度**（`CudaGraphScope` 枚举）：
  - `full_iteration`：整个训练迭代作为一张图（`FullCudaGraphWrapper`）
  - `attn` / `mlp` / `moe` / `mamba`：按子模块粒度捕获
  - `full_iteration_inference`：推理专用
- 捕获与回放流程：
  1. 前 `cuda_graph_warmup_steps` 步以 eager mode 运行，录制操作
  2. Warmup 结束后捕获图（注册 RNG 状态、分配 memory pool）
  3. 后续迭代直接 replay，跳过 kernel launch
- 约束与陷阱：
  - 静态 shape 要求：输入形状在 capture 后不可变
  - 与 PP 的交互：每个 microbatch 独立的 graph runner
  - 与 FP8/FP4 量化的集成：graph 内的 scaling 状态管理
- `GraphableMegatronModule`：可被图化的模块基类

**核心文件：**
- `megatron/core/transformer/cuda_graphs.py`（34KB）— `CudaGraphManager`, `_CudaGraphRunner`
- `megatron/core/full_cuda_graph.py` — `FullCudaGraphWrapper`, `StaticBufferLoader`
- `megatron/core/transformer/enums.py` — `CudaGraphScope` 枚举
- `megatron/core/transformer/module.py` — `GraphableMegatronModule` 基类

---

## 第四阶段：训练基础设施（课时 11-13）

### 课时 11：训练循环

**目标：** 理解 `pretrain()` 函数内的完整训练流程

- `pretrain()` 的主干逻辑：
  1. 初始化模型、优化器、学习率调度器
  2. 加载 checkpoint（如有）
  3. 进入 `while` 循环：`train_step()` → log → save → eval
- `train_step()` 精读：
  - 梯度累积：多个 microbatch 的前向/反向
  - 调用 PP 调度器（课时 6 的调度函数）
  - 梯度裁剪与 allreduce
  - 优化器 step
- 混合精度训练：FP16/BF16 的 loss scaling 机制
- 初始化流程：`initialize.py` 如何设置分布式环境、随机种子、进程组

**核心文件：**
- `megatron/training/training.py`（~156KB）— 重点读 `pretrain()` 和 `train_step()`
- `megatron/training/initialize.py`（582 行）

---

### 课时 12：优化器与混合精度

**目标：** 理解分布式优化器如何在多维并行下高效更新参数，以及 FP8 低精度训练

**Part A — 优化器：**
- 优化器层次结构：
  - `MegatronOptimizer`（基类）→ `Float16OptimizerWithFloat16Params` → `DistributedOptimizer`
  - `ChainedOptimizer`：组合多个优化器（例如对不同参数组用不同策略）
- `DistributedOptimizer`（124KB）核心思想：
  - 将优化器状态按 DP rank 分片（类似 ZeRO Stage 2）
  - 重叠通信与计算
  - 与 TP/PP 的交互
- 梯度裁剪：`clip_grads.py` — 跨所有并行维度正确计算全局梯度范数
- 动态 Loss Scaling：`grad_scaler.py` — FP16 训练的必需组件
- 新优化器：MUON (`muon.py`)、Lion 等

**Part B — FP8 低精度训练：**
- FP8 格式基础：
  - E4M3（精度优先）vs E5M2（范围优先）：前者用于前向权重/激活，后者用于反向梯度
  - `hybrid` 模式：前向用 E4M3，反向用 E5M2
- Scaling 策略（`Fp8Recipe` 枚举）：
  - `delayed`：维护 AMAX 历史缓冲区，基于历史推断 scaling factor
  - `tensorwise`（CurrentScaling）：每个 tensor 实时计算 scale
  - `blockwise`：按 block 粒度计算 scale，精度更高
  - `mxfp8`：Microscaling FP8，使用 E8M0 格式存储 scale
  - `custom`：通过 `fp8_quantizer_factory` 指定自定义 recipe
- Transformer Engine 集成：
  - `fp8_autocast` 上下文管理器：在 TE 层内启用 FP8 计算
  - `TEDelayedScaling`：Megatron 对 TE recipe 的封装
  - `fp8_param`：将模型参数持久保存为 FP8 以节省显存
  - `first_last_layers_bf16`：首尾层保持 BF16 以稳定训练
- FP8 与分布式训练的交互：
  - AMAX reduction：跨 TP/CP 组同步 scaling 信息（`tp_only_amax_red`）
  - FP8 param gather：分布式优化器在 all-gather 时直接传输 FP8 参数
- FP4 前瞻：`fp4_utils.py` 为 Blackwell 架构提供 NVFP4 支持

**核心文件：**
- `megatron/core/optimizer/optimizer.py`（57KB）
- `megatron/core/optimizer/distrib_optimizer.py`（124KB）
- `megatron/core/optimizer/clip_grads.py`
- `megatron/core/fp8_utils.py`（786 行）— FP8 recipe 构建、量化/反量化、AMAX 管理
- `megatron/core/extensions/transformer_engine.py`（32KB）— TE 集成层、`TEDelayedScaling`
- `megatron/core/enums.py` — `Fp8Recipe` / `Fp4Recipe` 枚举
- `megatron/core/transformer/transformer_config.py` — FP8 相关配置参数（`fp8`, `fp8_recipe`, `fp8_margin` 等）

---

### 课时 13：数据管线

**目标：** 理解大规模训练下的数据加载策略

- `MegatronDataset` 基类：定义 dataset 抽象接口
- `GPTDataset`：自回归语言模型数据集
  - 基于内存映射的 IndexedDataset：`indexed_dataset.py`
  - 高效随机访问，无需全量加载
- `BlendedMegatronDatasetBuilder`：多数据集混合
  - 数据调度：`data_schedule.py` — 不同训练阶段用不同数据比例
- Packed Sequences：将多个短序列打包到一个固定长度序列，提高 GPU 利用率
- Tokenizer 集成：`megatron/core/tokenizers/` 支持 SentencePiece、Tiktoken 等
- 数据预处理工具：`tools/preprocess_data.py`

**核心文件：**
- `megatron/core/datasets/gpt_dataset.py`
- `megatron/core/datasets/blended_megatron_dataset_builder.py`
- `megatron/core/datasets/indexed_dataset.py`

---

## 第五阶段：进阶主题（课时 14-15）

### 课时 14：分布式 Checkpoint

**目标：** 理解如何在改变并行配置后仍能恢复训练

- 核心问题：TP=4 训练保存的 checkpoint，如何用 TP=8 恢复？
- `ShardedTensor` 概念：每个 rank 保存自己那份 shard + 全局元数据
- `mapping.py`：`ShardedStateDict` 描述每个参数的全局形状和本地切片
- `serialization.py`：save/load 的序列化逻辑
- `exchange_utils.py`：resharding 时 rank 之间的数据交换协议
- Megatron Bridge：HuggingFace ↔ Megatron 双向 checkpoint 转换
- 异步保存：训练不停，后台保存 checkpoint

**核心文件：**
- `megatron/core/dist_checkpointing/mapping.py`（21KB）
- `megatron/core/dist_checkpointing/serialization.py`（18KB）
- `megatron/core/dist_checkpointing/exchange_utils.py`（25KB）
- `megatron/training/checkpointing.py`

---

### 课时 15：推理、RLHF 与全局回顾

**目标：** 了解训练之外的能力，并建立完整的架构心智模型

**Part A — 推理优化：**
- `megatron/core/inference/`：推理引擎架构
- KV Cache 管理、Speculative Decoding
- 推理场景下的 CUDA Graph（`full_iteration_inference` scope，回顾课时 10.5）
- MXFP8 推理量化：`inference/quantization/` 中的 Triton kernel 实现
- 推理专用层：`tensor_parallel/inference_layers.py`

**Part B — RLHF / 后训练：**
- `megatron/rl/`：强化学习训练管线
- `megatron/post_training/`：量化、蒸馏、剪枝
- `train_rl.py` 入口

**Part C — 全局回顾：**
- 完整数据流：数据加载 → 分布式采样 → 前向（TP+PP） → 反向 → 梯度同步（DP） → 优化器更新 → checkpoint
- 设计模式总结：
  - 进程组（parallel_state）作为全局基础设施
  - ModuleSpec 实现实现与架构解耦
  - 通信/计算重叠贯穿始终
- 扩展阅读：如何基于 Megatron Core 构建自己的模型

---

## 推荐论文配合阅读

| 课时 | 论文 |
|------|------|
| 1 | *Megatron-LM* (Shoeybi et al., 2020) — 张量并行 |
| 5-6 | *Efficient Large-Scale Language Model Training* (Narayanan et al., 2021) — PP + TP 组合 |
| 6 | *GPipe* (Huang et al., 2019) — 流水线并行基础 |
| 7 | *Switch Transformers* (Fedus et al., 2022) — MoE 入门 |
| 12 | *ZeRO* (Rajbhandari et al., 2020) — 分布式优化器对比 |
| 7 | *Ring Attention* (Liu et al., 2023) — 上下文并行 |
| 12 | *FP8 Formats for Deep Learning* (Micikevicius et al., 2022) — FP8 训练基础 |
| 10.5 | *CUDA Graphs* (NVIDIA docs) — graph capture/replay 机制 |
| 14 | *Reducing Activation Recomputation* (Korthikanti et al., 2022) — 显存优化 |

---

## 学习建议

- **课时 1-3** 是后续一切的基础，务必扎实理解 `parallel_state.py`
- **课时 4-7** 是框架核心价值所在，每种并行策略建议配合画图理解数据流
- **课时 8-10** 代码量大但模式统一，理解 ModuleSpec 后读其他模型都是同一套路
- **课时 11-13** 偏工程实现，适合边读代码边调试
- **课时 14-15** 适合在已掌握前 13 课后选择性深入
