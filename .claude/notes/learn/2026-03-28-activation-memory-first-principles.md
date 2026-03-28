# Learn: 从反向传播求导推导 Transformer 激活显存

Date: 2026-03-28
Source: 对话推导 + Korthikanti et al. 2022 "Reducing Activation Recomputation in Large Transformer Models"

## What I Studied

从第一性原理推导 Transformer 每层需要保存哪些激活值、为什么需要保存、以及三种优化策略的显存差异。

## Key Takeaways

- **核心原则**：对于 Y = f(X)，反向传播计算 dL/dX 时需要什么输入，就必须在前向时保存什么
- **线性层** Y = XW：backward 需要 X（输入）来算 dL/dW = X^T · dL/dY。权重 W 本身已作为参数存储，不占激活显存
- **Softmax**：导数公式 dL/dS_i = P_i(dL/dP_i - sum_j P_j · dL/dP_j) 只依赖输出 P，不需要输入 S
- **GeLU**：导数 GeLU'(x) = Φ(x) + x·φ(x) 依赖原始输入 x，不能从输出反推 —— 这是为什么 GeLU 输入必须保存（占 8sbh，是最大的单项之一）
- **ReLU 对比**：ReLU'(x) 只需知道输出是否 > 0，可以从输出判断，不需要保存输入
- **Dropout**：backward 需要 mask（1 byte/元素），因为 dL/dx = dL/dy · mask/(1-p)

## Patterns & Techniques Observed

- **每层激活公式**：约 34sbh + 3~5bas^2 bytes（s=序列长度, b=batch, h=hidden, a=num_heads）
- 34sbh 来自线性部分（LayerNorm 输入、QKV、MLP 中间层等约 10 个 tensor）
- 3~5bas^2 来自 attention scores（softmax 输出 P + dropout mask，O(s^2) 是长序列瓶颈）
- **三种策略**：无优化（全保存）→ 选择性重计算（砍掉 s^2 项）→ 全重计算（只存层输入 2sbh，多 30-40% 计算）
- **S vs P 的选择**：标准路径存 P（3bas^2 够用）；某些 recomputation 变体需要从 S 重算 P，此时需存 S（变成 5bas^2）

## Mental Models Formed

推导激活显存的方法论：
1. 画出完整前向传播的计算图（每步标注输出 shape）
2. 对每步写出 backward 的求导公式
3. 检查求导公式依赖哪些量：如果依赖输入（而输入不能从输出恢复），就必须保存
4. 汇总所有需保存的 tensor 大小

"需要保存什么"完全由求导公式决定，不需要死记硬背。

## Apply To

- 估算任意模型架构的激活显存（不限于标准 Transformer）
- 设计新的 activation checkpointing 策略时，知道哪些 tensor 值得重计算（s^2 项收益最大）
- 理解 FlashAttention 为什么能省显存（不 materialize 完整的 s×s attention matrix）
- 选择激活函数时考虑显存影响（GeLU 需存输入，ReLU/SwiGLU 各有不同）
