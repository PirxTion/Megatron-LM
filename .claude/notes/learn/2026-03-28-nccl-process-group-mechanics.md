# Learn: NCCL ProcessGroup 在 Megatron 中的工作机制

Date: 2026-03-28
Source: megatron/core/parallel_state.py + 对话讨论

## What I Studied

Megatron 的 `parallel_state.py` 中全局变量（如 `_TENSOR_MODEL_PARALLEL_GROUP`）的实际值是什么，以及 NCCL 进程组如何初始化和使用。

## Key Takeaways

- `_*_GROUP` 变量存的是 `torch.distributed.ProcessGroup` 对象，本质是 NCCL communicator 的不透明封装
- ProcessGroup 本身不暴露"组内有哪些 rank"的信息，所以需要额外的 `_*_GLOBAL_RANKS` 列表做 local rank → global rank 映射
- 初始化时 **所有 GPU 必须集体调用 `new_group()`**（NCCL 要求），但每个 GPU 只保存自己所在的那一个组：

```python
for ranks in all_tp_groups:          # 遍历所有 TP 组
    group = new_group(ranks)         # 全部 GPU 都要调用（集体操作）
    if my_rank in ranks:
        _TENSOR_MODEL_PARALLEL_GROUP = group  # 只有组内成员保存
```

- 使用时传给通信 API 的 `group` 参数决定了通信范围：

```python
# 只跟同 TP 组的 GPU 做 allreduce
torch.distributed.all_reduce(tensor, group=_TENSOR_MODEL_PARALLEL_GROUP)
```

## Patterns & Techniques Observed

- **不透明句柄 + 辅助查表**模式：ProcessGroup 是句柄，rank 列表是查表。两者配合使用
- **集体初始化**：即使 GPU 不在某个组里，也必须参与创建该组（NCCL 协议要求）
- **`_MPU_*` 覆盖值**：运行时 mock 并行大小而不重建进程组，典型用于推理时 TP=1 覆盖训练时的 TP=8

## Mental Models Formed

可以把 ProcessGroup 理解为一条专用通信管道。`new_group([4, 5])` 就是在 g4 和 g5 之间铺了一条管道。通信时指定管道（group 参数），数据只在管道内流动。全局变量存的是"我连着哪条管道"的引用。

## Apply To

- 理解 Megatron 中任何分布式通信代码时，先确认用的是哪个 group 变量，就知道通信范围
- 调试分布式问题时，检查 group 是否正确对应预期的 GPU 集合
- 自定义并行策略时需要创建新的 process group 并遵循集体初始化规则
