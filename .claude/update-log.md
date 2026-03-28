## 2026-03-28 16:00 — 课时 1 复习与深度扩展

**Lesson-01 讲义更新：**
- 修正思考题答案：混合精度训练需要 FP32 master weights（之前漏算 700 GB）
- 新增：Transformer 完整前向传播流程图（14 步 + shape 标注）
- 新增：每步求导公式 + 需要保存的激活推导（LaTeX 格式）
- 新增：GPU 排布顺序解释（TP→DP→PP，附 3D 网格图）
- 新增：`parallel_state.py` 49 个全局变量分 5 类详解
- 新增：练习 3 代码探索题答案（全局变量分类、pretrain 参数、pre/post_process）
- 修正：练习 2 进程组划分的 PP/DP 组成员错误

**Notes created:**
- [[notes/learn/2026-03-28-nccl-process-group-mechanics.md]] — NCCL ProcessGroup 的实际值、初始化机制、与 rank 列表的配合
- [[notes/learn/2026-03-28-activation-memory-first-principles.md]] — 从 backward 求导公式推导每层需要保存的激活，34sbh + 3~5bas^2 公式来源

**Open threads:**
- 课程大纲待补充 FP8 和 CUDA Graph 专题内容
- 课时 2 讲义待起草

---

## 2026-03-25 — 课时 1 详细讲义

**Notes created:**
- [[notes/learn/2026-03-25-lesson-01-why-megatron.md]] — 课时 1 完整讲义：三面墙、五种并行策略、项目结构、代码调用链预览、论文导读、动手练习

**Open threads:**
- 课时 2-15 讲义待起草

---

## 2026-03-25 — Megatron-LM 代码精读课程大纲

**Notes created:**
- [[notes/learn/2026-03-25-megatron-lm-curriculum.md]] — 15 课时课程大纲，由浅入深覆盖并行策略、模型构建、训练基础设施等核心模块

**Open threads:**
- none
