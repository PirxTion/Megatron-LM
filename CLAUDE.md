# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Megatron-LM is NVIDIA's framework for efficient large-scale distributed training of transformer-based models. It has two components:
- **Megatron Core** (`megatron/core/`) — composable library with GPU-optimized building blocks
- **Megatron Training** (`megatron/training/`) — reference training implementation

## Build & Install

```bash
# Install from source (editable)
pip install -e .

# With training extras (wandb, sentencepiece, transformers, etc.)
pip install -e ".[training]"

# With dev extras (transformer-engine, mamba, modelopt, etc.)
pip install -e ".[dev]"

# If build runs out of memory
MAX_JOBS=4 pip install -e .
```

The project uses `uv` as its primary package manager. C++ extensions (`megatron.core.datasets.helpers_cpp`) are built via `setup.py` with pybind11.

**Python >=3.10, torch >=2.6.0 required.**

## Testing

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run a single test file
pytest tests/unit_tests/test_utilities.py

# Run a specific test
pytest tests/unit_tests/test_utilities.py::TestUtilities::test_something -x
```

Pytest config is in `pyproject.toml`. Default options: `--durations=15 -s -rA -x` (stop on first failure). Markers: `internal`, `flaky`, `flaky_in_dev`.

Many tests require CUDA GPUs and distributed environments. The CI runs on A100/H100/GB200 clusters.

## Linting & Formatting

Linting applies only to `megatron/core/` and `tests/unit_tests/`. The autoformat script runs all tools on changed files vs main:

```bash
# Auto-format changed files
bash tools/autoformat.sh

# Check only (CI mode)
CHECK_ONLY=true bash tools/autoformat.sh
```

Individual tools:
```bash
# Black (line_length=100, skip string normalization, skip magic trailing comma)
black --skip-magic-trailing-comma --skip-string-normalization <files>

# isort (black-compatible profile)
isort <files>

# pylint
pylint <files>

# ruff
ruff check --fix <files>
```

Pre-commit hooks enforce black, pylint, isort on `megatron/core/` (and black on `tests/unit_tests/`).

## Architecture

### Parallelism (the core abstraction)

Megatron implements 5 dimensions of parallelism, all managed by `megatron/core/parallel_state.py`:

- **Tensor Parallelism (TP)** — splits individual layers across GPUs (`megatron/core/tensor_parallel/`)
- **Pipeline Parallelism (PP)** — splits model depth across GPUs with 1F1B scheduling (`megatron/core/pipeline_parallel/`)
- **Data Parallelism (DP)** — splits batches across GPUs (`megatron/core/distributed/`)
- **Context Parallelism (CP)** — splits sequence dimension for long contexts
- **Expert Parallelism (EP)** — distributes MoE experts across GPUs (`megatron/core/transformer/moe/`)

`parallel_state.py` (~90KB) maintains 40+ process group variables and is the single source of truth for rank/group queries across all dimensions.

### Model definitions

Models are defined via **ModuleSpec** (`megatron/core/transformer/spec_utils.py`) — a declarative system for specifying layer composition. This enables swapping between local PyTorch, Transformer Engine, and inference-optimized implementations without changing model code.

Key specs are in `megatron/core/models/gpt/gpt_layer_specs.py`. Models: GPT, BERT, T5, Mamba (SSM), Vision, Multimodal (VLM).

### Transformer stack

- **TransformerConfig** (`megatron/core/transformer/transformer_config.py`) — 200+ parameter dataclass for all model/parallelism/training settings
- **TransformerBlock** (`megatron/core/transformer/transformer_block.py`) — stacks transformer layers with PP support
- **TransformerLayer** (`megatron/core/transformer/transformer_layer.py`) — single block: attention + MLP + layernorm
- **Attention** (`megatron/core/transformer/attention.py`) — dot-product, multi-latent (MLA), multiple backends
- **MegatronModule** — base class for all modules, adds checkpoint support

### Training flow

```
pretrain_*.py → model_provider() → initialize_megatron() → pretrain()
                                                              ↓
                                          training loop (megatron/training/training.py)
                                          with gradient accumulation, loss scaling,
                                          distributed optimizer step, checkpointing
```

- `megatron/training/arguments.py` (~183KB) — comprehensive CLI argument parsing with 50+ groups
- `megatron/training/training.py` (~156KB) — main training loop
- `megatron/training/checkpointing.py` — checkpoint save/load
- `megatron/core/dist_checkpointing/` — distributed checkpoint framework supporting resharding across TP/PP changes

### Data pipeline

`megatron/core/datasets/` provides GPTDataset, BertDataset, T5Dataset with:
- Efficient disk-indexed datasets
- Multi-dataset blending
- Packed sequences for SFT
- Variable-length sequence handling with context parallelism

### Entry points

| Script | Purpose |
|--------|---------|
| `pretrain_gpt.py` | GPT training |
| `pretrain_bert.py` | BERT training |
| `pretrain_t5.py` | T5 training |
| `pretrain_mamba.py` | Mamba SSM training |
| `pretrain_vlm.py` | Vision-Language model training |
| `train_rl.py` | RLHF training |
| `tools/preprocess_data.py` | Data preprocessing |
| `tools/run_text_generation_server.py` | Inference server |
