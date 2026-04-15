# KIKI-Mac_tunner

Fine-tune large language models on Apple Silicon using the full unified memory — no quantization needed.

## What This Does

Distills Claude Opus reasoning into open-source LLMs (Mistral Large 123B, Qwen3.5-122B-A10B) using MLX on a single Mac Studio. BF16 full-precision training with LoRA, enabled by 512 GB unified memory.

Includes an ANE (Apple Neural Engine) research pipeline for hybrid inference and a multi-model distillation workflow.

## Machine

- **Mac Studio M3 Ultra** — 512 GB unified memory
- macOS 15+ (Sequoia)
- MLX with custom 3x Metal buffer limit (499K -> 1.5M buffers) for 122B BF16 training

## Quick Start

```bash
./setup.sh          # install dependencies (mlx, mlx-lm, mlx-tune)
./download.sh       # download model + dataset
./train.sh          # launch training (Ctrl+C saves checkpoint)
./train.sh --resume # resume from last checkpoint
./export.sh         # merge LoRA + convert to GGUF
```

For the 122B MoE model:
```bash
sudo sysctl -w iogpu.wired_limit_mb=458752   # cap Metal at 448 GiB
./scripts/train_122b_macport.sh               # launch 122B training
```

## Training Results

| Model | Method | Val Loss | Train Loss | Status |
|-------|--------|----------|------------|--------|
| Mistral Large 123B | LoRA bf16 | **0.479** | — | Done (iter 1100) |
| Qwen3.5-122B-A10B-Opus-v3 | mlx-tune LoRA bf16 | **0.468** (iter 400) | 0.177 (iter 270) | In progress |

Peak memory for 122B training: 383 GB.

## Inference Benchmarks

| Model | Engine | Throughput |
|-------|--------|------------|
| Qwen3.5-35B-A3B | mlx-vlm native | 45-89 tok/s |
| DeltaNet 40-layer (ANE) | CoreML | 14.4 tok/s (474 tok/s/layer) |
| MLX pure (full model) | MLX | 14.2 tok/s |
| ANE+CPU hybrid | CoreML+MLX | 9.9 tok/s |

## Datasets

| Dataset | Examples |
|---------|----------|
| combined-opus-14k (deduplicated) | 9,813 |
| final-opus-v3-1 | 11,880 train + 626 valid |
| Distilled (123B + 35B + vlm) | ~2,237 |

## ANE Research Highlights

World-first DeltaNet to CoreML ANE conversion:
- Gated DeltaNet layers converted to Conv2d for ANE dispatch
- 474 tok/s per layer on Apple Neural Engine
- Full 40-layer stack: 14.4 tok/s pure ANE
- Verdict: MLX native wins on M3 Ultra (45-89 tok/s via mlx-vlm), ANE useful only when GPU is busy (e.g. during training)

See [`research/ane-hybrid/`](research/ane-hybrid/) for details.

## Models

| Model | Size | Location |
|-------|------|----------|
| Qwen3.5-122B-A10B-BF16 | 233 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-bf16 | 65 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-vlm | — | fusion model (vision tower) |
| Mistral Large 123B | ~250 GB | `models/` |

## Architecture

```
KIKI-Mac_tunner/
├── setup.sh / download.sh / train.sh / export.sh   # main workflow
├── configs/                 # training + generation YAML configs
├── scripts/                 # training, distillation, export scripts
│   ├── train_122b_macport.sh    # 122B MoE training wrapper
│   └── watchdog_mem.sh          # swap-thrash kill switch
├── tools/
│   └── train_monitor_tui.py     # live Rich TUI monitor
├── data/                    # datasets (downloaded)
├── output/                  # checkpoints + LoRA adapters
├── models/                  # downloaded base models
├── lib/
│   └── mlx_lm_fork/        # SSD offload for MoE experts
├── research/
│   └── ane-hybrid/          # ANE + CoreML pipeline research
└── docs/
    └── plans/               # implementation plans
```

## Key Dependencies

- [MLX](https://github.com/ml-explore/mlx) (custom fork with 3x Metal buffer limit)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [mlx-tune](https://github.com/ml-explore/mlx-tune) 0.4.21+
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

## License

MIT
