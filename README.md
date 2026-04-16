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

## World Firsts

### 1. DeltaNet → CoreML/ANE Conversion
First-ever conversion of Gated DeltaNet (linear attention with recurrent state) to CoreML for Apple Neural Engine. No prior work existed — ANEMLL only supports standard transformer attention.

- Chunkwise parallel form expressed as CoreML MIL ops (matmul, cumsum, exp)
- `ct.StateType` for recurrent state persistence between decode steps
- **474 tok/s per layer**, 14.4 tok/s for full 40-layer stack on ANE
- Real Qwen3.5 weights loaded and verified

### 2. 122B MoE BF16 Training on Single Apple Silicon Machine
First documented fine-tuning of a 122B MoE model in BF16 on a single Mac. Previous record: dense 20B on 512 GB.

- Qwen3.5-122B-A10B (10B active params) at 383 GB peak memory
- Required custom MLX fork with 3x Metal buffer limit (499K → 1.5M)
- Val loss 0.497, train loss 0.177 at iter 270

### 3. First Qwen3.5-122B-A10B Opus-Distilled Model
No 122B Opus-distilled model exists on HuggingFace. Jackrong published 9B, 27B, and 35B variants — we created the first 122B.

- Distilled from Claude Opus 4.6 reasoning traces (11,880 examples)
- 5-phase training pipeline: SFT curriculum → SimPO → GRPO → merge → GGUF

## Micro_KIKI — 32 Expert Fleet

Fleet of 32 specialized MoE-LoRA experts on Qwen3.5-4B using [Brainstacks](https://arxiv.org/abs/2604.01152) (null-space projection for zero-forgetting continual learning). Deployable on RTX 4090 24 GB.

**Domains:** 12 coding languages + 10 embedded/hardware + 10 general (reasoning, French, web, etc.)

```bash
# Data pipeline (1.57M raw → 63K deduplicated)
bash scripts/micro_kiki/pipeline_data.sh

# Train all 32 stacks sequentially (~500 steps each)
bash scripts/micro_kiki/train_all_stacks.sh

# Evaluate forgetting matrix
uv run python scripts/micro_kiki/eval_stack.py --all
```

| Phase | Domains | Status |
|-------|---------|--------|
| 1. Foundations | chat-fr, reasoning | Data ready |
| 2. Coding core | python, typescript, cpp, rust | Data ready |
| 3. Coding secondary | html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy | Data ready |
| 4. Technical | embedded, stm32, iot, freecad, platformio, power, emc, dsp, spice-sim, electronics, kicad-pcb | Data ready |
| 5. Applications | web-frontend, web-backend, music-audio, devops, llm-orch | Data ready |
| 6. Complements | math, security | Data ready |

Architecture: 4 experts/stack, rank 16, top-2 routing, rsLoRA scaling. ~250 MB per frozen stack, ~8 GB total for 32 stacks.

## Sonnet-Devstral Pipeline

Fine-tune Devstral 2 123B (dense, 72.2% SWE-bench) pour du coding rapide style Sonnet. Dataset mixte ~18K exemples : traces de raisonnement R1, instructions code, trajectoires agentic SWE. Langages cibles : Python, TypeScript, Rust, Go.

```bash
./scripts/download_devstral.sh datasets   # télécharger les datasets de coding
python scripts/prepare_coding_dataset.py  # préparer 18K exemples filtrés
./scripts/download_devstral.sh model      # télécharger Devstral 2 123B (~250 Go)
python scripts/train_devstral_sonnet.py   # lancer le training LoRA
```

Config : `configs/mlx-lm-devstral2-sonnet.yaml` — LoRA rank 64, 4096 seq, 2000 iters.

See [`research/ane-hybrid/`](research/ane-hybrid/) for ANE research details.

## Models

| Model | Size | Location |
|-------|------|----------|
| Qwen3.5-122B-A10B-BF16 | 233 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-bf16 | 65 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-vlm | — | fusion model (vision tower) |
| Mistral Large 123B | ~250 GB | `models/` |
| Devstral 2 123B (dense) | ~250 GB | `models/` (à télécharger) |

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
