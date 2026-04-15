# KIKI-Mac_tunner

Fine-tuning LLMs sur Apple Silicon (512 Go RAM unifiee) via MLX.
Distille le raisonnement Claude Opus dans des modeles open-source.

## Machine

Mac Studio M3 Ultra, 512 Go memoire unifiee. MLX bf16 complet.

## Workflow

```
./setup.sh → ./download.sh → ./train.sh → ./export.sh
```

## Where to Look

| Tache | Emplacement |
|-------|-------------|
| Configs training/generation | `configs/` |
| Scripts (training, distill, export) | `scripts/` |
| Datasets | `data/` |
| Checkpoints et LoRA | `output/` |
| Recherche ANE hybrid | `research/ane-hybrid/` |
| Plans d'implementation | `docs/plans/` |
| Fork mlx_lm (SSD offload) | `lib/mlx_lm_fork/` |
| Fork MLX (3x Metal limit) | `/tmp/mlx-fork` (installe dans venv) |
| Modeles telecharges | `models/` |

## Dataset format

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n\n..."}]}
```

## Anti-Patterns

- Ne pas fine-tuner en 4-bit — bf16 est gratuit avec 512 Go
- Ne pas utiliser PyTorch MPS — MLX est 3-5x plus rapide
- Ne pas oublier `--resume` apres un Ctrl+C
- `huggingface-cli` deprecated → utiliser `hf`
- `mlx_lm.convert --dtype bf16` → `--dtype bfloat16`
- Pour le 122B : utiliser `mlx-tune` (0.4.21+), pas `mlx_lm.lora` directement
- MLX stock limite les Metal buffers a 499K → le fork 3x (`/tmp/mlx-fork`) est requis pour 122B bf16
- `iogpu.wired_limit_mb=458752` obligatoire avant training 122B (sinon OOM kernel)
