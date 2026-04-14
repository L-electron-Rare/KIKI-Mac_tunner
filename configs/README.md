# configs/

Configurations YAML pour le fine-tuning mlx-lm (LoRA, bf16) et la generation
de donnees synthetiques sur Apple Silicon (Studio M3 Ultra 512 Go, M4 Pro).

Voir aussi `CLAUDE.md` dans ce dossier pour les notes de setup detaillees.

## Convention de nommage

- `mlx-lm-<modele>.yaml` : config de training mlx-lm (`mlx_lm.lora --config`).
- `<modele>-macport.yaml` : variante profil "mac-port" (mmap, grad-checkpoint, wired-limit tune).
- `generation/<modele>.yaml` : parametres d'inference/generation teacher (temperature, max_tokens, thinking).
- Fichiers `.bak*` = snapshots d'anciennes variantes (ignorer).

## Fine-tuning mlx-lm

| Fichier | Modele / cible |
|---|---|
| `mlx-lm-mistral-large.yaml` | Mistral Large 123B — LoRA bf16 |
| `mlx-lm-devstral2-123b.yaml` | Devstral v2 123B — LoRA |
| `mlx-lm-deepseek-r1-distill-32b.yaml` | DeepSeek R1 Distill 32B |
| `mlx-lm-deepseek-r1-distill-70b.yaml` | DeepSeek R1 Distill 70B |
| `mlx-lm-qwq-32b.yaml` | Qwen QwQ 32B reasoning |
| `mlx-lm-qwen3-72b.yaml` | Qwen3 72B dense |
| `mlx-lm-qwen3-235b.yaml` | Qwen3 235B MoE |
| `mlx-lm-qwen35-122b.yaml` | Qwen3.5-122B-A10B MoE |
| `mlx-lm-qwen35-397b.yaml` | Qwen3.5 397B (plan) |
| `mlx-lm-qwen35-27b-opus.yaml` | Qwen3.5-27B Opus reasoning |
| `mlx-lm-qwen35-35b-opus.yaml` | Qwen3.5-35B-A3B-Opus MoE |
| `mlx-lm-qwen35-35b-opus-14k.yaml` | Qwen3.5-35B-A3B-Opus contexte 14k |
| `mlx-lm-qwen35-35b-opus-final.yaml` | Qwen3.5-35B-A3B-Opus — run final |
| `qwen35-122b-macport.yaml` | Qwen3.5-122B-A10B-BF16 — profil mac-port (rank 128, 5000 iters, cf `docs/plans/2026-04-15-qwen35-122b-macport-training.md`) |

## Pre-mlx (Unsloth / HF)

| Fichier | Usage |
|---|---|
| `mistral-large.yaml` | Legacy Unsloth |
| `mistral-small.yaml` | Legacy Unsloth |
| `qwen-27b.yaml` | Legacy HF 27B |

## Generation (teachers)

Configs de decodage pour les teachers (voir `scripts/generate_data.py`,
`scripts/distill_generate.py`) — champs `max_tokens`, `temperature`,
`enable_thinking`, `estimated_vram_gb`.

Modeles couverts : DeepSeek R1 671B + 70B, Qwen3 72B/235B, Qwen3.5 27B/35B/122B/397B.
