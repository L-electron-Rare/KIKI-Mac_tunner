# Documentation

## Plans

| Plan | Date | Statut |
|------|------|--------|
| Fix OOM training | 2026-04-12 | Fait |
| ANE hybrid pipeline | 2026-04-14 | Phases 1-3 faites |
| 122B Opus-v3 mlx-tune | 2026-04-15 | En cours |
| Micro_KIKI 32 experts | 2026-04-16 | Plan 1 fait, Plan 2 en cours |

## Datasets disponibles

| Source | Exemples |
|--------|----------|
| Opus 3K original | 2326 |
| Opus 12K SFT | 11673 |
| Combined-opus-14k (deduplique) | 9813 |
| Distille 123B | 87 |
| Distille 35B batch 1+2 | ~2000 |
| Distille mlx-vlm | ~150 |
| **final-opus-v3-1** | **11880 train + 626 valid** |

## Modeles entraines

| Modele | Val loss | Train loss | Checkpoint | Statut |
|--------|----------|------------|------------|--------|
| Mistral Large 123B LoRA | 0.479 | — | iter 1100 | Termine |
| Qwen3.5-122B-A10B Opus v3 | 0.468 (iter 400) | 0.177 (iter 270) | resume run | En cours |

## Benchmarks inference

| Modele | Moteur | tok/s |
|--------|--------|-------|
| Qwen3.5-35B-A3B | mlx-vlm natif | 45-89 |
| DeltaNet 40 couches ANE | CoreML | 14.4 (474/couche) |
| MLX pur (modele complet) | MLX | 14.2 |

## Pipeline Sonnet-Devstral

| Fichier | Rôle |
|---------|------|
| `configs/mlx-lm-devstral2-sonnet.yaml` | Config LoRA pour Devstral 2 123B coding |
| `scripts/download_devstral.sh` | Téléchargement modèle + 7 datasets coding |
| `scripts/prepare_coding_dataset.py` | Fusion, filtrage, dédup → 18K exemples |
| `scripts/train_devstral_sonnet.py` | Training mlx-tune LoRA sur Devstral 2 dense |
| `data/sonnet-coding/` | Dataset final (train.jsonl + valid.jsonl) |

Datasets sources : OpenCodeReasoning (nvidia), OpenCodeInstruct (nvidia), Codeforces-CoTs (open-r1), Magicoder OSS-Instruct, CodeFeedback, OpenHands trajectoires, Nemotron-SWE.

## Micro_KIKI Pipeline

32 experts MoE-LoRA sur Qwen3.5-4B via Brainstacks (null-space projection + residual boosting).

| Etape | Fichiers | Statut |
|-------|----------|--------|
| Data pipeline | `scripts/micro_kiki/classify_parallel.py`, `deduplicate.py`, `split_domains.py` | Fait (63K exemples, 32 domaines) |
| Brainstacks training | `scripts/micro_kiki/train_stack.py`, `eval_stack.py`, `train_all_stacks.sh` | En cours |
| Config | `configs/micro_kiki/brainstacks.yaml`, `domains.yaml` | Fait |
| Plans | `docs/plans/2026-04-15-micro-kiki-plan{1-4}*.md` | Plans 1-4 ecrits |

Datasets : 1.57M raw → 63K dedup (25 sources : CodeFeedback, Glaive-v3, OASST2, French-Alpaca-110K, Trendyol-Cybersec, STM32-HAL, LTspice, etc.)

## Infra cle

- mlx-tune 0.4.21
- MLX fork 3x Metal limit (`/tmp/mlx-fork`) installe dans venv
- llama.cpp CPU+GPU compile dans `/tmp/`
- ANEMLL dans `/tmp/anemll`
- Peak mem training 122B : 383 Go
