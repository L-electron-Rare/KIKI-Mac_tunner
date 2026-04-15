# Documentation

## Plans

| Plan | Date | Statut |
|------|------|--------|
| Fix OOM training | 2026-04-12 | Fait |
| ANE hybrid pipeline | 2026-04-14 | Phases 1-3 faites |
| 122B Opus-v3 mlx-tune | 2026-04-15 | En cours |

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

## Infra cle

- mlx-tune 0.4.21
- MLX fork 3x Metal limit (`/tmp/mlx-fork`) installe dans venv
- llama.cpp CPU+GPU compile dans `/tmp/`
- ANEMLL dans `/tmp/anemll`
- Peak mem training 122B : 383 Go
