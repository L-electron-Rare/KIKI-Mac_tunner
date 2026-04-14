# KIKI-Mac_tunner

Fine-tuning de gros LLMs sur Mac Studio M4 Pro (512 Go RAM unifiée) via MLX.

## Objectif principal

Distiller le raisonnement Claude Opus dans Mistral Large 123B en utilisant la mémoire unifiée Apple Silicon pour un training bf16 complet (pas de quantization) — qualité maximale. Supporte également le fine-tuning LoRA de Qwen3.5-122B-A10B (MoE hybride attention + Mamba/SSM) via le mac-port (cf. section dédiée ci-dessous).

## Prérequis

- Mac Studio M4 Pro avec **512 Go de RAM**
- macOS 15+ (Sequoia)
- Homebrew, Python 3.12+

## Installation rapide

```bash
./setup.sh
```

## Usage

```bash
# Télécharger le modèle + dataset
./download.sh

# Lancer le fine-tuning
./train.sh

# Pauser : Ctrl+C (checkpoint sauvegardé automatiquement)
# Reprendre :
./train.sh --resume

# Quand c'est fini : merger et convertir en GGUF
./export.sh
```

## Modèles supportés

| Modèle | RAM nécessaire | Précision | Temps estimé |
|--------|---------------|-----------|-------------|
| Mistral Large 123B | ~300 GB (bf16) | bf16 full | ~1-2 jours |
| Mistral Large 123B | ~80 GB (4-bit) | QLoRA | ~8-16h |
| Mistral Small 24B | ~50 GB (bf16) | bf16 full | ~4-8h |
| Qwen3.5-27B | ~55 GB (bf16) | bf16 full | ~4-8h |
| Qwen3.5-122B-A10B (MoE hybrid) | ~340 GB peak (bf16) | bf16 LoRA r128 | ~25-37h for 5000 iters |

## Structure

```
KIKI-Mac_tunner/
├── setup.sh              # Install dependencies
├── download.sh           # Download model + dataset
├── train.sh              # Training launcher (pause/resume)
├── export.sh             # Merge LoRA + convert to GGUF
├── configs/
│   ├── mistral-large.yaml    # Mistral Large 123B config
│   ├── mistral-small.yaml    # Mistral Small 24B config
│   └── qwen-27b.yaml        # Qwen3.5-27B config
├── scripts/
│   ├── train_mlx.py
│   ├── train_122b_macport.sh  # mac-port 122B training wrapper
│   ├── watchdog_mem.sh        # memory watchdog with swap-thrash kill
│   ├── merge_lora.py
│   └── convert_gguf.py
├── tools/
│   └── train_monitor_tui.py   # live training monitor (rich TUI)
├── docs/
│   └── plans/                  # design docs, per-feature plans
└── data/
    └── .gitkeep          # Dataset downloaded here
```

## Qwen3.5-122B-A10B mac-port training

Configuration dédiée pour le MoE hybride Qwen3.5-122B (13 self_attn + 36 linear_attn + 256 MoE experts + shared_expert) :

- Config : `configs/qwen35-122b-macport.yaml`
- Wrapper : `scripts/train_122b_macport.sh` (vérifie wired-limit, lance watchdog, mlx_lm.lora)
- Watchdog : `scripts/watchdog_mem.sh` (kill-switch si swap > 80 GB sustained)
- TUI monitoring : `tools/train_monitor_tui.py` (rich : progress, loss, mem, rate)

Voir [docs/plans/2026-04-15-qwen35-122b-macport-training.md](docs/plans/2026-04-15-qwen35-122b-macport-training.md) pour la documentation complète : architecture, rationale LoRA keys, bugs rencontrés, troubleshooting.

**Prérequis** : `sudo sysctl -w iogpu.wired_limit_mb=458752` (plafonne Metal à 448 GiB).

**Lancer** : `./scripts/train_122b_macport.sh`  puis `ssh -t studio "cd KIKI-Mac_tunner && .venv/bin/python tools/train_monitor_tui.py --total-iters 5000"` pour monitorer.
