# KIKI-Mac_tunner

Fine-tuning de gros LLMs sur Mac Studio M4 Pro (512 Go RAM unifiée) via MLX.

## Objectif principal

Distiller le raisonnement Claude Opus dans Mistral Large 123B en utilisant la mémoire unifiée Apple Silicon pour un training bf16 complet (pas de quantization) — qualité maximale.

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
│   ├── train_mlx.py      # MLX training script
│   ├── merge_lora.py     # Merge LoRA adapter into base model
│   └── convert_gguf.py   # Convert to GGUF for llama.cpp
└── data/
    └── .gitkeep          # Dataset downloaded here
```
