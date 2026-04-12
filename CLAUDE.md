# KIKI-Mac_tunner

Fine-tuning de LLMs sur Mac Studio M4 Pro (512 Go RAM unifiée) via MLX.
Distille le raisonnement Claude Opus dans Mistral Large 123B, Mistral Small 24B, ou Qwen3.5-27B.

## Cible matérielle

Apple Silicon M4 Pro, 512 Go mémoire unifiée.
MLX exploite la bande passante mémoire (~800 GB/s) pour du training bf16 complet sans quantization.

## Workflow

```
./setup.sh → ./download.sh → ./train.sh → ./export.sh
```

| Script | Rôle |
|--------|------|
| `setup.sh` | Install venv + MLX + deps |
| `download.sh` | Fetch modèle + dataset depuis HF |
| `train.sh` | Training LoRA (pause/resume via Ctrl+C) |
| `export.sh` | Merge LoRA → GGUF → quantize Q6_K/Q8_0 |

## Configs

`configs/*.yaml` — un fichier par modèle cible. Champs clés :
- `model_id` : repo HuggingFace du modèle base
- `dataset_id` : repo HuggingFace du dataset (default: `nohurry/Opus-4.6-Reasoning-3000x-filtered`)
- `lora_rank`, `lora_alpha` : dimensionnement LoRA (ratio alpha/rank = 2 standard)
- `precision` : toujours `bf16` sur Apple Silicon
- `output_dir` : où les checkpoints et le LoRA final sont sauvés

## Dataset format

Le dataset Opus utilise `problem/thinking/solution`. Le script formate en chat :
```
user: {problem}
assistant: <thinking>{thinking}</thinking>\n\n{solution}
```

## Réseau de machines (contexte)

Les modèles GGUF exportés vont sur le NFS `tank/models/` de kx6tm-23, accessibles par :
- kxkm-ai (GPU RTX 4090) via `/mnt/models/`
- Tower et Cils via NFS/SMB

## Anti-Patterns

- Ne pas fine-tuner en 4-bit sur Apple Silicon — bf16 est gratuit avec 512 Go
- Ne pas utiliser PyTorch MPS — MLX est 3-5× plus rapide sur M4
- Ne pas augmenter batch_size au-delà de 2 pour les modèles >100B
- Ne pas oublier `--resume` après un Ctrl+C
