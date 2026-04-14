# Configs

Un YAML par modele cible. Deux types de configs coexistent.

## Custom configs (principal)

Fichiers : `mistral-large.yaml`, `mistral-small.yaml`, `qwen-27b.yaml`
Lus par `scripts/train_mlx.py`.

### Champs obligatoires

| Champ | Role |
|-------|------|
| `model_id` | Repo HuggingFace du modele base |
| `dataset_id` | Repo HuggingFace du dataset |
| `lora_rank` / `lora_alpha` | Dimensionnement LoRA (ratio alpha/rank = 2) |
| `precision` | Toujours `bf16` sur Apple Silicon |
| `batch_size` | Taille de batch (1 pour >100B, 2 max sinon) |
| `memory_limit_gb` | Plafond Metal — 400 pour 123B, ~100 pour 24-27B |

## mlx-lm native config

Fichier : `mlx-lm-mistral-large.yaml`
Format natif `mlx_lm` CLI avec `model:`, `data:`, `fine_tune_type:`.
Alternative au custom pipeline si on veut utiliser `mlx_lm.lora` directement.

## Contraintes memoire (>100B)

- `batch_size: 1` obligatoire
- `lora_rank` <= 48 recommande (96 a cause des OOM en validation)
- `memory_limit_gb` <= 400 (laisser marge pour le systeme)
- `gradient_accumulation_steps` compense le petit batch

## Generation configs (`generation/`)

Configs pour la generation de donnees synthetiques avec des modeles teachers.
Lus par `generate.sh` et `scripts/generate_data.py`.

### Champs

| Champ | Role |
|-------|------|
| `model` | Chemin local du modele teacher |
| `precision` | `bf16` ou `4bit` |
| `estimated_vram_gb` | Estimation VRAM pour planifier le chargement |
| `max_tokens` | Tokens max par generation (8192 typique) |
| `temperature` | Temperature de sampling (0.7 par defaut) |
| `top_p` | Top-p sampling (0.95 par defaut) |
| `enable_thinking` | Active le mode thinking natif (Qwen3) |
| `thinking_tag` / `thinking_close_tag` | Tags de raisonnement du modele (`<think>`/`</think>`) |
| `output_format` | Format de sortie (`thinking` = trace de raisonnement) |
| `output_name` | Nom du dossier de sortie dans `data/` |

Les tags `<think>...</think>` sont normalises en `<thinking>...</thinking>` par le script de generation.

## Conversion MLX

Certains modeles ne sont pas disponibles en format MLX pre-converti.
Utiliser `scripts/convert_to_mlx.sh` pour les convertir depuis HuggingFace.

| Modele | Commande | Taille sortie |
|--------|----------|---------------|
| Qwen3.5-35B-A3B-Opus | `./scripts/convert_to_mlx.sh qwen35-35b-opus` | ~70 Go |
| Qwen3.5-397B-A17B (bf16) | `./scripts/convert_to_mlx.sh qwen35-397b` | ~794 Go |
| Qwen3.5-397B-A17B (4-bit) | `./scripts/convert_to_mlx.sh qwen35-397b-4bit` | ~222 Go |

## Anti-Patterns

- Ne pas monter batch_size au-dela de 2 pour les modeles >100B — OOM Metal garanti
- Ne pas oublier `memory_limit_gb` — sans lui MLX prend toute la RAM et crash
- Ne pas modifier un config pendant un training en cours
- Garder le ratio `lora_alpha / lora_rank = 2`
