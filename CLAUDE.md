# KIKI-Mac_tunner

Fine-tuning de LLMs sur Apple Silicon (512 Go RAM unifiee) via MLX.
Distille le raisonnement Claude Opus dans Mistral Large 123B, Mistral Small 24B, ou Qwen3.5-27B.

## Machine

M4 Pro, 512 Go memoire unifiee. MLX bf16 complet, ~800 GB/s bande passante.

## Workflow

```
./setup.sh → ./download.sh → ./train.sh → ./export.sh
```

| Script | Role |
|--------|------|
| `setup.sh` | Install venv + MLX + deps |
| `download.sh` | Fetch modele + dataset depuis HF |
| `train.sh` | Training LoRA (pause/resume via Ctrl+C + `--resume`) |
| `export.sh` | Merge LoRA → GGUF → quantize Q6_K/Q8_0 |

## Where to Look

| Tache | Emplacement |
|-------|-------------|
| Ajouter/modifier un modele cible | `configs/` |
| Modifier le pipeline de training | `scripts/train_mlx.py` |
| Modifier merge/export | `scripts/merge_lora.py`, `scripts/convert_gguf.py` |
| Dataset | `data/` (JSONL chat format) |
| Checkpoints et LoRA final | `output/<model-name>/` |
| Generation configs | `configs/generation/` |
| Data generation script | `scripts/generate_data.py` |
| Dataset merger | `scripts/merge_datasets.py` |
| Teacher model downloads | `scripts/download_teachers.sh` |
| Dataset downloads | `scripts/download_datasets.sh` |
| Dataset conversion | `scripts/convert_datasets.py` |
| Conversion MLX | `scripts/convert_to_mlx.sh` |
| Combined dataset prep | `scripts/prepare_combined_dataset.sh` |
| Generation CPU (llama.cpp) | `scripts/generate_cpu.sh`, `scripts/generate_data_cpu.py` |
| Modeles GGUF | `scripts/download_gguf.sh`, `models/gguf/` |

## Dataset format

Le dataset Opus utilise `problem/thinking/solution`, formate en chat :
```
user: {problem}
assistant: <thinking>{thinking}</thinking>\n\n{solution}
```

## Reseau

GGUF exportes → NFS `tank/models/` (kx6tm-23), accessibles par kxkm-ai (4090), Tower, Cils.

## Multi-Model Pipeline

### Phase 1: Telecharger et preparer le dataset elargi
```
./scripts/download_datasets.sh all                    # 3K + 10K + 12K Opus 4.6
./scripts/prepare_combined_dataset.sh                  # Convertir, fusionner, dedupliquer → ~20K exemples
```

### Phase 1b (optionnel): Generer des donnees supplementaires avec teachers
```
./scripts/download_teachers.sh teachers-2026           # Qwen3.5-397B + 122B
./generate.sh configs/generation/qwen35-122b.yaml --num-problems 5000
python scripts/merge_datasets.py \
  --sources combined-opus-20k generated-qwen35-122b \
  --output combined-opus-25k --deduplicate
```

### Phase 2: Fine-tune le 123B sur le dataset elargi
```
mlx_lm.lora --config configs/mlx-lm-mistral-large.yaml  # Mettre data: data/combined-opus-20k
```

### Phase 3: Comparer avec d'autres students
```
mlx_lm.lora --config configs/mlx-lm-qwen35-122b.yaml    # MoE 122B thinking
mlx_lm.lora --config configs/mlx-lm-devstral2-123b.yaml  # Dense 123B code
mlx_lm.lora --config configs/mlx-lm-qwen35-27b-opus.yaml # 27B deja Opus
```

## Teacher Models (data generation)

| Model | Size | VRAM | Config |
|-------|------|------|--------|
| DeepSeek-R1 671B | 671B MoE | ~335 GB (4-bit) | `configs/generation/deepseek-r1-671b.yaml` |
| Qwen3-235B-A22B | 235B MoE | ~200 GB (bf16) | `configs/generation/qwen3-235b.yaml` |
| Qwen3-72B | 72B | ~145 GB (bf16) | `configs/generation/qwen3-72b.yaml` |
| DeepSeek-R1-Distill-70B | 70B | ~140 GB (bf16) | `configs/generation/deepseek-r1-distill-70b.yaml` |
| Qwen3.5-397B-A17B | 397B MoE | ~350 GB (bf16) | `configs/generation/qwen35-397b.yaml` | Fev 2026 |
| Qwen3.5-122B-A10B | 122B MoE | ~130 GB (bf16) | `configs/generation/qwen35-122b.yaml` | Fev 2026 |
| Qwen3.5-27B-Opus-v2 | 27B | ~56 GB (bf16) | `configs/generation/qwen35-27b-opus.yaml` | Mars 2026 |
| Qwen3.5-35B-A3B-Opus | 35B MoE (3B actifs) | ~70 GB (bf16) | `configs/generation/qwen35-35b-opus.yaml` | Mars 2026 |

## Student Models (fine-tuning)

| Model | Size | VRAM Training | Config |
|-------|------|---------------|--------|
| Mistral Large | 123B | ~270 GB | `configs/mlx-lm-mistral-large.yaml` |
| Qwen3-72B | 72B | ~180 GB | `configs/mlx-lm-qwen3-72b.yaml` |
| DeepSeek-R1-Distill-70B | 70B | ~175 GB | `configs/mlx-lm-deepseek-r1-distill-70b.yaml` |
| QwQ-32B | 32B | ~100 GB | `configs/mlx-lm-qwq-32b.yaml` |
| DeepSeek-R1-Distill-32B | 32B | ~100 GB | `configs/mlx-lm-deepseek-r1-distill-32b.yaml` |
| Qwen3-235B-A22B | 235B MoE | ~350 GB | `configs/mlx-lm-qwen3-235b.yaml` |
| Qwen3.5-397B-A17B | 397B MoE | ~400 GB | `configs/mlx-lm-qwen35-397b.yaml` | Fev 2026 |
| Qwen3.5-122B-A10B | 122B MoE | ~180 GB | `configs/mlx-lm-qwen35-122b.yaml` | Fev 2026 |
| Qwen3.5-27B-Opus-v2 | 27B | ~80 GB | `configs/mlx-lm-qwen35-27b-opus.yaml` | Mars 2026 |
| Devstral 2 123B | 123B dense | ~270 GB | `configs/mlx-lm-devstral2-123b.yaml` | Dec 2025 |
| Qwen3.5-35B-A3B-Opus | 35B MoE (3B actifs) | ~90 GB | `configs/mlx-lm-qwen35-35b-opus.yaml` | Mars 2026 |

## Datasets Opus 4.6

| Dataset | Taille | HuggingFace ID | Script |
|---------|--------|----------------|--------|
| Reasoning 3K (actuel) | 2326 ex | `nohurry/Opus-4.6-Reasoning-3000x-filtered` | — |
| Opus 10K | ~10000 ex | `Roman1111111/claude-opus-4.6-10000x` | `download_datasets.sh opus-10k` |
| Reasoning SFT 12K | ~12000 ex | `ykarout/Opus-4.6-reasoning-sft-12k` | `download_datasets.sh opus-12k` |
| **Combine** | **~20K ex** | — | `prepare_combined_dataset.sh` |

## Inference Parallele

Pendant le training MLX (GPU), on peut generer des donnees en parallele :

| Unite | Framework | Usage |
|-------|-----------|-------|
| GPU Metal (76 cores) | MLX | Training LoRA |
| CPU (24 cores) | llama.cpp `--ngl 0` | Inference teacher GGUF |
| Neural Engine (32 cores) | CoreML | Inference teacher quantifie |

### Generation CPU (pendant le training)
```
./scripts/download_gguf.sh qwen35-35b-opus                    # GGUF Q4 (~20 Go)
./scripts/generate_cpu.sh models/gguf/Qwen3.5-35B-A3B-Opus-Q4/*.gguf \
    data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl \
    generated-cpu-35b \
    5000
```

### Builds llama.cpp
- CPU only : `/tmp/llama-cpp-cpu/build/bin/llama-cli`
- GPU Metal : `/tmp/llama-cpp-gpu/build/bin/llama-cli`

## Anti-Patterns

- Ne pas fine-tuner en 4-bit — bf16 est gratuit avec 512 Go
- Ne pas utiliser PyTorch MPS — MLX est 3-5x plus rapide sur M4
- Ne pas oublier `--resume` apres un Ctrl+C
