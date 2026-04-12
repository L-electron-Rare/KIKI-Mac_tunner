# Scripts

Trois scripts Python formant le pipeline post-training.

## train_mlx.py (principal)

Training LoRA via MLX. Lit un config YAML, charge le modèle, applique LoRA, entraîne.

- Entrée : `--config configs/xxx.yaml`, `--resume` optionnel
- Sortie : checkpoints dans `output/<name>/`, LoRA final dans `output/<name>/final-lora/`
- Le dataset est formaté en JSONL chat (`messages[]`) dans `data/`
- Checkpoint auto via mlx-lm trainer, reprise par détection du dernier `checkpoint-*`

## merge_lora.py

Merge LoRA adapter dans le modèle base → safetensors complet.

- Utilise `mlx_lm.tuner.lora.dequantize_and_merge`
- Copie tokenizer + config depuis le modèle base
- Sortie : répertoire safetensors prêt pour conversion GGUF

## convert_gguf.py

Conversion safetensors → GGUF + quantization.

- Clone llama.cpp si absent (auto)
- Étapes : safetensors → F16 GGUF → Q6_K/Q8_0 (configurable via `--quants`)
- Le F16 intermédiaire peut être supprimé après quantization

## Anti-Patterns

- Ne pas modifier les scripts sans mettre à jour les configs YAML correspondants
- Ne pas hardcoder des chemins — tout passe par les arguments et le config
- Ne pas lancer merge_lora.py avant que le training soit terminé (vérifier `final-lora/`)
- Ne pas supprimer le F16 avant d'avoir vérifié les quants
