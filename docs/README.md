# docs/

Documentation projet. Les specs et plans de travail datees vivent dans
`plans/`, une entree par feature ou incident.

## Convention

- Un fichier par plan / decision, nomme `YYYY-MM-DD-<slug>.md`.
- Le slug decrit la feature (`ane-hybrid-pipeline`, `fix-oom-training`).
- Les plans contiennent : contexte, objectif, etapes, criteres d'acceptation.

## Plans

| Fichier | Objet |
|---|---|
| `plans/2026-04-12-fix-oom-training.md` | Diagnostic + fix OOM training mlx-lm (wired-limit, mmap, grad-checkpoint, batch). |
| `plans/2026-04-14-ane-hybrid-pipeline.md` | Pipeline hybride Qwen3.5-35B-A3B sur Apple Neural Engine (DeltaNet + Full Attention + MoE) — cf `research/ane-hybrid/`. |
| `plans/2026-04-15-qwen35-122b-macport-training.md` | Training Qwen3.5-122B-A10B-BF16 profil mac-port sur Studio M3 Ultra (LoRA rank 128, 5000 iters, 3.37 epochs) — cf `configs/qwen35-122b-macport.yaml` + `scripts/train_122b_macport.sh`. |
