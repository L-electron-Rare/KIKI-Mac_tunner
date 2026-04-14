# research/

Notes et prototypes de recherche exploratoire, hors du scope d'un plan
feature precis. Code souvent experimental, non integre a la pipeline prod.

## Sous-dossiers

| Dossier | Sujet |
|---|---|
| `ane-hybrid/` | Portage hybride Qwen3.5-35B-A3B (MoE DeltaNet + Full Attention) sur Apple Neural Engine via ANEMLL + CoreML. Contient la reference PyTorch DeltaNet, la variante Conv2d ANE-friendly, la conversion CoreML, et les pipelines Phase 2 (stack 40 couches CoreML) / Phase 3 (hybride ANE+Metal) / benchmark MLX pur. Plan associe : `docs/plans/2026-04-14-ane-hybrid-pipeline.md`. README propre dans le dossier. |

## Convention

- Chaque sous-dossier a son propre `README.md` decrivant phases et fichiers.
- Les `.mlpackage` generes sont ignores par git (voir `.gitignore` racine).
- Les scripts numerotes `phase<N>_*.py` suivent l'ordre d'execution du plan.
