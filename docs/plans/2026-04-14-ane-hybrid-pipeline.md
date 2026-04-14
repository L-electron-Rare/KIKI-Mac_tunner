# Pipeline Hybride ANE+Metal pour Qwen3.5-35B-A3B

**Date** : 2026-04-14
**Objectif** : Faire tourner Qwen3.5-35B-A3B-Opus sur Apple Neural Engine
**Performance cible** : ~28 tok/s sur M3 Ultra
**Memoire active** : ~4 Go

## Contexte

Qwen3.5-35B-A3B est un MoE hybride :
- 40 couches au total (10 blocs de 4)
- 30 couches DeltaNet (attention lineaire recurrente)
- 10 couches Full Attention (GQA, 16Q/2KV heads)
- 256 experts par couche, 8 actifs + 1 shared par token
- 3B parametres actifs sur 35B totaux

Deux blocages identifies :
1. DeltaNet non supporte par ANEMLL/CoreML
2. Routage MoE dynamique incompatible avec le graphe statique CoreML

## Architecture hybride

```
Embedding (ANE CoreML)
  |
  v
Pour chaque couche (x40) :
  |
  +-- RMSNorm → ANE
  |
  +-- Attention :
  |     DeltaNet (x30) → ANE (forme chunkwise parallele)
  |     Full Attn (x10) → ANE (ANEMLL standard)
  |
  +-- MoE FFN :
  |     Router → CPU (softmax + top-8)
  |     Shared expert → ANE (CoreML)
  |     8 experts routes → GPU (Metal, Flash-MoE)
  |     Combine → GPU
  |
  v
LM Head (ANE CoreML)
```

Pipeline parallele : ANE traite couche N+1 pendant que GPU fait le MoE de couche N.

## Phase 1 : Prototype DeltaNet CoreML (Semaines 1-3)

### Objectif
Convertir UNE couche DeltaNet en CoreML et verifier qu'elle tourne sur ANE.

### Sous-taches

1.1 **Reference PyTorch** (3 jours)
   - Extraire une couche DeltaNet de Qwen3.5 en PyTorch pur
   - Implementer la forme chunkwise parallele (C=64)
   - Verifier l'equivalence numerique avec le modele HF
   - Fichier : `research/ane-hybrid/deltanet_reference.py`

1.2 **Conversion Conv2d** (2 jours)
   - Reecrire toutes les projections lineaires en Conv2d(kernel_size=1)
   - Pattern ANEMLL : [B, C, 1, T] au lieu de [B, T, C]
   - Fichier : `research/ane-hybrid/deltanet_conv2d.py`

1.3 **CoreML conversion** (3 jours)
   - Tracer avec coremltools ct.convert()
   - Ajouter ct.StateType pour l'etat recurrent S [B, H, K, V]
   - Separer prefill (chunk=64) et decode (single token)
   - Fichier : `research/ane-hybrid/convert_deltanet.py`

1.4 **Test ANE** (2 jours)
   - Charger le .mlpackage, verifier execution sur ANE
   - Profiler avec ANEMLL ANE Profiler
   - Comparer precision FP16 vs reference FP32
   - Fichier : `research/ane-hybrid/test_deltanet_ane.py`

### Ops CoreML MIL necessaires

| Op DeltaNet | Op CoreML | Note |
|-------------|-----------|------|
| Q @ K^T | matmul | Standard |
| State update | matmul + ct.StateType | iOS 18+ |
| Gating exp(g) | exp | Standard |
| Cum. decay | log + cumsum + exp | Standard |
| Conv1D short | Conv1d ou unrolled | Kernel=4 |
| WY transform | matmul triangulaire (unroll C=64) | Statique |
| Beta gate | sigmoid | Standard |

### Etat recurrent (ct.StateType)

```python
# Chaque couche DeltaNet maintient un etat S
# Shape: [batch, num_key_heads, key_head_dim, value_head_dim] = [1, 16, 128, 128]
# Taille: 16 * 128 * 128 * 2 bytes = 512 Ko par couche
# Total 30 couches: ~15 Mo
state_spec = ct.StateType(
    wrapped_type=ct.TensorType(shape=(1, 16, 128, 128)),
    default_value=np.zeros((1, 16, 128, 128), dtype=np.float16)
)
```

## Phase 2 : Stack complet DeltaNet + Attention (Semaines 3-5)

2.1 Convertir les 30 couches DeltaNet en chunks CoreML
2.2 Convertir les 10 couches Full Attention via ANEMLL existant
2.3 Integrer le KV-cache pour les couches Full Attention
2.4 Pipeline de prefill complet (embed → 40 couches → lm_head)
2.5 Pipeline de decode complet

## Phase 3 : Integration MoE hybride ANE+Metal (Semaines 5-8)

3.1 Adapter Flash-MoE Metal shaders pour les dimensions 35B
3.2 Compiler le shared expert comme CoreML → ANE
3.3 Router sur CPU avec prediction co-activation
3.4 Pipeline hybride : ANE fait attention, GPU fait experts
3.5 Implementer le pipeline parallele (ANE couche N+1 // GPU MoE couche N)

## Phase 4 : Optimisation (Semaines 8-10)

4.1 Quantification LUT 4-bit des poids DeltaNet
4.2 Profiling ANE/GPU overlap
4.3 Tuning chunk size et batch size
4.4 Integration ANEMLL Swift CLI
4.5 Benchmarks et comparaison avec MLX pur

## Estimation performance

| Composant | Temps/token (decode) |
|-----------|---------------------|
| DeltaNet 30 couches (ANE) | ~9 ms |
| Full Attention 10 couches (ANE) | ~5 ms |
| MoE routing + experts (CPU+GPU) | ~40 ms |
| **Total sequentiel** | **~54 ms (~18 tok/s)** |
| **Avec pipeline parallele** | **~35 ms (~28 tok/s)** |

## Dependances

- ANEMLL 0.3.5+ (`/tmp/anemll`)
- coremltools 9.0+ (installe)
- Flash-MoE Metal shaders (`/tmp/mac-code/research/expert-sniper/`)
- Qwen3.5-35B-A3B-Opus (HF weights deja en cache)
- FLA (flash-linear-attention) pour reference DeltaNet

## References

- [DeltaNet chunkwise parallel form](https://sustcsonglin.github.io/blog/2024/deltanet-2/)
- [Gated DeltaNet paper (ICLR 2025)](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)
- [FLA implementation](https://github.com/fla-org/flash-linear-attention)
- [CoreML Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [CoreML MIL ops](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html)
- [Flash-MoE (Anemll)](https://github.com/Anemll/flash-moe)
- [Apple Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
