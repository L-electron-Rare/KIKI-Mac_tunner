# Lib

## mlx_lm_fork

Fork de mlx_lm avec support SSD offload pour les experts MoE.

Modification dans `models/switch_layers.py` :
- `SwitchLinear` : ajout `_ssd_loader`, `_ssd_placeholder`, `_ssd_dims`
- Charge les poids experts depuis SSD a chaque forward
- Libere apres usage → RAM ~90 Go au lieu de 233 Go pour le 122B BF16

### Usage

```python
# Dans train_offload.py, le fork est injecte via monkey-patch :
import mlx_lm_fork.models.switch_layers as fork
import mlx_lm.models.switch_layers as orig
orig.SwitchLinear = fork.SwitchLinear
```

## MLX 3x Metal Buffer Limit

MLX stock limite les Metal buffers a 499K, insuffisant pour le 122B BF16.
Fork recompile dans `/tmp/mlx-fork` avec limite 3x (1.5M buffers), installe dans le venv.

Sans ce fork, le training 122B echoue avec des erreurs Metal buffer allocation.

### Anti-Patterns

- Ne pas faire `pip install mlx-lm` sans re-verifier le fork
- Le fork mlx_lm ne modifie QUE switch_layers.py, tout le reste est identique
- Ne pas reinstaller mlx via pip — le fork `/tmp/mlx-fork` doit rester dans le venv
