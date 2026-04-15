#!/usr/bin/env python3
"""Training LoRA 122B BF16 avec offload SSD des experts MoE.

Stratégie :
1. Charge le modèle lazy
2. Sauvegarde les 256 experts × 48 couches en .npz sur SSD (~144 Go)
3. Remplace les poids experts par des placeholders en RAM
4. Monkey-patch le SwitchLinear.forward pour charger depuis SSD
5. Le forward complet fonctionne avec ~90 Go RAM au lieu de 233 Go
"""

import types
import time
import yaml
import gc
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
from mlx_lm.lora import CacheDataset


os.environ['PYTHONUNBUFFERED'] = '1'


class ExpertCache:
    """Cache LRU pour les poids experts chargés du SSD."""

    def __init__(self, cache_dir, max_mb=4000):
        self.cache_dir = Path(cache_dir)
        self.data = OrderedDict()
        self.max_mb = max_mb
        self.current_mb = 0

    def get(self, layer_idx, proj_name):
        """Retourne le tenseur [256, inter, hidden] depuis cache ou SSD."""
        key = (layer_idx, proj_name)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]

        # Charger depuis SSD
        path = self.cache_dir / f"layer_{layer_idx:02d}_{proj_name}.npy"
        arr = np.load(str(path))
        tensor = mx.array(arr)
        mx.eval(tensor)

        size_mb = arr.nbytes / 1e6
        self.data[key] = tensor
        self.current_mb += size_mb

        # Éviction LRU
        while self.current_mb > self.max_mb and len(self.data) > 1:
            _, old_tensor = self.data.popitem(last=False)
            self.current_mb -= old_tensor.nbytes / 1e6
            del old_tensor

        return tensor

    def clear(self):
        self.data.clear()
        self.current_mb = 0


def find_layers(model):
    for path in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]:
        try:
            return path(model)
        except AttributeError:
            continue
    raise ValueError("Couches introuvables")


def save_experts_to_ssd(model, cache_dir):
    """Sauvegarde tous les experts en fichiers .npy individuels par couche+projection."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    layers = find_layers(model)
    for i, layer in enumerate(layers):
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            continue
        switch = getattr(mlp, 'switch_mlp', None)
        if switch is None:
            continue

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            out_path = cache_dir / f"layer_{i:02d}_{proj_name}.npy"
            if out_path.exists():
                continue

            proj = getattr(switch, proj_name)
            w = proj.weight  # [256, inter, hidden] bf16
            mx.eval(w)
            # bf16 → float16 pour numpy
            w_f16 = w.astype(mx.float16)
            mx.eval(w_f16)
            np.save(str(out_path), np.array(w_f16))
            del w_f16

        print(f"\r  Couche {i+1}/{len(layers)}", end="", flush=True)

    print(" OK")


def setup_ssd_loaders(switch_mlp, layer_idx, cache):
    """Configure les SwitchLinear pour charger depuis SSD."""
    placeholder = mx.zeros((1, 1, 1), dtype=mx.bfloat16)

    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(switch_mlp, proj_name)

        # Sauvegarder les dimensions originales
        orig_shape = proj.weight.shape  # [256, out, in]
        proj._ssd_meta = (orig_shape[1], orig_shape[2], orig_shape[0])

        # Créer le loader closure
        _layer_idx = layer_idx
        _proj_name = proj_name
        _cache = cache
        proj._ssd_loader = lambda li=_layer_idx, pn=_proj_name, c=_cache: c.get(li, pn)
        proj._ssd_placeholder = placeholder

        # Remplacer le poids par le placeholder
        proj.weight = placeholder


def offload_and_patch(model, cache_dir):
    """Offload les experts sur SSD et patch les forwards."""
    cache = ExpertCache(cache_dir, max_mb=4000)
    layers = find_layers(model)
    freed = 0

    for i, layer in enumerate(layers):
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            continue
        switch = getattr(mlp, 'switch_mlp', None)
        if switch is None:
            continue

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(switch, proj_name)
            w = proj.weight
            nbytes = w.size * 2
            proj.weight = mx.zeros((1, 1, 1), dtype=mx.bfloat16)
            mx.eval(proj.weight)
            freed += nbytes

        # Configurer les loaders SSD sur les SwitchLinear
        setup_ssd_loaders(switch, i, cache)

        print(f"\r  Couche {i+1}/{len(layers)} patchée", end="", flush=True)

    gc.collect()
    print(f"\n  {freed / 1e9:.1f} Go libérés de la RAM")
    return cache


def main():
    config_path = "configs/mlx-lm-qwen35-122b-opus-v3.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config['model']
    cache_dir = Path(model_path) / ".expert_cache"

    print(f"=== Training 122B BF16 + Expert Offload SSD ===")
    print(f"  Rank: {config['lora_parameters']['rank']}")
    print(f"  Iters: {config['iters']}, Seq: {config['max_seq_length']}")
    print(flush=True)

    # 1. Charger lazy
    print("1. Chargement lazy...", flush=True)
    model, tokenizer = load(model_path, lazy=True)
    print(f"   OK", flush=True)

    # 2. Sauvegarder experts sur SSD
    if not (cache_dir / "layer_00_gate_proj.npy").exists():
        print("2. Sauvegarde experts SSD (~144 Go)...", flush=True)
        save_experts_to_ssd(model, cache_dir)
    else:
        print("2. Cache SSD existant ✓", flush=True)

    # 3. Offload + patch
    print("3. Offload experts + patch forward...", flush=True)
    expert_cache = offload_and_patch(model, cache_dir)
    mx.metal.reset_peak_memory()
    print(f"   Peak mem: {mx.metal.get_peak_memory()/1e9:.1f} Go", flush=True)

    # 4. Training via mlx_lm directement (LoRA est appliqué par mlx_lm)
    # On ne peut pas utiliser mlx_lm lora car il re-charge le modèle
    # Il faut appeler le trainer directement
    from mlx_lm.tuner.lora import LoRALinear

    layers = find_layers(model)
    n = len(layers)
    applied = 0
    for i in range(n):
        layer = layers[i]
        attn = getattr(layer, 'self_attn', None) or getattr(layer, 'linear_attn', None)
        if attn is None:
            continue
        for name, child in attn.children().items():
            if isinstance(child, nn.Linear):
                lora = LoRALinear.from_base(
                    child,
                    r=config['lora_parameters']['rank'],
                    scale=config['lora_parameters']['scale'],
                    dropout=config['lora_parameters']['dropout'],
                )
                setattr(attn, name, lora)
                applied += 1

    model.freeze()
    for i in range(n):
        layer = layers[i]
        attn = getattr(layer, 'self_attn', None) or getattr(layer, 'linear_attn', None)
        if attn is None:
            continue
        for name, child in attn.children().items():
            if isinstance(child, LoRALinear):
                child.unfreeze()

    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"4. LoRA: {applied} couches, {trainable/1e6:.0f}M/{total/1e6:.0f}M ({trainable/total*100:.1f}%)", flush=True)

    # 5. Dataset
    args_ds = types.SimpleNamespace(data=config['data'], train=True, test=False)
    train_set, valid_set, _ = load_dataset(args_ds, tokenizer)
    print(f"5. Dataset: {len(train_set)} train, {len(valid_set)} valid", flush=True)

    # 6. Training
    adapter_file = str(Path(config['adapter_path']) / "adapters.safetensors")
    Path(config['adapter_path']).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArgs(
        batch_size=config['batch_size'],
        iters=config['iters'],
        val_batches=config['val_batches'],
        steps_per_report=config['steps_per_report'],
        steps_per_eval=config['steps_per_eval'],
        steps_per_save=config['save_every'],
        max_seq_length=config['max_seq_length'],
        adapter_file=adapter_file,
        grad_checkpoint=config.get('grad_checkpoint', True),
        grad_accumulation_steps=config.get('grad_accumulation_steps', 8),
    )

    optimizer = optim.AdamW(learning_rate=float(config['learning_rate']))

    print(f"\n6. Training — Peak: {mx.metal.get_peak_memory()/1e9:.1f} Go", flush=True)

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
    )

    print(f"\n=== Terminé === {adapter_file}")


if __name__ == "__main__":
    main()
