#!/usr/bin/env python3
"""TUI de monitoring du training en temps réel.

Affiche : progression, loss, vitesse, mémoire, ETA.
Lit le log du training et met à jour en continu.

Usage : python scripts/training_tui.py [logfile]
"""

import sys
import re
import time
import os
from pathlib import Path
from datetime import datetime, timedelta


def parse_iter_line(line):
    """Parse une ligne Iter du log mlx_lm."""
    result = {}

    # Iter N: Val loss X, Val took Xs
    m = re.match(r'Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s', line)
    if m:
        result['iter'] = int(m.group(1))
        result['val_loss'] = float(m.group(2))
        result['val_time'] = float(m.group(3))
        result['type'] = 'val'
        return result

    # Iter N: Train loss X, Learning Rate X, It/sec X, Tokens/sec X, Trained Tokens X, Peak mem X GB
    m = re.match(r'Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB', line)
    if m:
        result['iter'] = int(m.group(1))
        result['train_loss'] = float(m.group(2))
        result['lr'] = float(m.group(3))
        result['it_sec'] = float(m.group(4))
        result['tok_sec'] = float(m.group(5))
        result['tokens'] = int(m.group(6))
        result['peak_mem'] = float(m.group(7))
        result['type'] = 'train'
        return result

    # Iter N: Saved adapter weights
    m = re.match(r'Iter (\d+): Saved', line)
    if m:
        result['iter'] = int(m.group(1))
        result['type'] = 'save'
        return result

    return None


def parse_config_line(line):
    """Parse les lignes de config du training."""
    m = re.match(r'Starting training\.\.\., iters: (\d+)', line)
    if m:
        return {'total_iters': int(m.group(1))}

    m = re.match(r'Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)', line)
    if m:
        return {
            'trainable_pct': float(m.group(1)),
            'trainable_m': float(m.group(2)),
            'total_m': float(m.group(3)),
        }
    return None


def render_bar(progress, width=40, filled='█', empty='░'):
    """Barre de progression."""
    n = int(progress * width)
    return filled * n + empty * (width - n)


def render_tui(state):
    """Affiche le TUI."""
    os.system('clear')

    now = datetime.now().strftime('%H:%M:%S')

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  🔥 KIKI Training Monitor                        {now}  ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")

    # Modèle
    model = state.get('model', '?')
    print(f"║  Modèle : {model:<50}║")

    # Params
    if 'trainable_m' in state:
        print(f"║  Params : {state['trainable_m']:.0f}M trainable / {state['total_m']:.0f}M ({state['trainable_pct']:.1f}%)       ║")

    print(f"╠══════════════════════════════════════════════════════════════╣")

    # Progression
    current = state.get('current_iter', 0)
    total = state.get('total_iters', 3000)
    progress = current / total if total > 0 else 0
    bar = render_bar(progress)
    print(f"║  Progression : {bar} {progress*100:5.1f}%  ║")
    print(f"║  Iter : {current:>6} / {total:<6}                                  ║")

    # ETA
    if state.get('it_sec', 0) > 0:
        remaining = (total - current) / state['it_sec']
        eta = datetime.now() + timedelta(seconds=remaining)
        hours = int(remaining // 3600)
        mins = int((remaining % 3600) // 60)
        print(f"║  ETA  : {hours}h{mins:02d}m (fin ~{eta.strftime('%H:%M')})                            ║")

    print(f"╠══════════════════════════════════════════════════════════════╣")

    # Loss
    train_loss = state.get('train_loss', '—')
    val_loss = state.get('val_loss', '—')
    best_val = state.get('best_val_loss', '—')

    if isinstance(train_loss, float):
        train_str = f"{train_loss:.4f}"
    else:
        train_str = str(train_loss)

    if isinstance(val_loss, float):
        val_str = f"{val_loss:.4f}"
    else:
        val_str = str(val_loss)

    if isinstance(best_val, float):
        best_str = f"{best_val:.4f}"
    else:
        best_str = str(best_val)

    print(f"║  Train Loss : {train_str:<10}                                  ║")
    print(f"║  Val Loss   : {val_str:<10}  (best: {best_str})                  ║")

    # Historique val loss
    val_history = state.get('val_history', [])
    if val_history:
        hist = ' → '.join(f"{v:.3f}" for v in val_history[-6:])
        print(f"║  Val hist   : {hist:<46}║")

    print(f"╠══════════════════════════════════════════════════════════════╣")

    # Performance
    tok_sec = state.get('tok_sec', 0)
    it_sec = state.get('it_sec', 0)
    peak_mem = state.get('peak_mem', 0)

    print(f"║  Vitesse  : {tok_sec:>6.1f} tok/s | {it_sec:.3f} it/s                     ║")
    print(f"║  Mémoire  : {peak_mem:>6.1f} Go / 512 Go                           ║")

    mem_bar = render_bar(peak_mem / 512, width=30, filled='▓', empty='░')
    print(f"║  Metal    : {mem_bar} {peak_mem/512*100:.0f}%         ║")

    # Tokens
    tokens = state.get('tokens', 0)
    print(f"║  Tokens   : {tokens:>10,}                                    ║")

    print(f"╠══════════════════════════════════════════════════════════════╣")

    # Derniers checkpoints
    saves = state.get('saves', [])
    if saves:
        print(f"║  Checkpoints : {', '.join(str(s) for s in saves[-5:]): <44}║")
    else:
        print(f"║  Checkpoints : aucun                                        ║")

    # Status
    status = state.get('status', 'En cours')
    print(f"║  Status     : {status:<46}║")

    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Ctrl+C pour quitter | Log : {state.get('logfile', '?')}")


def main():
    logfile = sys.argv[1] if len(sys.argv) > 1 else "logs/pipeline-122b.log"

    # Aussi chercher dans les task outputs
    if not Path(logfile).exists():
        # Chercher le log le plus récent
        for candidate in [
            "logs/train-122b-offload.log",
            "logs/pipeline-122b.log",
            "logs/pipeline-35b.log",
        ]:
            if Path(candidate).exists():
                logfile = candidate
                break

    state = {
        'logfile': logfile,
        'model': 'Qwen3.5-122B-A10B-BF16',
        'total_iters': 3000,
        'current_iter': 0,
        'val_history': [],
        'best_val_loss': float('inf'),
        'saves': [],
        'status': 'Démarrage...',
    }

    last_size = 0

    try:
        while True:
            # Lire les nouvelles lignes du log
            if Path(logfile).exists():
                with open(logfile) as f:
                    content = f.read()

                if len(content) > last_size:
                    new_content = content[last_size:]
                    last_size = len(content)

                    for line in new_content.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue

                        # Config
                        cfg = parse_config_line(line)
                        if cfg:
                            state.update(cfg)

                        # Iter
                        parsed = parse_iter_line(line)
                        if parsed:
                            state['current_iter'] = parsed['iter']
                            state['status'] = 'En cours'

                            if parsed['type'] == 'train':
                                state['train_loss'] = parsed['train_loss']
                                state['lr'] = parsed['lr']
                                state['it_sec'] = parsed['it_sec']
                                state['tok_sec'] = parsed['tok_sec']
                                state['tokens'] = parsed['tokens']
                                state['peak_mem'] = parsed['peak_mem']

                            elif parsed['type'] == 'val':
                                state['val_loss'] = parsed['val_loss']
                                state['val_history'].append(parsed['val_loss'])
                                if parsed['val_loss'] < state['best_val_loss']:
                                    state['best_val_loss'] = parsed['val_loss']

                            elif parsed['type'] == 'save':
                                state['saves'].append(parsed['iter'])

                        # Erreurs
                        if 'Error' in line or 'error' in line:
                            state['status'] = f"ERREUR: {line[:40]}..."

                        if 'Training terminé' in line or 'Saved final' in line:
                            state['status'] = 'Terminé ✓'

            render_tui(state)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nMonitor arrêté.")


if __name__ == "__main__":
    main()
