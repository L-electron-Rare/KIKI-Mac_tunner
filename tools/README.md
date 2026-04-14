# tools/

Outils autonomes (monitoring, TUI, utilitaires) utilisables hors pipeline.

## Fichiers

| Fichier | Role |
|---|---|
| `train_monitor_tui.py` | TUI live (rich) pour le training Qwen3.5-122B-A10B-BF16 mac-port. Tail le log `mlx_lm.lora` + `memcsv` watchdog, affiche progress bar, iter/ETA, sparkline loss, trajectoire val, memoire (RSS/swap/peak Metal), rate (it/s, tok/s, LR), health checks et log recent. |

## Usage

Sur Studio :
```bash
.venv/bin/python tools/train_monitor_tui.py
```

Depuis GrosMac (TTY requis) :
```bash
ssh -t studio "cd KIKI-Mac_tunner && .venv/bin/python tools/train_monitor_tui.py"
```

Options principales :
- `--log PATH` : log explicite (sinon dernier `train-*.log` sous `--root`).
- `--memcsv PATH` : memcsv explicite (sinon dernier `memcsv-*.csv`).
- `--root DIR` : base logs (defaut `logs/122b-macport`).
- `--total-iters N` : total iters pour la progress bar (defaut 3000).
- `--refresh SEC` : intervalle rafraichissement UI (defaut 2.0s).
