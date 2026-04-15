#!/usr/bin/env python3
"""Prépare le dataset en 3 phases curriculum (court → moyen → long).

Phase 1: < 512 tokens  — apprentissage rapide des patterns
Phase 2: 512-1024      — raisonnement moyen
Phase 3: 1024+         — raisonnement complexe, chaînes longues

Chaque phase a son propre train.jsonl + valid.jsonl.
Le training lance les 3 phases séquentiellement avec des LR décroissants.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/final-opus-v3-1")
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/Qwen3.5-122B-A10B-BF16"
    out_base = Path("data/curriculum")

    print(f"Dataset: {data_dir}")
    print(f"Tokenizer: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Limites des phases
    phases = [
        {"name": "phase1-short", "max": 512,  "min": 0},
        {"name": "phase2-medium", "max": 1024, "min": 512},
        {"name": "phase3-long", "max": 99999, "min": 1024},
    ]

    for split in ["train", "valid"]:
        src = data_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  {src} introuvable, skip")
            continue

        # Tokeniser et trier
        examples = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get("messages", [])
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
                tokens = len(tokenizer.encode(text))
                examples.append((tokens, line))

        print(f"\n{split}: {len(examples)} exemples")

        # Répartir par phase
        for phase in phases:
            phase_dir = out_base / phase["name"]
            phase_dir.mkdir(parents=True, exist_ok=True)

            filtered = [(t, l) for t, l in examples if phase["min"] <= t < phase["max"]]
            # Trier par longueur croissante dans chaque phase
            filtered.sort(key=lambda x: x[0])

            with open(phase_dir / f"{split}.jsonl", "w") as f:
                for _, line in filtered:
                    f.write(line + "\n")

            avg_tok = sum(t for t, _ in filtered) / len(filtered) if filtered else 0
            print(f"  {phase['name']}: {len(filtered)} exemples (avg {avg_tok:.0f} tok)")

    # Résumé
    print("\n=== Curriculum prêt ===")
    for phase in phases:
        phase_dir = out_base / phase["name"]
        train_count = sum(1 for _ in open(phase_dir / "train.jsonl")) if (phase_dir / "train.jsonl").exists() else 0
        print(f"  {phase['name']}: {train_count} train")

    print(f"\nLancer le training curriculum :")
    print(f"  ./scripts/train_curriculum.sh")


if __name__ == "__main__":
    main()
