#!/usr/bin/env python3
"""Fusionne plusieurs datasets JSONL en un jeu de données d'entraînement combiné."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fusionner plusieurs datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Répertoires sources dans data/",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Nom du répertoire de sortie dans data/",
    )
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Supprimer les problèmes en double",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    examples: list[dict] = []
    seen_problems: set[str] = set()

    for source in args.sources:
        source_dir = Path("data") / source
        count = 0
        for jsonl_file in sorted(source_dir.glob("*.jsonl")):
            if jsonl_file.name in ("valid.jsonl", "test.jsonl"):
                continue  # Ne fusionner que les données train, reconstruire le split val
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line)
                    if args.deduplicate:
                        problem = data["messages"][0]["content"]
                        if problem in seen_problems:
                            continue
                        seen_problems.add(problem)
                    examples.append(data)
                    count += 1
        print(f"  {source} : {count} exemples")

    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_split))
    train = examples[:split_idx]
    valid = examples[split_idx:]

    output_dir = Path("data") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{len(examples)} exemples fusionnés depuis {len(args.sources)} sources")
    print(f"Train : {len(train)}, Valid : {len(valid)}")
    print(f"Sortie : {output_dir}")


if __name__ == "__main__":
    main()
