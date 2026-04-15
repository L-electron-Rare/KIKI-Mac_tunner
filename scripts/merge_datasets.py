#!/usr/bin/env python3
"""Fusionne plusieurs datasets JSONL en un jeu de données d'entraînement combiné.

Dedup key = (user_content, assistant_content) pour préserver la diversité
de reasoning (même problème + réponses différentes = exemples distincts).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fusionner plusieurs datasets")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    examples: list[dict] = []
    seen_keys: set[str] = set()

    for source in args.sources:
        source_dir = Path("data") / source
        count = 0
        for jsonl_file in sorted(source_dir.glob("*.jsonl")):
            if jsonl_file.name in ("valid.jsonl", "test.jsonl"):
                continue
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line)
                    if args.deduplicate:
                        msgs = data.get("messages", [])
                        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                        assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                        key = hashlib.sha256((user + "\n###\n" + assistant).encode("utf-8")).hexdigest()
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                    examples.append(data)
                    count += 1
        print(f"  {source} : {count} exemples uniques")

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
