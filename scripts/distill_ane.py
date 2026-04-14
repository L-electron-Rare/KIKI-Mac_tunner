#!/usr/bin/env python3
"""Distillation via ANE — generation rapide sur Apple Neural Engine.

Utilise l'import direct d'ANEMLL (pas subprocess) pour 8x de gain.
Supporte le scoring parallele pour 3x de plus.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from ane_inference import ANEModel, ParallelANEScorer


def normalize_thinking(text: str) -> str:
    """Normalise les tags de raisonnement (<think> -> <thinking>)."""
    text = re.sub(r"<think>", "<thinking>", text)
    text = re.sub(r"</think>", "</thinking>", text)
    return text


def extract_or_wrap_thinking(response: str) -> tuple[str, str] | None:
    """Extrait thinking/solution ou wrappe la reponse.

    Returns:
        Tuple (thinking, solution) ou None si la reponse est trop courte.
    """
    response = normalize_thinking(response)

    # Cas 1: Tags <thinking> explicites
    match = re.search(
        r"<thinking>(.*?)</thinking>\s*(.*)", response, re.DOTALL
    )
    if match:
        thinking = match.group(1).strip()
        solution = match.group(2).strip()
        if len(thinking) < 50 or len(solution) < 10:
            return None
        return thinking, solution

    # Cas 2: Pas de tags — wrap le raisonnement
    response = response.strip()
    if len(response) < 100:
        return None

    # Chercher un pattern de conclusion
    split_match = re.search(
        r"\n\n(?:(?:The |So |Therefore|Thus|Hence|In conclusion|Final)[^\n]*)",
        response,
    )
    if split_match:
        thinking = response[: split_match.start()].strip()
        solution = response[split_match.start() :].strip()
    else:
        # Pas de separateur clair — 80% thinking, 20% solution
        lines = response.split("\n")
        split_at = max(1, int(len(lines) * 0.8))
        thinking = "\n".join(lines[:split_at]).strip()
        solution = "\n".join(lines[split_at:]).strip()

    if len(thinking) < 50 or len(solution) < 10:
        return None
    return thinking, solution


def load_problems(paths: list[str]) -> list[str]:
    """Charge les problemes depuis des fichiers ou dossiers JSONL."""
    problems = []
    seen: set[str] = set()

    for p in paths:
        path = Path(p)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]

        for f in files:
            if not f.exists():
                print(f"Attention: fichier introuvable {f}")
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "messages" in data:
                        for msg in data["messages"]:
                            if msg.get("role") == "user":
                                content = msg["content"]
                                if content not in seen:
                                    problems.append(content)
                                    seen.add(content)
                                break
                    elif "problem" in data:
                        content = data["problem"]
                        if content not in seen:
                            problems.append(content)
                            seen.add(content)

    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Distillation ANE rapide — generation via Apple Neural Engine"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Dossier du modele ANE converti (contient meta.yaml)",
    )
    parser.add_argument(
        "--problems", nargs="+", required=True,
        help="Fichiers ou dossiers JSONL de problemes",
    )
    parser.add_argument(
        "--output", required=True,
        help="Nom du dossier sortie dans data/",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--num-problems", type=int, default=0,
        help="Limite le nombre de problemes (0=tous)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Nombre d'instances ANE paralleles (1-3)",
    )
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reprendre depuis les donnees existantes",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Charger les problemes
    problems = load_problems(args.problems)
    if args.num_problems > 0:
        problems = problems[: args.num_problems]
    random.shuffle(problems)
    print(f"Charge {len(problems)} problemes")

    # Charger le modele ANE
    t0 = time.time()
    if args.parallel > 1:
        print(f"Mode parallele : {args.parallel} instances ANE")
        scorer = ParallelANEScorer(
            args.model_dir, num_workers=args.parallel
        )

        def generate_fn(prompt):
            return scorer.generate_parallel(
                [prompt], args.max_tokens, args.temperature
            )[0]

    else:
        print("Mode sequentiel : 1 instance ANE")
        model = ANEModel(args.model_dir)

        def generate_fn(prompt):
            return model.generate(
                prompt, args.max_tokens, args.temperature
            )

    load_time = time.time() - t0
    print(f"Modele charge en {load_time:.1f}s")

    # Warmup (2 appels courts pour stabiliser CoreML)
    print("Warmup ANE...", end=" ", flush=True)
    for _ in range(2):
        if args.parallel > 1:
            scorer.generate_parallel(["Hello"], max_tokens=10, temperature=0.0)
        else:
            model.generate("Hello", max_tokens=10, temperature=0.0)
    print("OK")

    # Dossier de sortie
    output_dir = Path("data") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reprise eventuelle
    existing: list[dict] = []
    all_file = output_dir / "all.jsonl"
    if args.resume and all_file.exists():
        with open(all_file) as f:
            existing = [json.loads(line) for line in f if line.strip()]
        print(f"Reprise : {len(existing)} exemples existants")

    examples = list(existing)
    failed = 0
    start_idx = len(existing)
    gen_start = time.time()

    for i, problem in enumerate(problems[start_idx:], start=start_idx):
        elapsed = time.time() - gen_start
        rate = (i - start_idx + 1) / elapsed if elapsed > 0 else 0
        print(
            f"\r[{i + 1}/{len(problems)}] ANE... "
            f"({len(examples)} ok, {failed} echecs, {rate:.1f} ex/min)",
            end="",
            flush=True,
        )

        response = generate_fn(problem)
        if response is None:
            failed += 1
            continue

        result = extract_or_wrap_thinking(response)
        if result is None:
            failed += 1
            continue

        thinking, solution = result
        example = {
            "messages": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}",
                },
            ]
        }
        examples.append(example)

        # Sauvegarde incrementale
        with open(all_file, "a") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    total_time = time.time() - gen_start
    print(
        f"\n\nDistillation terminee : {len(examples)} exemples "
        f"({failed} echecs) en {total_time:.0f}s"
    )

    # Split train/valid
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_split))

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in examples[:split_idx]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in examples[split_idx:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"Sauvegarde : {split_idx} train, {len(examples) - split_idx} valid"
    )
    print(f"Sortie : {output_dir}")

    # Nettoyage
    if args.parallel > 1:
        scorer.shutdown()


if __name__ == "__main__":
    main()
