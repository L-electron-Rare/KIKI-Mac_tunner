#!/usr/bin/env python3
"""Génère des données de raisonnement synthétiques avec des modèles teachers via MLX."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MIN_THINKING_LENGTH = 100
MAX_THINKING_LENGTH = 16000
MIN_SOLUTION_LENGTH = 20


def load_problems(source_path: str) -> list[str]:
    """Charge les problèmes depuis un dataset existant ou un fichier de prompts."""
    problems: list[str] = []
    path = Path(source_path)

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data:
                    for msg in data["messages"]:
                        if msg["role"] == "user":
                            problems.append(msg["content"])
                elif "problem" in data:
                    problems.append(data["problem"])
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        problems.append(item)
                    elif isinstance(item, dict) and "problem" in item:
                        problems.append(item["problem"])

    return problems


def normalize_thinking_tags(text: str) -> str:
    """Convertit les différents formats de tags thinking en <thinking>...</thinking>."""
    text = re.sub(r"<think>", "<thinking>", text)
    text = re.sub(r"</think>", "</thinking>", text)
    return text


def extract_thinking_and_solution(response: str) -> tuple[str, str] | None:
    """Extrait le raisonnement et la solution de la réponse du modèle."""
    response = normalize_thinking_tags(response)

    match = re.search(
        r"<thinking>(.*?)</thinking>\s*(.*)", response, re.DOTALL
    )
    if match:
        thinking = match.group(1).strip()
        solution = match.group(2).strip()
        return thinking, solution

    return None


def generate_example(
    model,
    tokenizer,
    problem: str,
    max_tokens: int = 8192,
    temp: float = 0.7,
) -> dict | None:
    """Génère un seul exemple de raisonnement."""
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp, top_p=0.95),
    )

    result = extract_thinking_and_solution(response)
    if result is None:
        return None

    thinking, solution = result

    if len(thinking) < MIN_THINKING_LENGTH:
        return None
    if len(thinking) > MAX_THINKING_LENGTH:
        return None
    if len(solution) < MIN_SOLUTION_LENGTH:
        return None

    return {
        "messages": [
            {"role": "user", "content": problem},
            {
                "role": "assistant",
                "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}",
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Générer des données de raisonnement synthétiques"
    )
    parser.add_argument(
        "--model", required=True, help="Chemin vers le modèle teacher"
    )
    parser.add_argument(
        "--problems",
        required=True,
        help="Chemin vers le fichier de problèmes (jsonl/json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Nom du répertoire de sortie (sera dans data/)",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--num-problems",
        type=int,
        default=0,
        help="Limiter le nombre de problèmes (0=tous)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Ratio du split de validation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true", help="Reprendre depuis une sortie existante"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Chargement du modèle
    print(f"Chargement du modèle : {args.model}")
    model, tokenizer = load(args.model)

    # Chargement des problèmes
    problems = load_problems(args.problems)
    if args.num_problems > 0:
        problems = problems[: args.num_problems]
    print(f"{len(problems)} problèmes chargés")

    # Répertoire de sortie
    output_dir = Path("data") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support de la reprise
    existing: list[dict] = []
    if args.resume and (output_dir / "all.jsonl").exists():
        with open(output_dir / "all.jsonl") as f:
            existing = [json.loads(line) for line in f]
        print(f"Reprise : {len(existing)} exemples existants")

    # Génération
    examples = list(existing)
    failed = 0

    for i, problem in enumerate(
        problems[len(existing) :], start=len(existing)
    ):
        print(
            f"\r[{i + 1}/{len(problems)}] Génération... ({failed} échoués)",
            end="",
            flush=True,
        )

        example = generate_example(
            model,
            tokenizer,
            problem,
            max_tokens=args.max_tokens,
            temp=args.temperature,
        )

        if example is None:
            failed += 1
            continue

        examples.append(example)

        # Sauvegarde incrémentale
        with open(output_dir / "all.jsonl", "a") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # Libération périodique de la mémoire
        if (i + 1) % 50 == 0:
            mx.metal.reset_peak_memory()

    print(f"\n{len(examples)} exemples générés ({failed} échoués)")

    # Séparation train/valid
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_split))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Sauvegardé : {len(train_examples)} train, {len(valid_examples)} valid")
    print(f"Sortie : {output_dir}")


if __name__ == "__main__":
    main()
