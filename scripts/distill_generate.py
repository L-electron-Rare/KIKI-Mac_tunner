#!/usr/bin/env python3
"""Genere des donnees de distillation avec le modele 123B fuse via MLX.

Utilise le Mistral Large entraine comme teacher pour creer des traces
de raisonnement de haute qualite pour le student Qwen3.5-35B.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def load_problems(source_paths: list[str]) -> list[str]:
    """Charge les problemes depuis un ou plusieurs fichiers JSONL."""
    problems = []
    seen = set()

    for source_path in source_paths:
        path = Path(source_path)
        files = []

        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        else:
            files = [path]

        for filepath in files:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'messages' in data:
                            for msg in data['messages']:
                                if msg.get('role') == 'user':
                                    content = msg['content']
                                    if content not in seen:
                                        problems.append(content)
                                        seen.add(content)
                                    break
                        elif 'problem' in data:
                            content = data['problem']
                            if content not in seen:
                                problems.append(content)
                                seen.add(content)
                    except json.JSONDecodeError:
                        continue

    return problems


def normalize_thinking(text: str) -> str:
    """Normalise les tags de raisonnement."""
    text = re.sub(r'<think>', '<thinking>', text)
    text = re.sub(r'</think>', '</thinking>', text)
    return text


def generate_example(
    model, tokenizer, problem: str,
    max_tokens: int = 2048,
    temp: float = 0.7,
) -> dict | None:
    """Genere un exemple de raisonnement avec le teacher."""
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampler = make_sampler(temp=temp, top_p=0.95)
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    response = normalize_thinking(response)

    # Extraire thinking + solution
    match = re.search(
        r'<thinking>(.*?)</thinking>\s*(.*)', response, re.DOTALL
    )
    if match:
        thinking = match.group(1).strip()
        solution = match.group(2).strip()
    else:
        # Pas de tags thinking — le modèle raisonne en texte libre (style Opus)
        # Wrappe toute la réponse comme thinking + solution
        response_stripped = response.strip()
        if len(response_stripped) < 100:
            return None
        # Séparer le raisonnement de la conclusion
        # Chercher un pattern de conclusion ("Answer:", "Therefore", "The answer", "So,", etc.)
        split_match = re.search(
            r'\n\n(?:(?:The |So |Therefore|Thus|Hence|In conclusion|Final)[^\n]*)',
            response_stripped
        )
        if split_match:
            thinking = response_stripped[:split_match.start()].strip()
            solution = response_stripped[split_match.start():].strip()
        else:
            # Pas de séparateur clair — prend les 80% comme thinking, 20% comme solution
            lines = response_stripped.split('\n')
            split_at = max(1, int(len(lines) * 0.8))
            thinking = '\n'.join(lines[:split_at]).strip()
            solution = '\n'.join(lines[split_at:]).strip()

    # Filtres de qualité
    if len(thinking) < 50:
        return None
    if len(solution) < 10:
        return None

    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"}
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Distillation : genere des traces de raisonnement avec le teacher 123B"
    )
    parser.add_argument("--model", required=True,
                        help="Chemin du modele teacher fuse")
    parser.add_argument("--problems", nargs="+", required=True,
                        help="Fichiers ou dossiers de problemes")
    parser.add_argument("--output", required=True,
                        help="Dossier de sortie dans data/")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-problems", type=int, default=0,
                        help="Limite le nombre de problemes (0=tous)")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre depuis les donnees existantes")
    args = parser.parse_args()

    random.seed(args.seed)

    # Charger le modele teacher
    print(f"Chargement du teacher : {args.model}")
    model, tokenizer = load(args.model)
    print("Teacher charge.")

    # Charger les problemes
    problems = load_problems(args.problems)
    if args.num_problems > 0:
        problems = problems[:args.num_problems]
    random.shuffle(problems)
    print(f"{len(problems)} problemes charges")

    # Dossier de sortie
    output_dir = Path("data") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reprise
    existing = []
    all_file = output_dir / "all.jsonl"
    if args.resume and all_file.exists():
        with open(all_file) as f:
            existing = [json.loads(line) for line in f if line.strip()]
        print(f"Reprise : {len(existing)} exemples existants")

    examples = list(existing)
    failed = 0
    start_idx = len(existing)

    for i, problem in enumerate(problems[start_idx:], start=start_idx):
        print(f"\r[{i+1}/{len(problems)}] Distillation... "
              f"({len(examples)} ok, {failed} echecs)", end="", flush=True)

        example = generate_example(
            model, tokenizer, problem,
            max_tokens=args.max_tokens,
            temp=args.temperature,
        )

        if example is None:
            failed += 1
            continue

        examples.append(example)

        # Sauvegarde incrementale
        with open(all_file, "a") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # Liberer la memoire Metal periodiquement
        if (i + 1) % 25 == 0:
            mx.metal.reset_peak_memory()

    print(f"\n\nDistillation terminee : {len(examples)} exemples "
          f"({failed} echecs sur {len(problems)})")

    # Split train/valid
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

    print(f"Sauvegarde : {len(train_examples)} train, {len(valid_examples)} valid")
    print(f"Sortie : {output_dir}")


if __name__ == "__main__":
    main()
