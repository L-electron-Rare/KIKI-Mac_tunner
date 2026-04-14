#!/usr/bin/env python3
"""Genere des donnees de raisonnement via llama.cpp en mode CPU.

Permet de tourner en parallele avec le training MLX sur GPU Metal.
Utilise llama-cli avec --threads pour exploiter tous les CPU cores.
"""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

TIMEOUT_SECONDS = 600
MIN_THINKING_LEN = 100
MAX_THINKING_LEN = 16000
MIN_SOLUTION_LEN = 20


def load_problems(source_path: str) -> list[str]:
    """Charge les problemes depuis un fichier JSONL."""
    problems: list[str] = []
    path = Path(source_path)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "messages" in data:
                for msg in data["messages"]:
                    if msg["role"] == "user":
                        problems.append(msg["content"])
            elif "problem" in data:
                problems.append(data["problem"])

    return problems


def normalize_thinking_tags(text: str) -> str:
    """Normalise les tags de raisonnement vers <thinking>...</thinking>."""
    text = re.sub(r"<think>", "<thinking>", text)
    text = re.sub(r"</think>", "</thinking>", text)
    return text


def extract_thinking_and_solution(response: str) -> tuple[str, str] | None:
    """Extrait le raisonnement et la solution."""
    response = normalize_thinking_tags(response)

    match = re.search(r"<thinking>(.*?)</thinking>\s*(.*)", response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        solution = match.group(2).strip()
        return thinking, solution
    return None


def generate_with_llamacpp(
    llamacpp_path: str,
    model_path: str,
    prompt: str,
    *,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    threads: int = 16,
) -> str | None:
    """Génère une réponse via llama-completion en mode CPU."""
    # Utiliser llama-completion (mode batch) au lieu de llama-cli (interactif)
    completion_path = llamacpp_path.replace("llama-cli", "llama-completion")
    cmd = [
        completion_path,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--top-p",
        "0.95",
        "--threads",
        str(threads),
        "--no-mmap",
        "-ngl",
        "0",
        "--log-disable",
        "--no-display-prompt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return result.stdout
    except subprocess.TimeoutExpired:
        print(" [timeout]", end="", flush=True)
    except Exception as e:
        print(f" [erreur: {e}]", end="", flush=True)

    return None


def passes_quality_filter(thinking: str, solution: str) -> bool:
    """Verifie les criteres de qualite minimaux."""
    if len(thinking) < MIN_THINKING_LEN:
        return False
    if len(thinking) > MAX_THINKING_LEN:
        return False
    if len(solution) < MIN_SOLUTION_LEN:
        return False
    return True


def save_splits(
    examples: list[dict], output_dir: Path, val_split: float
) -> None:
    """Sauvegarde les splits train/valid."""
    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Sauvegarde : {len(train_examples)} train, {len(valid_examples)} valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation CPU via llama.cpp")
    parser.add_argument("--llamacpp", required=True, help="Chemin vers llama-cli")
    parser.add_argument("--model", required=True, help="Chemin vers le modele GGUF")
    parser.add_argument(
        "--problems", required=True, help="Fichier de problemes (JSONL)"
    )
    parser.add_argument(
        "--output", required=True, help="Dossier de sortie dans data/"
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-problems", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    # Charger les problemes
    problems = load_problems(args.problems)
    if args.num_problems > 0:
        problems = problems[: args.num_problems]
    print(f"Charge {len(problems)} problemes")
    print(f"Threads CPU : {args.threads}")
    print(f"GPU layers : 0 (tout sur CPU)")

    # Dossier de sortie
    output_dir = Path("data") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reprise
    existing: list[dict] = []
    if args.resume and (output_dir / "all.jsonl").exists():
        with open(output_dir / "all.jsonl") as f:
            existing = [json.loads(line) for line in f if line.strip()]
        print(f"Reprise : {len(existing)} exemples existants")

    examples = list(existing)
    failed = 0

    for i, problem in enumerate(problems[len(existing) :], start=len(existing)):
        print(
            f"\r[{i + 1}/{len(problems)}] Generation CPU... ({failed} echecs)",
            end="",
            flush=True,
        )

        # Formater le prompt (format chat simple)
        prompt = (
            f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        )

        response = generate_with_llamacpp(
            args.llamacpp,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            threads=args.threads,
        )

        if response is None:
            failed += 1
            continue

        result = extract_thinking_and_solution(response)
        if result is None:
            failed += 1
            continue

        thinking, solution = result

        if not passes_quality_filter(thinking, solution):
            failed += 1
            continue

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
        with open(output_dir / "all.jsonl", "a") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nGenere {len(examples)} exemples ({failed} echecs)")

    # Split train/valid
    shuffled = list(examples)
    random.shuffle(shuffled)
    save_splits(shuffled, output_dir, args.val_split)

    print(f"Sortie : {output_dir}")


if __name__ == "__main__":
    main()
