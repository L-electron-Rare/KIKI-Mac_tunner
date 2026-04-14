#!/usr/bin/env python3
"""Distillation rapide avec mlx-vlm (~45 tok/s).

Utilise mlx-vlm natif pour générer des traces de raisonnement
à partir du Qwen3.5-35B-A3B 4bit. 3-6x plus rapide que notre
implémentation MLX custom.
"""

import json
import time
import random
import re
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def load_problems(sources, max_problems=5000):
    problems = []
    seen = set()
    for src in sources:
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if 'messages' in d:
                    for m in d['messages']:
                        if m.get('role') == 'user':
                            c = m['content']
                            if c not in seen:
                                problems.append(c)
                                seen.add(c)
                            break
    random.seed(42)
    random.shuffle(problems)
    return problems[:max_problems]


def extract_thinking(response):
    """Extrait thinking/solution ou wrappe la réponse."""
    response = re.sub(r'<think>', '<thinking>', response)
    response = re.sub(r'</think>', '</thinking>', response)

    match = re.search(r'<thinking>(.*?)</thinking>\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    response = response.strip()
    if len(response) < 100:
        return None, None

    split = re.search(
        r'\n\n(?:(?:The |So |Therefore|Thus|Hence|Final)[^\n]*)', response
    )
    if split:
        return response[:split.start()].strip(), response[split.start():].strip()

    lines = response.split('\n')
    sp = max(1, int(len(lines) * 0.8))
    return '\n'.join(lines[:sp]).strip(), '\n'.join(lines[sp:]).strip()


def main():
    print("Chargement mlx-vlm Qwen3.5-35B-A3B 4bit...")
    model, processor = load("models/Qwen3.5-35B-A3B-Opus-vlm")
    config = load_config("models/Qwen3.5-35B-A3B-Opus-vlm")
    print("OK")

    problems = load_problems([
        "data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl",
        "data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl",
    ], max_problems=5000)
    print(f"{len(problems)} problèmes")

    output_dir = Path("data/distilled-mlxvlm-35b")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_file = output_dir / "all.jsonl"

    # Resume
    existing = 0
    if all_file.exists():
        with open(all_file) as f:
            existing = sum(1 for _ in f)
        print(f"Reprise depuis {existing}")

    ok = existing
    failed = 0
    t_start = time.time()

    for i, problem in enumerate(problems[existing:], start=existing):
        try:
            # Prompt direct ChatML (apply_chat_template a des bugs avec ce modèle)
            prompt_text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
            result = generate(model, processor, prompt_text, max_tokens=2048, verbose=False)
            response = result.text if hasattr(result, 'text') else str(result)

            thinking, solution = extract_thinking(response)
            if not thinking or len(thinking) < 50 or not solution or len(solution) < 10:
                failed += 1
                continue

            example = {"messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"}
            ]}

            with open(all_file, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            ok += 1

            elapsed = time.time() - t_start
            rate = (ok - existing) / elapsed * 3600 if elapsed > 0 else 0
            if ok % 10 == 0:
                print(f"[{ok}/{len(problems)}] {rate:.0f} ex/h, {failed} échecs, "
                      f"{elapsed/60:.0f} min")

        except Exception as e:
            failed += 1
            if failed % 50 == 0:
                print(f"Erreur #{failed}: {e}")

    # Split train/valid
    print(f"\nTerminé: {ok} exemples, {failed} échecs")
    all_examples = []
    with open(all_file) as f:
        all_examples = [json.loads(l) for l in f if l.strip()]
    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.95)
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in all_examples[:split]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in all_examples[split:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Sauvé: {split} train, {len(all_examples)-split} valid")


if __name__ == "__main__":
    main()
