#!/usr/bin/env python3
"""Prepare les donnees GRPO : prompts math/code avec reponses verifiables."""
import json
import random
from pathlib import Path

def generate_math_prompts(n=1000):
    """Genere des problemes math avec reponse attendue."""
    prompts = []
    random.seed(42)

    for _ in range(n):
        # Arithmetique
        a, b = random.randint(10, 999), random.randint(10, 999)
        op = random.choice(['+', '-', '*'])
        answer = eval(f"{a} {op} {b}")
        prompts.append({
            "prompt": f"Calculate {a} {op} {b}. Show your reasoning step by step.",
            "expected_answer": str(answer),
            "type": "math"
        })

    return prompts

def generate_code_prompts(n=200):
    """Genere des problemes code avec test."""
    prompts = []
    templates = [
        {
            "prompt": "Write a Python function `is_prime(n)` that returns True if n is prime.",
            "test": "assert is_prime(7) == True\nassert is_prime(4) == False\nassert is_prime(2) == True\nprint('OK')",
        },
        {
            "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number.",
            "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nprint('OK')",
        },
        {
            "prompt": "Write a Python function `reverse_string(s)` that reverses a string.",
            "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nprint('OK')",
        },
    ]

    for t in templates * (n // len(templates)):
        prompts.append({
            "prompt": t["prompt"],
            "test_code": t["test"],
            "type": "code"
        })

    return prompts

def main():
    out_dir = Path("data/grpo-prompts")
    out_dir.mkdir(parents=True, exist_ok=True)

    math = generate_math_prompts(1000)
    code = generate_code_prompts(200)

    all_prompts = math + code
    random.shuffle(all_prompts)

    with open(out_dir / "train.jsonl", "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Genere {len(all_prompts)} prompts GRPO")
    print(f"  Math: {len(math)}")
    print(f"  Code: {len(code)}")
    print(f"  Output: {out_dir}/train.jsonl")

if __name__ == "__main__":
    main()
