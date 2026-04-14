#!/usr/bin/env python3
"""Convertit les datasets téléchargés au format chat JSONL standard.

Chaque dataset a un format différent. Ce script les normalise tous vers :
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n\n..."}]}
"""

import json
import re
from pathlib import Path


def normalize_thinking(text: str) -> str:
    """Normalise les tags de raisonnement vers <thinking>...</thinking>."""
    text = re.sub(r'<think>', '<thinking>', text)
    text = re.sub(r'</think>', '</thinking>', text)
    return text


def has_thinking(text: str) -> bool:
    """Vérifie si le texte contient une trace de raisonnement."""
    return '<thinking>' in text and '</thinking>' in text


def convert_messages_format(data: dict) -> dict | None:
    """Convertit un exemple au format messages standard."""
    if 'messages' in data:
        msgs = data['messages']
        if len(msgs) >= 2:
            user_msg = None
            assistant_msg = None
            for msg in msgs:
                if msg['role'] == 'user' and user_msg is None:
                    user_msg = msg['content']
                elif msg['role'] == 'assistant' and assistant_msg is None:
                    assistant_msg = normalize_thinking(msg['content'])
            if user_msg and assistant_msg and has_thinking(assistant_msg):
                return {
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                }

    # Format problem/thinking/solution
    if 'problem' in data and 'thinking' in data and 'solution' in data:
        thinking = data['thinking'].strip()
        solution = data['solution'].strip()
        if thinking and solution:
            return {
                "messages": [
                    {"role": "user", "content": data['problem']},
                    {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"}
                ]
            }

    # Format prompt/response ou instruction/output
    prompt_key = next((k for k in ('prompt', 'instruction', 'input', 'question') if k in data), None)
    response_key = next((k for k in ('response', 'output', 'answer', 'completion') if k in data), None)
    if prompt_key and response_key:
        response = normalize_thinking(data[response_key])
        if has_thinking(response):
            return {
                "messages": [
                    {"role": "user", "content": data[prompt_key]},
                    {"role": "assistant", "content": response}
                ]
            }

    # Format conversations (ShareGPT-like)
    if 'conversations' in data:
        convs = data['conversations']
        if len(convs) >= 2:
            user_content = convs[0].get('value', convs[0].get('content', ''))
            assistant_content = normalize_thinking(convs[1].get('value', convs[1].get('content', '')))
            if user_content and assistant_content and has_thinking(assistant_content):
                return {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }

    return None


def convert_dataset(input_dir: Path, output_dir: Path) -> int:
    """Convertit un dataset vers le format chat JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []

    # Chercher tous les fichiers de données
    for pattern in ('*.jsonl', '*.json', 'train.*', 'data.*'):
        for filepath in sorted(input_dir.glob(pattern)):
            if filepath.suffix == '.jsonl':
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            result = convert_messages_format(data)
                            if result:
                                examples.append(result)
                        except json.JSONDecodeError:
                            continue
            elif filepath.suffix == '.json':
                with open(filepath) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            result = convert_messages_format(item)
                            if result:
                                examples.append(result)

    # Chercher les fichiers parquet
    try:
        import pyarrow.parquet as pq
        for filepath in sorted(input_dir.glob('**/*.parquet')):
            table = pq.read_table(filepath)
            for row in table.to_pylist():
                result = convert_messages_format(row)
                if result:
                    examples.append(result)
    except ImportError:
        pass

    if not examples:
        print(f"  Aucun exemple trouvé dans {input_dir}")
        return 0

    # Écrire train.jsonl
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  {len(examples)} exemples convertis → {output_dir}")
    return len(examples)


def main():
    data_dir = Path("data")
    total = 0

    # Dataset 10K
    src = data_dir / "claude-opus-4.6-10000x"
    if src.exists():
        print(f"Conversion de {src.name}...")
        total += convert_dataset(src, data_dir / "claude-opus-4.6-10000x-chat")

    # Dataset 12K
    src = data_dir / "Opus-4.6-reasoning-sft-12k"
    if src.exists():
        print(f"Conversion de {src.name}...")
        total += convert_dataset(src, data_dir / "Opus-4.6-reasoning-sft-12k-chat")

    print(f"\nTotal : {total} exemples convertis")


if __name__ == "__main__":
    main()
