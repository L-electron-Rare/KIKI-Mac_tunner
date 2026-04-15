#!/usr/bin/env python3
"""Prépare le dataset de coding pour le fine-tuning Sonnet-Devstral.

Charge les datasets bruts téléchargés, les convertit au format chat JSONL,
déduplique, filtre par qualité et langages cibles, et produit un split train/valid.

Langages cibles : Python, TypeScript, Rust, Go
Objectif : 15-20K exemples de haute qualité avec traces de raisonnement
Sortie : data/sonnet-coding/train.jsonl + valid.jsonl
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

# Langages cibles pour le filtrage
TARGET_LANGUAGES = {"python", "typescript", "rust", "go", "javascript"}

# Mots-clés pour détecter les langages dans le contenu
LANG_PATTERNS = {
    "python": re.compile(r'\b(def |import |class |print\(|python|\.py\b)', re.IGNORECASE),
    "typescript": re.compile(r'\b(interface |type |const |=>|typescript|\.ts\b|\.tsx\b)', re.IGNORECASE),
    "rust": re.compile(r'\b(fn |let mut|impl |use |pub |rust|\.rs\b)', re.IGNORECASE),
    "go": re.compile(r'\b(func |package |import \(|go\b|\.go\b|goroutine)', re.IGNORECASE),
    "javascript": re.compile(r'\b(function |const |let |var |=>|\.js\b|javascript)', re.IGNORECASE),
}


def detect_language(text: str) -> str | None:
    """Détecte le langage principal dans un texte de code."""
    scores = {}
    for lang, pattern in LANG_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            scores[lang] = len(matches)
    if not scores:
        return None
    return max(scores, key=scores.get)


def normalize_thinking(text: str) -> str:
    """Normalise les tags de raisonnement vers <thinking>...</thinking>."""
    text = re.sub(r'<think>', '<thinking>', text)
    text = re.sub(r'</think>', '</thinking>', text)
    return text


def has_reasoning(text: str) -> bool:
    """Vérifie si le texte contient une trace de raisonnement."""
    return bool(
        ('<thinking>' in text and '</thinking>' in text)
        or ('<think>' in text and '</think>' in text)
        or ('## ' in text and len(text) > 500)  # Raisonnement structuré sans tags
    )


def dedup_key(user: str, assistant: str) -> str:
    """Génère une clé de déduplication."""
    content = user.strip()[:500] + "\n###\n" + assistant.strip()[:500]
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_jsonl(filepath: Path) -> list[dict]:
    """Charge un fichier JSONL."""
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def load_json(filepath: Path) -> list[dict]:
    """Charge un fichier JSON (liste ou objet)."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def load_parquet(filepath: Path) -> list[dict]:
    """Charge un fichier Parquet."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(filepath)
        return table.to_pylist()
    except ImportError:
        print("  pyarrow non disponible, skip parquet")
        return []


def load_all_files(directory: Path) -> list[dict]:
    """Charge tous les fichiers de données d'un répertoire."""
    examples = []
    if not directory.exists():
        return examples

    for filepath in sorted(directory.rglob("*")):
        if filepath.suffix == ".jsonl":
            examples.extend(load_jsonl(filepath))
        elif filepath.suffix == ".json" and filepath.name != "dataset_info.json":
            examples.extend(load_json(filepath))
        elif filepath.suffix == ".parquet":
            examples.extend(load_parquet(filepath))
    return examples


def convert_opencodeinstruct(raw: dict) -> dict | None:
    """Convertit un exemple OpenCodeInstruct au format chat."""
    # Format : question, solution, (optionnel) reasoning
    question = raw.get("question") or raw.get("instruction") or raw.get("prompt", "")
    solution = raw.get("solution") or raw.get("output") or raw.get("response", "")
    reasoning = raw.get("reasoning", "")

    if not question or not solution:
        return None

    if reasoning:
        content = f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{solution.strip()}"
    else:
        content = solution.strip()

    return {
        "messages": [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": normalize_thinking(content)},
        ]
    }


def convert_opencodereasoning(raw: dict) -> dict | None:
    """Convertit un exemple OpenCodeReasoning (traces R1)."""
    input_text = raw.get("input", "")
    output_text = raw.get("output", "")

    if not input_text or not output_text:
        return None

    return {
        "messages": [
            {"role": "user", "content": input_text.strip()},
            {"role": "assistant", "content": normalize_thinking(output_text.strip())},
        ]
    }


def convert_codeforces_cots(raw: dict) -> dict | None:
    """Convertit un exemple Codeforces CoTs (chaînes de raisonnement)."""
    # Format : messages ou problem + solution
    if "messages" in raw:
        msgs = raw["messages"]
        if len(msgs) >= 2:
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if user and assistant:
                return {
                    "messages": [
                        {"role": "user", "content": user.strip()},
                        {"role": "assistant", "content": normalize_thinking(assistant.strip())},
                    ]
                }

    problem = raw.get("problem", "") or raw.get("question", "")
    solution = raw.get("solution", "") or raw.get("output", "")
    thinking = raw.get("thinking", "") or raw.get("reasoning", "")

    if not problem or not solution:
        return None

    if thinking:
        content = f"<thinking>\n{thinking.strip()}\n</thinking>\n\n{solution.strip()}"
    else:
        content = normalize_thinking(solution.strip())

    return {
        "messages": [
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": content},
        ]
    }


def convert_magicoder(raw: dict) -> dict | None:
    """Convertit un exemple Magicoder OSS-Instruct."""
    problem = raw.get("problem", "") or raw.get("instruction", "")
    solution = raw.get("solution", "") or raw.get("output", "")

    if not problem or not solution:
        return None

    return {
        "messages": [
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": solution.strip()},
        ]
    }


def convert_codefeedback(raw: dict) -> dict | None:
    """Convertit un exemple CodeFeedback."""
    query = raw.get("query", "") or raw.get("instruction", "") or raw.get("input", "")
    answer = raw.get("answer", "") or raw.get("output", "") or raw.get("response", "")

    if not query or not answer:
        return None

    return {
        "messages": [
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def convert_swe_trajectories(raw: dict) -> dict | None:
    """Convertit une trajectoire SWE agentic en conversation."""
    # Format variable — messages ou trajectory
    if "messages" in raw:
        msgs = raw["messages"]
        if len(msgs) >= 2:
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if user and assistant:
                return {
                    "messages": [
                        {"role": "user", "content": user.strip()},
                        {"role": "assistant", "content": normalize_thinking(assistant.strip())},
                    ]
                }

    # Format issue + patch
    issue = raw.get("problem_statement", "") or raw.get("issue", "")
    patch = raw.get("patch", "") or raw.get("solution", "")
    if issue and patch:
        return {
            "messages": [
                {"role": "user", "content": f"Fix this issue:\n\n{issue.strip()}"},
                {"role": "assistant", "content": patch.strip()},
            ]
        }

    return None


def convert_generic(raw: dict) -> dict | None:
    """Convertisseur générique pour les formats inconnus."""
    # Format messages
    if "messages" in raw:
        msgs = raw["messages"]
        if len(msgs) >= 2:
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if user and assistant:
                return {
                    "messages": [
                        {"role": "user", "content": user.strip()},
                        {"role": "assistant", "content": normalize_thinking(assistant.strip())},
                    ]
                }

    # Format prompt/response
    prompt_key = next((k for k in ("prompt", "instruction", "input", "question", "problem", "query") if k in raw), None)
    response_key = next((k for k in ("response", "output", "answer", "solution", "completion") if k in raw), None)
    if prompt_key and response_key and raw[prompt_key] and raw[response_key]:
        return {
            "messages": [
                {"role": "user", "content": str(raw[prompt_key]).strip()},
                {"role": "assistant", "content": normalize_thinking(str(raw[response_key]).strip())},
            ]
        }

    return None


# Mapping source → convertisseur
SOURCE_CONVERTERS = {
    "OpenCodeReasoning": convert_opencodereasoning,
    "OpenCodeInstruct": convert_opencodeinstruct,
    "codeforces-cots": convert_codeforces_cots,
    "Magicoder-OSS-Instruct-75K": convert_magicoder,
    "CodeFeedback-Filtered-Instruction": convert_codefeedback,
    "OpenHands-Sampled-Trajectories": convert_swe_trajectories,
    "Nemotron-SWE-v1": convert_swe_trajectories,
}

# Quotas par source pour équilibrer le dataset final (~15-20K)
SOURCE_QUOTAS = {
    "OpenCodeReasoning": 5000,        # Traces de raisonnement R1 — haute valeur
    "OpenCodeInstruct": 4000,         # Large diversité, filtrer les meilleurs
    "codeforces-cots": 3000,          # Raisonnement compétitif
    "Magicoder-OSS-Instruct-75K": 2000,  # Instructions OSS diversifiées
    "CodeFeedback-Filtered-Instruction": 2000,  # Haute qualité filtrée
    "OpenHands-Sampled-Trajectories": 500,  # Trajectoires agentic (petit mais précieux)
    "Nemotron-SWE-v1": 3000,          # Trajectoires agentic large
}


def process_source(
    source_name: str,
    raw_dir: Path,
    quota: int,
    prefer_reasoning: bool = True,
) -> list[dict]:
    """Charge et convertit les exemples d'une source avec quota."""
    converter = SOURCE_CONVERTERS.get(source_name, convert_generic)
    raw_examples = load_all_files(raw_dir)

    if not raw_examples:
        print(f"  {source_name}: aucun fichier trouvé dans {raw_dir}")
        return []

    print(f"  {source_name}: {len(raw_examples)} exemples bruts chargés")

    # Convertir
    converted = []
    for raw in raw_examples:
        result = converter(raw)
        if result:
            converted.append(result)

    print(f"  {source_name}: {len(converted)} exemples convertis")

    if not converted:
        return []

    # Prioriser les exemples avec raisonnement
    if prefer_reasoning:
        with_reasoning = [
            ex for ex in converted
            if has_reasoning(ex["messages"][-1]["content"])
        ]
        without_reasoning = [
            ex for ex in converted
            if not has_reasoning(ex["messages"][-1]["content"])
        ]

        # Mélanger pour diversité
        random.shuffle(with_reasoning)
        random.shuffle(without_reasoning)

        # Prendre d'abord ceux avec raisonnement, puis compléter
        selected = with_reasoning[:quota]
        remaining_quota = quota - len(selected)
        if remaining_quota > 0:
            selected.extend(without_reasoning[:remaining_quota])

        print(f"  {source_name}: {len(with_reasoning)} avec raisonnement, {len(selected)} sélectionnés (quota {quota})")
    else:
        random.shuffle(converted)
        selected = converted[:quota]
        print(f"  {source_name}: {len(selected)} sélectionnés (quota {quota})")

    return selected


def filter_by_language(examples: list[dict]) -> list[dict]:
    """Filtre les exemples pour garder les langages cibles."""
    filtered = []
    lang_counts = {}

    for ex in examples:
        content = ex["messages"][-1]["content"]
        lang = detect_language(content)

        # Garder les exemples dans les langages cibles ou non détectés (souvent multi-lang)
        if lang is None or lang in TARGET_LANGUAGES:
            filtered.append(ex)
            key = lang or "unknown"
            lang_counts[key] = lang_counts.get(key, 0) + 1

    print(f"\nDistribution langages après filtrage :")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    return filtered


def filter_by_quality(examples: list[dict], min_length: int = 100, max_length: int = 32000) -> list[dict]:
    """Filtre par qualité basique (longueur, contenu)."""
    filtered = []
    for ex in examples:
        assistant_content = ex["messages"][-1]["content"]
        user_content = ex["messages"][0]["content"]

        # Trop court = pas utile
        if len(assistant_content) < min_length:
            continue
        # Trop long = dépasse le contexte
        if len(assistant_content) > max_length:
            continue
        # Prompt trop court
        if len(user_content) < 20:
            continue

        filtered.append(ex)

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset Sonnet-coding")
    parser.add_argument("--raw-dir", default="data/sonnet-coding-raw",
                        help="Répertoire des datasets bruts téléchargés")
    parser.add_argument("--output-dir", default="data/sonnet-coding",
                        help="Répertoire de sortie")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Ratio de validation (défaut: 5%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire")
    parser.add_argument("--target-size", type=int, default=18000,
                        help="Nombre d'exemples cible")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Préparation dataset Sonnet-coding ===\n")

    # Charger et convertir chaque source
    all_examples = []
    for source_name, quota in SOURCE_QUOTAS.items():
        source_dir = raw_dir / source_name
        examples = process_source(source_name, source_dir, quota)
        all_examples.extend(examples)

    print(f"\nTotal avant filtrage : {len(all_examples)}")

    # Déduplication
    seen_keys: set[str] = set()
    deduped = []
    for ex in all_examples:
        user = ex["messages"][0]["content"]
        assistant = ex["messages"][-1]["content"]
        key = dedup_key(user, assistant)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(ex)

    print(f"Après déduplication : {len(deduped)} (supprimé {len(all_examples) - len(deduped)} doublons)")

    # Filtrage qualité
    quality = filter_by_quality(deduped)
    print(f"Après filtrage qualité : {len(quality)}")

    # Filtrage langages
    filtered = filter_by_language(quality)
    print(f"Après filtrage langages : {len(filtered)}")

    # Limiter au target si nécessaire
    if len(filtered) > args.target_size:
        random.shuffle(filtered)
        filtered = filtered[:args.target_size]
        print(f"Limité à {args.target_size} exemples")

    # Split train/valid
    random.shuffle(filtered)
    split_idx = int(len(filtered) * (1 - args.val_split))
    train = filtered[:split_idx]
    valid = filtered[split_idx:]

    # Écriture
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n=== Dataset Sonnet-coding prêt ===")
    print(f"Train : {len(train)} exemples")
    print(f"Valid : {len(valid)} exemples")
    print(f"Sortie : {output_dir}")
    print(f"\nLancer le training :")
    print(f"  python scripts/train_devstral_sonnet.py")


if __name__ == "__main__":
    main()
