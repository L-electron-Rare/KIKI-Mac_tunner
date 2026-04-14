#!/bin/bash
# Distillation avec le Qwen3.5-35B-A3B-Opus (rapide, 3B actifs)
# Tourne en parallele avec la distillation 123B
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

TEACHER="models/Qwen3.5-35B-A3B-Opus-bf16"
OUTPUT="distilled-qwen35-35b-opus"

PROBLEMS=(
    "data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl"
    "data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl"
)

echo "=== Distillation 35B-Opus → donnees pour student ==="
echo "Teacher : $TEACHER"
echo "Sortie : data/$OUTPUT/"
echo ""

python scripts/distill_generate.py \
    --model "$TEACHER" \
    --problems "${PROBLEMS[@]}" \
    --output "$OUTPUT" \
    --max-tokens 2048 \
    --temperature 0.7 \
    "$@"
