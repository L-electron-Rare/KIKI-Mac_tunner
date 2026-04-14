#!/bin/bash
# Distillation : genere des traces de raisonnement avec le teacher 123B fuse
# Utilisation : ./distill.sh [--num-problems N] [--resume]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

TEACHER="output/mistral-large-opus-fused"
OUTPUT="distilled-mistral-large-123b"

# Utiliser tous les problemes disponibles
PROBLEMS=(
    "data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl"
    "data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl"
)

echo "=== Distillation 123B → donnees pour student ==="
echo "Teacher : $TEACHER"
echo "Problemes : ${PROBLEMS[*]}"
echo "Sortie : data/$OUTPUT/"
echo ""

python scripts/distill_generate.py \
    --model "$TEACHER" \
    --problems "${PROBLEMS[@]}" \
    --output "$OUTPUT" \
    --max-tokens 2048 \
    --temperature 0.7 \
    "$@"
