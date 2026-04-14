#!/bin/bash
# Distillation ANE rapide — Qwen3.5-9B-Opus sur Apple Neural Engine
# 8x plus rapide que subprocess, scoring parallele optionnel
# Utilisation : ./distill-ane.sh [--parallel 3] [--num-problems N] [--resume]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

MODEL_DIR="/tmp/anemll-qwen35-9b-opus"
OUTPUT="distilled-ane-9b-opus"

PROBLEMS=(
    "data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl"
    "data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl"
)

echo "=== Distillation ANE — Qwen3.5-9B-Opus ==="
echo "Modele : $MODEL_DIR"
echo "Sortie : data/$OUTPUT/"
echo ""

python scripts/distill_ane.py \
    --model-dir "$MODEL_DIR" \
    --problems "${PROBLEMS[@]}" \
    --output "$OUTPUT" \
    --max-tokens 2048 \
    --temperature 0.7 \
    "$@"
