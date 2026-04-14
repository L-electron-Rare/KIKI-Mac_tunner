#!/bin/bash
# Distillation 35B-Opus — batch 2 (problemes 3001-6000)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

TEACHER="models/Qwen3.5-35B-A3B-Opus-bf16"
OUTPUT="distilled-qwen35-35b-opus-batch2"

PROBLEMS=(
    "data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl"
    "data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl"
)

echo "=== Distillation 35B-Opus Batch 2 ==="
python scripts/distill_generate.py \
    --model "$TEACHER" \
    --problems "${PROBLEMS[@]}" \
    --output "$OUTPUT" \
    --max-tokens 2048 \
    --temperature 0.8 \
    --seed 1337 \
    "$@"
