#!/bin/bash
# =============================================================
# KIKI-Mac_tunner — Export: merge LoRA + convert GGUF + quantize
# =============================================================
# Usage:
#   ./export.sh                                 # Uses defaults from mistral-large config
#   ./export.sh --config configs/qwen-27b.yaml  # Other model
#   ./export.sh --quants Q4_K_M,Q6_K,Q8_0      # Custom quant list
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

CONFIG="configs/mistral-large.yaml"
QUANTS="Q6_K,Q8_0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --quants) QUANTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Parse config
MODEL_ID=$(grep "^model_id:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')
OUTPUT_DIR=$(grep "^output_dir:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')
MODEL_NAME=$(basename "$MODEL_ID")
MERGED_DIR="$SCRIPT_DIR/models/${MODEL_NAME}-Opus-Reasoning"

echo "=== KIKI-Mac_tunner Export ==="
echo "Base model: models/$MODEL_NAME"
echo "Adapter: $OUTPUT_DIR/final-lora"
echo "Merged → $MERGED_DIR"
echo "GGUF quants: $QUANTS"
echo ""

# Step 1: Merge LoRA
echo "[1/2] Merging LoRA adapter into base model..."
python3 "$SCRIPT_DIR/scripts/merge_lora.py" \
    --model "$SCRIPT_DIR/models/$MODEL_NAME" \
    --adapter "$SCRIPT_DIR/$OUTPUT_DIR/final-lora" \
    --output "$MERGED_DIR"

# Step 2: Convert to GGUF + quantize
echo ""
echo "[2/2] Converting to GGUF + quantizing..."

# Clone llama.cpp if needed
if [[ ! -d "$SCRIPT_DIR/llama.cpp" ]]; then
    echo "Cloning llama.cpp for conversion tools..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SCRIPT_DIR/llama.cpp"
fi

python3 "$SCRIPT_DIR/scripts/convert_gguf.py" \
    --model "$MERGED_DIR" \
    --output "$SCRIPT_DIR/models/gguf" \
    --quants "$QUANTS" \
    --llama-cpp "$SCRIPT_DIR/llama.cpp"

echo ""
echo "=== Export complete! ==="
echo "GGUF files ready in: $SCRIPT_DIR/models/gguf/"
echo ""
echo "To use with llama.cpp:"
echo "  llama-server --model models/gguf/${MODEL_NAME}-Opus-Reasoning-Q8_0.gguf --host 0.0.0.0"
echo ""
echo "To copy to your NFS share:"
echo "  cp models/gguf/*.gguf /mnt/models/"
