#!/bin/bash
# Generer des donnees de raisonnement avec un teacher sur CPU
# Permet de tourner en parallele avec le training MLX sur GPU
# Utilisation : ./scripts/generate_cpu.sh <model_gguf> <problems_jsonl> <output_dir> [num_problems]

set -e

LLAMACPP_CPU="/tmp/llama-cpp-cpu/build/bin"
MODEL="${1:?Utilisation : $0 <model.gguf> <problems.jsonl> <output_dir> [num_problems]}"
PROBLEMS="${2:?Fichier de problemes requis}"
OUTPUT_DIR="${3:?Dossier de sortie requis}"
NUM_PROBLEMS="${4:-0}"

# Verifier que llama-cli existe
if [ ! -f "$LLAMACPP_CPU/llama-cli" ]; then
    echo "ERREUR : llama-cli non trouve dans $LLAMACPP_CPU"
    echo "Lancer d'abord : cd /tmp/llama-cpp-cpu && cmake --build build -j\$(sysctl -n hw.ncpu)"
    exit 1
fi

echo "=== Generation CPU avec llama.cpp ==="
echo "Modele : $MODEL"
echo "Problemes : $PROBLEMS"
echo "Sortie : data/$OUTPUT_DIR/"
echo "Threads CPU : $(sysctl -n hw.ncpu)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

python scripts/generate_data_cpu.py \
    --llamacpp "$LLAMACPP_CPU/llama-cli" \
    --model "$MODEL" \
    --problems "$PROBLEMS" \
    --output "$OUTPUT_DIR" \
    --num-problems "$NUM_PROBLEMS" \
    --max-tokens 2048 \
    --threads "$(sysctl -n hw.performancecores 2>/dev/null || echo 16)"
