#!/bin/bash
# Telecharger des modeles GGUF pour llama.cpp (CPU ou GPU)
# Utilisation : ./scripts/download_gguf.sh <modele>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

MODELS_DIR="models/gguf"
mkdir -p "$MODELS_DIR"

download_qwen35_35b_opus_q4() {
    echo "=== Qwen3.5-35B-A3B-Opus Q4_K_M GGUF ==="
    hf download bartowski/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF \
        --include "*Q4_K_M*" \
        --local-dir "$MODELS_DIR/Qwen3.5-35B-A3B-Opus-Q4"
}

download_qwen35_27b_opus_q4() {
    echo "=== Qwen3.5-27B-Opus-v2 Q4_K_M GGUF ==="
    hf download bartowski/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF \
        --include "*Q4_K_M*" \
        --local-dir "$MODELS_DIR/Qwen3.5-27B-Opus-v2-Q4"
}

download_qwq_32b_q4() {
    echo "=== QwQ-32B Q4_K_M GGUF ==="
    hf download Qwen/QwQ-32B-GGUF \
        --include "*q4_k_m*" \
        --local-dir "$MODELS_DIR/QwQ-32B-Q4"
}

case "${1:-help}" in
    qwen35-35b-opus)  download_qwen35_35b_opus_q4 ;;
    qwen35-27b-opus)  download_qwen35_27b_opus_q4 ;;
    qwq-32b)          download_qwq_32b_q4 ;;
    all)
        download_qwen35_35b_opus_q4
        download_qwen35_27b_opus_q4
        download_qwq_32b_q4
        ;;
    *)
        echo "Utilisation : $0 <modele|all>"
        echo ""
        echo "Modeles GGUF disponibles :"
        echo "  qwen35-35b-opus   Qwen3.5-35B-A3B Opus Q4_K_M (~20 Go)"
        echo "  qwen35-27b-opus   Qwen3.5-27B Opus v2 Q4_K_M (~16 Go)"
        echo "  qwq-32b           QwQ-32B Q4_K_M (~18 Go)"
        echo "  all               Tous les GGUF (~54 Go)"
        ;;
esac
