#!/bin/bash
# Pipeline complet : train student 35B sur donnees distillees + export GGUF
# Utilisation : ./scripts/export_gguf.sh [train|fuse|convert|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

STUDENT_CONFIG="configs/mlx-lm-qwen35-35b-opus.yaml"
STUDENT_ADAPTER="output/qwen35-35b-opus-v3"
STUDENT_MODEL="models/Qwen3.5-35B-A3B-Opus-bf16"
FUSED_PATH="output/qwen35-35b-opus-fused"
GGUF_PATH="output/qwen35-35b-opus-fused.gguf"

train_student() {
    echo "=== Etape 1 : Training LoRA du student 35B ==="
    python -m mlx_lm lora --config "$STUDENT_CONFIG"
}

fuse_student() {
    echo "=== Etape 2 : Fusion LoRA dans le student ==="
    python -m mlx_lm fuse \
        --model "$STUDENT_MODEL" \
        --adapter-path "$STUDENT_ADAPTER/adapters.safetensors" \
        --save-path "$FUSED_PATH" \
        --de-quantize
}

convert_gguf() {
    echo "=== Etape 3 : Conversion en GGUF Q4_K_M ==="

    # Verifier que llama.cpp est disponible
    LLAMACPP="/tmp/llama-cpp-gpu/build/bin"
    if [ ! -f "$LLAMACPP/llama-quantize" ]; then
        echo "ERREUR : llama-quantize non trouve. Compiler llama.cpp d'abord."
        exit 1
    fi

    # Convertir safetensors → GGUF F16
    echo "Conversion safetensors → GGUF F16..."
    python "$LLAMACPP/../convert_hf_to_gguf.py" \
        "$FUSED_PATH" \
        --outfile "${GGUF_PATH%.gguf}-f16.gguf" \
        --outtype f16

    # Quantifier F16 → Q4_K_M
    echo "Quantification F16 → Q4_K_M..."
    "$LLAMACPP/llama-quantize" \
        "${GGUF_PATH%.gguf}-f16.gguf" \
        "$GGUF_PATH" \
        Q4_K_M

    # Taille finale
    echo ""
    echo "=== Modele final ==="
    ls -lh "$GGUF_PATH"
    echo "Pret pour deploiement (Ollama, llama.cpp, LM Studio)"
}

cleanup() {
    echo "=== Nettoyage des fichiers intermediaires ==="
    rm -f "${GGUF_PATH%.gguf}-f16.gguf"
    echo "F16 intermediaire supprime."
}

case "${1:-all}" in
    train)   train_student ;;
    fuse)    fuse_student ;;
    convert) convert_gguf ;;
    cleanup) cleanup ;;
    all)
        train_student
        fuse_student
        convert_gguf
        cleanup
        echo ""
        echo "=== Pipeline complet termine ==="
        echo "Modele final : $GGUF_PATH (~20 Go)"
        ;;
    *)
        echo "Utilisation : $0 [train|fuse|convert|cleanup|all]"
        echo ""
        echo "Etapes :"
        echo "  train    Training LoRA du student Qwen3.5-35B-A3B-Opus"
        echo "  fuse     Fusion LoRA dans le modele"
        echo "  convert  Conversion GGUF Q4_K_M (~20 Go)"
        echo "  cleanup  Supprime les fichiers intermediaires"
        echo "  all      Pipeline complet (defaut)"
        ;;
esac
