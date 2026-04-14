#!/bin/bash
# Telecharger les datasets de raisonnement Opus 4.6
# Utilisation : ./scripts/download_datasets.sh [dataset|all]

set -e

# Activer le venv pour huggingface-cli
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

DATA_DIR="data"
mkdir -p "$DATA_DIR"

download_opus_3k() {
    echo "=== Dataset Opus 4.6 Reasoning 3K (actuel) ==="
    echo "Deja present dans $DATA_DIR/Opus-4.6-Reasoning-3000x-filtered/"
}

download_opus_10k() {
    echo "=== Dataset Opus 4.6 10K ==="
    hf download Roman1111111/claude-opus-4.6-10000x --repo-type dataset --local-dir "$DATA_DIR/claude-opus-4.6-10000x"
}

download_opus_12k() {
    echo "=== Dataset Opus 4.6 Reasoning SFT 12K ==="
    hf download ykarout/Opus-4.6-reasoning-sft-12k --repo-type dataset --local-dir "$DATA_DIR/Opus-4.6-reasoning-sft-12k"
}

case "${1:-help}" in
    opus-3k)   download_opus_3k ;;
    opus-10k)  download_opus_10k ;;
    opus-12k)  download_opus_12k ;;
    all)
        download_opus_3k
        download_opus_10k
        download_opus_12k
        ;;
    *)
        echo "Utilisation : $0 <dataset|all>"
        echo ""
        echo "Datasets disponibles :"
        echo "  opus-3k    Opus 4.6 Reasoning 3K (2326 ex, deja present)"
        echo "  opus-10k   Opus 4.6 10K (Roman1111111, ~10000 ex)"
        echo "  opus-12k   Opus 4.6 Reasoning SFT 12K (ykarout, ~12000 ex)"
        echo "  all        Tous les datasets"
        ;;
esac
