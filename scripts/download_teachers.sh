#!/bin/bash
# Telecharger les modeles teachers pour la generation de donnees synthetiques
# Utilisation : ./scripts/download_teachers.sh [nom_modele|all]

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

download_qwq_32b() {
    echo "=== Telechargement de QwQ-32B bf16 (MLX) ==="
    hf download mlx-community/QwQ-32B-bf16 --local-dir "$MODELS_DIR/QwQ-32B-bf16"
}

download_deepseek_r1_distill_32b() {
    echo "=== Telechargement de DeepSeek-R1-Distill-Qwen-32B bf16 (MLX) ==="
    hf download mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16 --local-dir "$MODELS_DIR/DeepSeek-R1-Distill-Qwen-32B-bf16"
}

download_deepseek_r1_distill_70b() {
    echo "=== Telechargement de DeepSeek-R1-Distill-Llama-70B bf16 (MLX) ==="
    hf download mlx-community/DeepSeek-R1-Distill-Llama-70B-bf16 --local-dir "$MODELS_DIR/DeepSeek-R1-Distill-Llama-70B-bf16"
}

download_qwen3_72b() {
    echo "=== Telechargement de Qwen3-72B bf16 (MLX) ==="
    hf download mlx-community/Qwen3-72B-bf16 --local-dir "$MODELS_DIR/Qwen3-72B-bf16"
}

download_qwen3_235b() {
    echo "=== Telechargement de Qwen3-235B-A22B bf16 (MLX) ==="
    hf download mlx-community/Qwen3-235B-A22B-bf16 --local-dir "$MODELS_DIR/Qwen3-235B-A22B-bf16"
}

download_deepseek_r1_671b() {
    echo "=== Telechargement de DeepSeek-R1 671B 4-bit (MLX) ==="
    echo "ATTENTION : Ce modele fait ~335 Go. Verifiez l'espace disque disponible."
    hf download mlx-community/DeepSeek-R1-4bit --local-dir "$MODELS_DIR/DeepSeek-R1-4bit"
}

download_qwen35_397b() {
    echo "=== Telechargement Qwen3.5-397B-A17B bf16 (~350 Go) ==="
    hf download Qwen/Qwen3.5-397B-A17B --local-dir "$MODELS_DIR/Qwen3.5-397B-A17B-bf16"
}

download_qwen35_122b() {
    echo "=== Telechargement Qwen3.5-122B-A10B bf16 (~130 Go) ==="
    hf download Qwen/Qwen3.5-122B-A10B --local-dir "$MODELS_DIR/Qwen3.5-122B-A10B-bf16"
}

download_qwen35_27b_opus() {
    echo "=== Telechargement Qwen3.5-27B Opus Distilled v2 (~56 Go) ==="
    hf download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 --local-dir "$MODELS_DIR/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2"
}

download_devstral2_123b() {
    echo "=== Telechargement Devstral 2 123B (~246 Go) ==="
    hf download mistralai/Devstral-2-123B-Instruct-2512 --local-dir "$MODELS_DIR/Devstral-2-123B-Instruct"
}

case "${1:-help}" in
    qwq-32b)           download_qwq_32b ;;
    deepseek-r1-32b)   download_deepseek_r1_distill_32b ;;
    deepseek-r1-70b)   download_deepseek_r1_distill_70b ;;
    qwen3-72b)         download_qwen3_72b ;;
    qwen3-235b)        download_qwen3_235b ;;
    deepseek-r1-671b)  download_deepseek_r1_671b ;;
    qwen35-397b)       download_qwen35_397b ;;
    qwen35-122b)       download_qwen35_122b ;;
    qwen35-27b-opus)   download_qwen35_27b_opus ;;
    devstral2-123b)    download_devstral2_123b ;;
    students)
        download_qwq_32b
        download_deepseek_r1_distill_32b
        ;;
    teachers-medium)
        download_qwen3_72b
        download_deepseek_r1_distill_70b
        ;;
    teachers-large)
        download_qwen3_235b
        download_deepseek_r1_671b
        ;;
    teachers-2026)
        download_qwen35_397b
        download_qwen35_122b
        ;;
    students-opus)
        download_qwen35_27b_opus
        download_devstral2_123b
        ;;
    all)
        download_qwq_32b
        download_deepseek_r1_distill_32b
        download_deepseek_r1_distill_70b
        download_qwen3_72b
        download_qwen3_235b
        download_deepseek_r1_671b
        download_qwen35_397b
        download_qwen35_122b
        download_qwen35_27b_opus
        download_devstral2_123b
        ;;
    *)
        echo "Utilisation : $0 <modele|groupe>"
        echo ""
        echo "Modeles individuels :"
        echo "  qwq-32b           QwQ-32B bf16 (~65 Go)"
        echo "  deepseek-r1-32b   DeepSeek-R1-Distill-Qwen-32B bf16 (~65 Go)"
        echo "  deepseek-r1-70b   DeepSeek-R1-Distill-Llama-70B bf16 (~140 Go)"
        echo "  qwen3-72b         Qwen3-72B bf16 (~145 Go)"
        echo "  qwen3-235b        Qwen3-235B-A22B bf16 (~200 Go)"
        echo "  deepseek-r1-671b  DeepSeek-R1 671B 4-bit (~335 Go)"
        echo "  qwen35-397b       Qwen3.5-397B-A17B bf16 (~350 Go)"
        echo "  qwen35-122b       Qwen3.5-122B-A10B bf16 (~130 Go)"
        echo "  qwen35-27b-opus   Qwen3.5-27B Opus Distilled v2 (~56 Go)"
        echo "  devstral2-123b    Devstral 2 123B (~246 Go)"
        echo ""
        echo "Groupes :"
        echo "  students          Modeles 32B pour le training LoRA"
        echo "  teachers-medium   Modeles 70B pour la generation de donnees"
        echo "  teachers-large    Modeles 235B+ pour la generation de donnees"
        echo "  teachers-2026     Qwen3.5 397B + 122B (Fev 2026)"
        echo "  students-opus     Qwen3.5-27B-Opus + Devstral2 123B"
        echo "  all               Tout (~1700 Go au total)"
        ;;
esac
