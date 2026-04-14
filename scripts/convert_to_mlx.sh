#!/bin/bash
# Convertir des modeles HuggingFace au format MLX bf16
# Utilisation : ./scripts/convert_to_mlx.sh <modele>

set -e
source .venv/bin/activate

MODELS_DIR="models"

cleanup_hf_cache() {
    # Supprime le cache HuggingFace du modele source apres conversion
    local REPO="$1"
    local CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    local SAFE_NAME=$(echo "$REPO" | tr '/' '--')
    local CACHE_PATH="$CACHE_DIR/models--$SAFE_NAME"

    if [ -d "$CACHE_PATH" ]; then
        local SIZE=$(du -sh "$CACHE_PATH" 2>/dev/null | cut -f1)
        echo "Nettoyage du cache HF : $CACHE_PATH ($SIZE)"
        rm -rf "$CACHE_PATH"
        echo "Cache supprime."
    else
        echo "Pas de cache a nettoyer pour $REPO"
    fi
}

convert_qwen35_35b_opus() {
    local SRC="Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
    local DST="$MODELS_DIR/Qwen3.5-35B-A3B-Opus-bf16"

    if [ -d "$DST" ]; then
        echo "Deja converti : $DST"
        return
    fi

    echo "=== Conversion Qwen3.5-35B-A3B-Opus vers MLX bf16 ==="
    echo "Source : $SRC"
    echo "Destination : $DST"
    echo "Taille estimee : ~70 Go"
    echo ""

    python -m mlx_lm.convert \
        --hf-path "$SRC" \
        --mlx-path "$DST" \
        --dtype bfloat16

    echo "Conversion terminee : $DST"
    cleanup_hf_cache "$SRC"
}

convert_qwen35_397b() {
    local SRC="Qwen/Qwen3.5-397B-A17B"
    local DST="$MODELS_DIR/Qwen3.5-397B-A17B-bf16"

    if [ -d "$DST" ]; then
        echo "Deja converti : $DST"
        return
    fi

    echo "=== Conversion Qwen3.5-397B-A17B vers MLX bf16 ==="
    echo "Source : $SRC"
    echo "Destination : $DST"
    echo "ATTENTION : Ce modele fait ~794 Go en bf16. Necessite ~800 Go d'espace disque."
    echo "La conversion va telecharger puis convertir. Cela peut prendre plusieurs heures."
    echo ""

    python -m mlx_lm.convert \
        --hf-path "$SRC" \
        --mlx-path "$DST" \
        --dtype bfloat16

    echo "Conversion terminee : $DST"
    cleanup_hf_cache "$SRC"
}

convert_qwen35_397b_4bit() {
    local SRC="Qwen/Qwen3.5-397B-A17B"
    local DST="$MODELS_DIR/Qwen3.5-397B-A17B-4bit"

    if [ -d "$DST" ]; then
        echo "Deja converti : $DST"
        return
    fi

    echo "=== Conversion Qwen3.5-397B-A17B vers MLX 4-bit ==="
    echo "Source : $SRC"
    echo "Destination : $DST"
    echo "Taille estimee : ~222 Go (4-bit)"
    echo ""

    python -m mlx_lm.convert \
        --hf-path "$SRC" \
        --mlx-path "$DST" \
        --dtype bfloat16 \
        -q \
        --q-bits 4

    echo "Conversion terminee : $DST"
    cleanup_hf_cache "$SRC"
}

case "${1:-help}" in
    qwen35-35b-opus)       convert_qwen35_35b_opus ;;
    qwen35-397b)           convert_qwen35_397b ;;
    qwen35-397b-4bit)      convert_qwen35_397b_4bit ;;
    all)
        convert_qwen35_35b_opus
        convert_qwen35_397b
        ;;
    *)
        echo "Utilisation : $0 <modele>"
        echo ""
        echo "Modeles a convertir :"
        echo "  qwen35-35b-opus    Qwen3.5-35B-A3B Opus Distilled → MLX bf16 (~70 Go)"
        echo "  qwen35-397b        Qwen3.5-397B-A17B → MLX bf16 (~794 Go)"
        echo "  qwen35-397b-4bit   Qwen3.5-397B-A17B → MLX 4-bit (~222 Go)"
        echo "  all                Tous les modeles (bf16)"
        ;;
esac
