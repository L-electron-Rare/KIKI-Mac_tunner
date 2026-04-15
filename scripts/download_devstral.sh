#!/bin/bash
# Télécharger Devstral 2 123B et les datasets de coding pour le pipeline Sonnet
# Utilisation : ./scripts/download_devstral.sh [model|datasets|all]
#
# Datasets sélectionnés (recherche avril 2026) :
#   1. nvidia/OpenCodeReasoning    — 735K exemples Python, traces de raisonnement DeepSeek-R1
#   2. nvidia/OpenCodeInstruct     — 5M exemples, le plus grand dataset code open-source
#   3. open-r1/codeforces-cots     — 10K problèmes Codeforces, chaînes de raisonnement R1
#   4. ise-uiuc/Magicoder-OSS-Instruct-75K — 75K, instructions OSS diversifiées
#   5. m-a-p/CodeFeedback-Filtered-Instruction — 156K instructions code filtrées qualité 4-5/5
#   6. SWE-Gym/OpenHands-Sampled-Trajectories — 491 trajectoires SWE agentic réussies
#   7. nvidia/Nemotron-SWE-v1      — 59K trajectoires agentic OpenHands
#
# On télécharge un sous-ensemble ciblé (~15-20K exemples après filtrage)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

MODELS_DIR="models"
DATA_DIR="data/sonnet-coding-raw"
mkdir -p "$MODELS_DIR" "$DATA_DIR"

download_model() {
    echo "=== Devstral 2 123B Instruct (mistralai/Devstral-2-123B-Instruct-2512) ==="
    echo "Taille estimée : ~250 Go (BF16 dense)"
    echo ""

    # Vérifier si déjà téléchargé
    if [ -d "$MODELS_DIR/Devstral-2-123B-Instruct" ] && [ "$(ls -A $MODELS_DIR/Devstral-2-123B-Instruct/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Modèle déjà présent dans $MODELS_DIR/Devstral-2-123B-Instruct"
        return 0
    fi

    hf download mistralai/Devstral-2-123B-Instruct-2512 \
        --local-dir "$MODELS_DIR/Devstral-2-123B-Instruct"

    echo "Modèle téléchargé dans $MODELS_DIR/Devstral-2-123B-Instruct"
}

download_datasets() {
    echo "=== Téléchargement des datasets de coding ==="
    echo ""

    # 1. OpenCodeReasoning — traces de raisonnement sur problèmes compétitifs (Python)
    echo "--- nvidia/OpenCodeReasoning (split_0, ~568K exemples) ---"
    hf download nvidia/OpenCodeReasoning \
        --repo-type dataset \
        --local-dir "$DATA_DIR/OpenCodeReasoning"

    # 2. OpenCodeInstruct — le plus large dataset code instruction (5M, on filtrera)
    echo "--- nvidia/OpenCodeInstruct (5M exemples, filtrage après) ---"
    hf download nvidia/OpenCodeInstruct \
        --repo-type dataset \
        --local-dir "$DATA_DIR/OpenCodeInstruct"

    # 3. Codeforces CoTs — chaînes de raisonnement R1 sur problèmes Codeforces
    echo "--- open-r1/codeforces-cots (~100K traces sur 10K problèmes) ---"
    hf download open-r1/codeforces-cots \
        --repo-type dataset \
        --local-dir "$DATA_DIR/codeforces-cots"

    # 4. Magicoder OSS-Instruct — instructions code diversifiées
    echo "--- ise-uiuc/Magicoder-OSS-Instruct-75K ---"
    hf download ise-uiuc/Magicoder-OSS-Instruct-75K \
        --repo-type dataset \
        --local-dir "$DATA_DIR/Magicoder-OSS-Instruct-75K"

    # 5. CodeFeedback — instructions filtrées haute qualité
    echo "--- m-a-p/CodeFeedback-Filtered-Instruction (156K, qualité 4-5/5) ---"
    hf download m-a-p/CodeFeedback-Filtered-Instruction \
        --repo-type dataset \
        --local-dir "$DATA_DIR/CodeFeedback-Filtered-Instruction"

    # 6. SWE-Gym trajectoires agentic
    echo "--- SWE-Gym/OpenHands-Sampled-Trajectories (491 trajectoires réussies) ---"
    hf download SWE-Gym/OpenHands-Sampled-Trajectories \
        --repo-type dataset \
        --local-dir "$DATA_DIR/OpenHands-Sampled-Trajectories"

    # 7. Nemotron SWE — trajectoires agentic plus large
    echo "--- nvidia/Nemotron-SWE-v1 (59K trajectoires) ---"
    hf download nvidia/Nemotron-SWE-v1 \
        --repo-type dataset \
        --local-dir "$DATA_DIR/Nemotron-SWE-v1"

    echo ""
    echo "=== Tous les datasets téléchargés dans $DATA_DIR ==="
    echo "Lancer ensuite : python scripts/prepare_coding_dataset.py"
}

case "${1:-help}" in
    model)
        download_model
        ;;
    datasets)
        download_datasets
        ;;
    all)
        download_model
        download_datasets
        ;;
    *)
        echo "Utilisation : $0 <model|datasets|all>"
        echo ""
        echo "Commandes :"
        echo "  model      Télécharger Devstral 2 123B (~250 Go)"
        echo "  datasets   Télécharger les datasets de coding"
        echo "  all        Modèle + datasets"
        echo ""
        echo "ATTENTION : le modèle fait ~250 Go, vérifier l'espace disque."
        echo "Les datasets bruts font ~50-100 Go, on filtrera à 15-20K exemples."
        ;;
esac
