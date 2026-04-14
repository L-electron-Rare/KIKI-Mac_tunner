#!/usr/bin/env bash
# convert_ane.sh — Convertit un modèle Qwen3 dense pour Apple Neural Engine via ANEMLL
#
# ANEMLL v0.3.5 supporte Qwen 3 dense (0.6B, 1.7B, 8B).
# Le Qwen3.5-35B-A3B (MoE, 256 experts, architecture qwen3_5_moe) n'est PAS supporté.
# MoE est dans la roadmap ANEMLL mais pas encore implémenté.
#
# Usage:
#   ./scripts/convert_ane.sh                        # Qwen3-8B par défaut
#   ./scripts/convert_ane.sh Qwen/Qwen3-1.7B       # Autre modèle
#   ./scripts/convert_ane.sh /chemin/vers/modele    # Chemin local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANEMLL_DIR="/tmp/anemll"
ANEMLL_ENV="${ANEMLL_DIR}/env-anemll"
OUTPUT_BASE="${PROJECT_ROOT}/output/ane"

# Modèle par défaut : Qwen3-8B (dense, supporté par ANEMLL)
MODEL="${1:-Qwen/Qwen3-8B}"

# Extraire un nom court pour le dossier de sortie
MODEL_SHORT="$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_SHORT}"

# Paramètres de conversion
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LUT1="${LUT1:-}"          # Embeddings : pas de quantization par défaut
LUT2="${LUT2:-4}"         # FFN : LUT4
LUT3="${LUT3:-6}"         # LM head : LUT6
NUM_CHUNKS="${NUM_CHUNKS:-2}"  # 2 chunks pour 8B

echo "========================================"
echo "  ANEMLL — Conversion pour ANE"
echo "========================================"
echo "Modèle:        $MODEL"
echo "Sortie:        $OUTPUT_DIR"
echo "Contexte:      $CONTEXT_LENGTH"
echo "Batch:         $BATCH_SIZE"
echo "LUT (emb/ffn/lm): ${LUT1:-none}/${LUT2}/${LUT3}"
echo "Chunks:        $NUM_CHUNKS"
echo "========================================"

# Vérifier ANEMLL
if [ ! -d "$ANEMLL_DIR" ]; then
    echo "ERREUR: ANEMLL non trouvé dans $ANEMLL_DIR"
    echo "Cloner avec: cd /tmp && git clone https://github.com/Anemll/Anemll.git anemll"
    exit 1
fi

# Vérifier l'environnement ANEMLL
if [ ! -f "${ANEMLL_ENV}/bin/activate" ]; then
    echo "ERREUR: Environnement ANEMLL non trouvé."
    echo "Créer avec: cd $ANEMLL_DIR && ./create_uv_env.sh && source env-anemll/bin/activate && ./install_dependencies.sh"
    exit 1
fi

# Activer l'environnement ANEMLL (Python 3.9 + coremltools 9.0)
echo ""
echo "[1/4] Activation de l'environnement ANEMLL..."
source "${ANEMLL_ENV}/bin/activate"
echo "  Python: $(python --version 2>&1)"
echo "  coremltools: $(python -c 'import coremltools; print(coremltools.__version__)' 2>/dev/null || echo 'non installé')"

# Vérifier que le modèle n'est pas MoE
echo ""
echo "[2/4] Vérification de l'architecture du modèle..."

check_model_arch() {
    local model_path="$1"
    local config_file=""

    # Chercher config.json localement ou dans le cache HF
    if [ -f "${model_path}/config.json" ]; then
        config_file="${model_path}/config.json"
    else
        # Le modèle sera téléchargé par ANEMLL, on ne peut pas vérifier à l'avance
        echo "  Modèle HuggingFace: $model_path (sera téléchargé si nécessaire)"
        return 0
    fi

    local model_type
    model_type=$(python3 -c "
import json
with open('${config_file}') as f:
    c = json.load(f)
mt = c.get('model_type', '')
# Vérifier aussi dans text_config pour les modèles multimodaux
if 'text_config' in c:
    mt = c.get('text_config', {}).get('model_type', mt)
print(mt)
" 2>/dev/null)

    if [[ "$model_type" == *"moe"* ]]; then
        echo "  ERREUR: Architecture MoE détectée (${model_type})"
        echo ""
        echo "  ANEMLL v0.3.5 ne supporte PAS les modèles MoE."
        echo "  Le support MoE est dans la roadmap (https://github.com/Anemll/Anemll/blob/main/Roadmap.MD)"
        echo ""
        echo "  Modèles Qwen supportés par ANEMLL :"
        echo "    - Qwen 3 : 0.6B, 1.7B, 8B (dense)"
        echo "    - Qwen 2.5 : 0.5B, 1.5B, 3B, 7B (dense)"
        echo ""
        echo "  Alternatives :"
        echo "    1. Qwen/Qwen3-8B (dense, 8B params, conversion validée)"
        echo "    2. Attendre le support MoE dans ANEMLL"
        echo "    3. Utiliser MLX pour l'inférence GPU du modèle MoE"
        return 1
    fi

    local num_experts
    num_experts=$(python3 -c "
import json
with open('${config_file}') as f:
    c = json.load(f)
tc = c.get('text_config', c)
print(tc.get('num_experts', 0))
" 2>/dev/null)

    if [ "$num_experts" -gt 1 ] 2>/dev/null; then
        echo "  ERREUR: Modèle MoE détecté (${num_experts} experts)"
        echo "  ANEMLL ne supporte pas encore les architectures MoE."
        return 1
    fi

    echo "  Architecture: ${model_type} — OK (dense)"
    return 0
}

if ! check_model_arch "$MODEL"; then
    exit 1
fi

# Créer le dossier de sortie
mkdir -p "$OUTPUT_DIR"

# Lancer la conversion
echo ""
echo "[3/4] Lancement de la conversion ANEMLL..."
echo "  Commande: ${ANEMLL_DIR}/anemll/utils/convert_model.sh"

CONVERT_ARGS=(
    --model "$MODEL"
    --output "$OUTPUT_DIR"
    --context "$CONTEXT_LENGTH"
    --batch "$BATCH_SIZE"
    --lut2 "$LUT2"
    --lut3 "$LUT3"
    --chunk "$NUM_CHUNKS"
)

# Ajouter LUT1 seulement si défini
if [ -n "$LUT1" ]; then
    CONVERT_ARGS+=(--lut1 "$LUT1")
fi

echo "  Args: ${CONVERT_ARGS[*]}"
echo ""

"${ANEMLL_DIR}/anemll/utils/convert_model.sh" "${CONVERT_ARGS[@]}"

# Vérification
echo ""
echo "[4/4] Vérification..."
if [ -f "${OUTPUT_DIR}/meta.yaml" ]; then
    echo "  Conversion réussie."
    echo "  meta.yaml: ${OUTPUT_DIR}/meta.yaml"
    echo ""
    echo "  Pour tester :"
    echo "    source ${ANEMLL_ENV}/bin/activate"
    echo "    python ${ANEMLL_DIR}/tests/chat.py --meta ${OUTPUT_DIR}/meta.yaml --prompt 'Bonjour!'"
    echo ""
    echo "  Pour inférence Swift CLI :"
    echo "    cd ${ANEMLL_DIR}/anemll-swift-cli"
    echo "    swift build -c release"
    echo "    .build/release/anemll-cli --meta ${OUTPUT_DIR}/meta.yaml"
else
    echo "  ATTENTION: meta.yaml non trouvé. Vérifier les logs ci-dessus."
    exit 1
fi
