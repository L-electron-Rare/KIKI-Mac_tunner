#!/bin/bash
# Generer des donnees de raisonnement synthetiques avec des modeles teachers
# Utilisation : ./generate.sh <config> [--num-problems N] [--resume]

set -e

CONFIG="${1:?Utilisation : ./generate.sh <config> [--num-problems N] [--resume]}"
shift

source .venv/bin/activate

# Extraire le chemin du modele et le nom de sortie depuis le config
MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model'])")
OUTPUT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('output_name', 'generated'))")

echo "=== Generation de donnees avec $MODEL ==="
echo "Config : $CONFIG"
echo "Sortie : data/$OUTPUT/"

python scripts/generate_data.py \
    --model "$MODEL" \
    --problems data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl \
    --output "$OUTPUT" \
    "$@"
