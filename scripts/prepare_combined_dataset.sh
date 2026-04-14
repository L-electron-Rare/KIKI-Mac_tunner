#!/bin/bash
# Prepare le dataset combine pour le fine-tuning final
# Telecharge, convertit au format chat JSONL, et fusionne tous les datasets
# Utilisation : ./scripts/prepare_combined_dataset.sh

set -e
source .venv/bin/activate

echo "=== Etape 1 : Telechargement des datasets ==="
./scripts/download_datasets.sh all

echo ""
echo "=== Etape 2 : Conversion au format chat JSONL ==="
python scripts/convert_datasets.py

echo ""
echo "=== Etape 3 : Fusion et deduplication ==="
python scripts/merge_datasets.py \
    --sources \
        Opus-4.6-Reasoning-3000x-filtered \
        claude-opus-4.6-10000x-chat \
        Opus-4.6-reasoning-sft-12k-chat \
    --output combined-opus-20k \
    --deduplicate \
    --val-split 0.05

echo ""
echo "=== Termine ==="
echo "Dataset combine pret dans data/combined-opus-20k/"
echo "Mettre a jour le champ 'data:' dans les configs pour pointer vers data/combined-opus-20k"
