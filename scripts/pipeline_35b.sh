#!/bin/bash
# Pipeline complet : merge donnees → fine-tune 35B-Opus → export GGUF Q4
# Utilisation : ./scripts/pipeline_35b.sh [merge|train|fuse|gguf|all|status]
# Prereq : distillations terminees dans data/distilled-*

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"
source .venv/bin/activate

COMBINED_DATA="data/final-opus-distilled"
STUDENT_MODEL="models/Qwen3.5-35B-A3B-Opus-bf16"
STUDENT_CONFIG="configs/mlx-lm-qwen35-35b-opus-final.yaml"
ADAPTER_DIR="output/qwen35-35b-opus-final"
FUSED_DIR="output/qwen35-35b-opus-final-fused"
GGUF_DIR="output/gguf"

merge_data() {
    echo "=== Etape 1 : Fusion des datasets ==="
    echo ""

    # Lister les sources disponibles
    SOURCES=()

    # Dataset original
    if [ -d "data/Opus-4.6-Reasoning-3000x-filtered" ]; then
        SOURCES+=("Opus-4.6-Reasoning-3000x-filtered")
        echo "  + Opus 3K original"
    fi

    # Dataset 12K converti
    if [ -d "data/Opus-4.6-reasoning-sft-12k-chat" ]; then
        SOURCES+=("Opus-4.6-reasoning-sft-12k-chat")
        echo "  + Opus 12K SFT"
    fi

    # Distillation 123B
    if [ -f "data/distilled-mistral-large-123b/all.jsonl" ]; then
        # Convertir all.jsonl en train.jsonl si besoin
        if [ ! -f "data/distilled-mistral-large-123b/train.jsonl" ]; then
            cp "data/distilled-mistral-large-123b/all.jsonl" "data/distilled-mistral-large-123b/train.jsonl"
        fi
        SOURCES+=("distilled-mistral-large-123b")
        COUNT=$(wc -l < "data/distilled-mistral-large-123b/all.jsonl")
        echo "  + Distille 123B ($COUNT exemples)"
    fi

    # Distillation 35B batch 1
    if [ -f "data/distilled-qwen35-35b-opus/all.jsonl" ]; then
        if [ ! -f "data/distilled-qwen35-35b-opus/train.jsonl" ]; then
            cp "data/distilled-qwen35-35b-opus/all.jsonl" "data/distilled-qwen35-35b-opus/train.jsonl"
        fi
        SOURCES+=("distilled-qwen35-35b-opus")
        COUNT=$(wc -l < "data/distilled-qwen35-35b-opus/all.jsonl")
        echo "  + Distille 35B batch 1 ($COUNT exemples)"
    fi

    # Distillation 35B batch 2
    if [ -f "data/distilled-qwen35-35b-opus-batch2/all.jsonl" ]; then
        if [ ! -f "data/distilled-qwen35-35b-opus-batch2/train.jsonl" ]; then
            cp "data/distilled-qwen35-35b-opus-batch2/all.jsonl" "data/distilled-qwen35-35b-opus-batch2/train.jsonl"
        fi
        SOURCES+=("distilled-qwen35-35b-opus-batch2")
        COUNT=$(wc -l < "data/distilled-qwen35-35b-opus-batch2/all.jsonl")
        echo "  + Distille 35B batch 2 ($COUNT exemples)"
    fi

    echo ""

    if [ ${#SOURCES[@]} -eq 0 ]; then
        echo "ERREUR : Aucune source de donnees trouvee dans data/"
        echo "Lancer d'abord les distillations ou telecharger les datasets."
        exit 1
    fi

    echo "Fusion de ${#SOURCES[@]} sources..."

    python scripts/merge_datasets.py \
        --sources "${SOURCES[@]}" \
        --output final-opus-distilled \
        --deduplicate \
        --val-split 0.05

    echo ""
    TOTAL=$(wc -l < "$COMBINED_DATA/train.jsonl")
    VAL=$(wc -l < "$COMBINED_DATA/valid.jsonl")
    echo "Dataset final : $TOTAL train + $VAL valid"
}

train_student() {
    echo ""
    echo "=== Etape 2 : Fine-tune Qwen3.5-35B-A3B-Opus ==="
    echo "Config : $STUDENT_CONFIG"
    echo "Dataset : $COMBINED_DATA"
    echo ""

    if [ ! -d "$STUDENT_MODEL" ]; then
        echo "ERREUR : Modele base introuvable : $STUDENT_MODEL"
        echo "Lancer : ./scripts/convert_to_mlx.sh qwen35-35b-opus"
        exit 1
    fi

    if [ ! -d "$COMBINED_DATA" ]; then
        echo "ERREUR : Dataset introuvable : $COMBINED_DATA"
        echo "Lancer d'abord : $0 merge"
        exit 1
    fi

    python -m mlx_lm lora --config "$STUDENT_CONFIG"
}

fuse_student() {
    echo ""
    echo "=== Etape 3 : Fusion LoRA ==="

    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "ERREUR : Adapter introuvable : $ADAPTER_DIR"
        echo "Lancer d'abord : $0 train"
        exit 1
    fi

    python -m mlx_lm fuse \
        --model "$STUDENT_MODEL" \
        --adapter-path "$ADAPTER_DIR" \
        --save-path "$FUSED_DIR"

    echo "Modele fuse : $FUSED_DIR"
    du -sh "$FUSED_DIR"
}

export_gguf() {
    echo ""
    echo "=== Etape 4 : Export GGUF Q4_K_M ==="

    mkdir -p "$GGUF_DIR"

    if [ ! -d "$FUSED_DIR" ]; then
        echo "ERREUR : Modele fuse introuvable : $FUSED_DIR"
        echo "Lancer d'abord : $0 fuse"
        exit 1
    fi

    # Chercher le script de conversion llama.cpp
    CONVERT_SCRIPT=""
    for candidate in \
        /tmp/llama-cpp-gpu/convert_hf_to_gguf.py \
        /tmp/llama-cpp-cpu/convert_hf_to_gguf.py \
        /opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py; do
        if [ -f "$candidate" ]; then
            CONVERT_SCRIPT="$candidate"
            break
        fi
    done

    if [ -z "$CONVERT_SCRIPT" ]; then
        echo "convert_hf_to_gguf.py introuvable — clonage de llama.cpp..."
        LLAMA_DIR="/tmp/llama-cpp-gguf"
        if [ ! -d "$LLAMA_DIR" ]; then
            git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
        fi
        # Installer les dependances de conversion
        pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt" 2>/dev/null || true
        CONVERT_SCRIPT="$LLAMA_DIR/convert_hf_to_gguf.py"

        # Compiler llama-quantize si absent
        if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
            echo "Compilation de llama-quantize..."
            cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
                -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
            cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(sysctl -n hw.ncpu)"
        fi
    fi

    QUANTIZE=""
    for candidate in \
        /tmp/llama-cpp-gpu/build/bin/llama-quantize \
        /tmp/llama-cpp-cpu/build/bin/llama-quantize \
        /tmp/llama-cpp-gguf/build/bin/llama-quantize \
        "$(which llama-quantize 2>/dev/null)"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            QUANTIZE="$candidate"
            break
        fi
    done

    if [ -z "$QUANTIZE" ]; then
        echo "ERREUR : llama-quantize introuvable"
        echo "Compiler llama.cpp ou installer via homebrew"
        exit 1
    fi

    echo "Convert : $CONVERT_SCRIPT"
    echo "Quantize : $QUANTIZE"
    echo ""

    # Etape 4a : safetensors → GGUF F16
    F16_GGUF="$GGUF_DIR/qwen35-35b-opus-final-f16.gguf"
    echo "Conversion safetensors → GGUF F16..."
    python "$CONVERT_SCRIPT" "$FUSED_DIR" --outfile "$F16_GGUF" --outtype f16

    # Etape 4b : F16 → Q4_K_M
    Q4_GGUF="$GGUF_DIR/qwen35-35b-opus-final-Q4_K_M.gguf"
    echo "Quantification F16 → Q4_K_M..."
    "$QUANTIZE" "$F16_GGUF" "$Q4_GGUF" Q4_K_M

    # Etape 4c : aussi Q8_0 pour comparaison qualite
    Q8_GGUF="$GGUF_DIR/qwen35-35b-opus-final-Q8_0.gguf"
    echo "Quantification F16 → Q8_0..."
    "$QUANTIZE" "$F16_GGUF" "$Q8_GGUF" Q8_0

    # Nettoyage F16 intermediaire
    echo ""
    echo "Nettoyage F16 intermediaire..."
    rm -f "$F16_GGUF"

    echo ""
    echo "=== Modeles finaux ==="
    ls -lh "$GGUF_DIR"/*.gguf
}

summary() {
    echo ""
    echo "=========================================="
    echo "  Pipeline termine !"
    echo "=========================================="
    echo ""
    echo "Modeles produits :"
    echo "  Q4_K_M : $GGUF_DIR/qwen35-35b-opus-final-Q4_K_M.gguf (~20 Go)"
    echo "  Q8_0   : $GGUF_DIR/qwen35-35b-opus-final-Q8_0.gguf (~35 Go)"
    echo ""
    echo "Deploiement :"
    echo "  Ollama : cd $GGUF_DIR && ollama create kiki-opus -f Modelfile"
    echo "  llama-server : llama-server -m $GGUF_DIR/qwen35-35b-opus-final-Q4_K_M.gguf -ngl 99"
    echo "  LM Studio : importer le .gguf"
    echo ""
    echo "Le Q4_K_M (~20 Go) tient dans une RTX 3090/4090 24 Go."
}

case "${1:-all}" in
    merge)   merge_data ;;
    train)   train_student ;;
    fuse)    fuse_student ;;
    gguf)    export_gguf ;;
    all)
        merge_data
        train_student
        fuse_student
        export_gguf
        summary
        ;;
    status)
        echo "=== Donnees disponibles ==="
        for d in Opus-4.6-Reasoning-3000x-filtered Opus-4.6-reasoning-sft-12k-chat \
                 distilled-mistral-large-123b distilled-qwen35-35b-opus distilled-qwen35-35b-opus-batch2; do
            if [ -f "data/$d/train.jsonl" ]; then
                COUNT=$(wc -l < "data/$d/train.jsonl")
                echo "  $d : $COUNT exemples"
            elif [ -f "data/$d/all.jsonl" ]; then
                COUNT=$(wc -l < "data/$d/all.jsonl")
                echo "  $d : $COUNT exemples"
            else
                echo "  $d : (absent)"
            fi
        done
        echo ""
        echo "=== Dataset fusionne ==="
        if [ -f "$COMBINED_DATA/train.jsonl" ]; then
            TOTAL=$(wc -l < "$COMBINED_DATA/train.jsonl")
            VAL=$(wc -l < "$COMBINED_DATA/valid.jsonl" 2>/dev/null || echo "0")
            echo "  $COMBINED_DATA : $TOTAL train + $VAL valid"
        else
            echo "  (pas encore fusionne — lancer: $0 merge)"
        fi
        echo ""
        echo "=== Modeles GGUF ==="
        if ls "$GGUF_DIR"/*.gguf 1>/dev/null 2>&1; then
            ls -lh "$GGUF_DIR"/*.gguf
        else
            echo "  Pas encore de GGUF"
        fi
        ;;
    *)
        echo "Utilisation : $0 [merge|train|fuse|gguf|all|status]"
        echo ""
        echo "Etapes :"
        echo "  merge    Fusionne tous les datasets disponibles"
        echo "  train    Fine-tune le 35B-Opus sur le dataset fusionne"
        echo "  fuse     Fusionne LoRA dans le modele"
        echo "  gguf     Exporte en GGUF Q4_K_M + Q8_0"
        echo "  all      Pipeline complet (defaut)"
        echo "  status   Affiche les donnees et modeles disponibles"
        ;;
esac
