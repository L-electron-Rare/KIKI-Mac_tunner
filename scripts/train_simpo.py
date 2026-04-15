#!/usr/bin/env python3
"""Phase 2 : SimPO alignment sans reference model.

Genere des paires preferred/rejected puis aligne via SimPO.
Utilise mlx-tune SimPO loss native.
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer
from datasets import load_dataset
import json
from pathlib import Path

# NOTE: SimPO requires preference pairs dataset
# Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
#
# To generate pairs:
# 1. Run inference on prompts with the SFT model (temperature=0.7)
# 2. Score responses with Opus or a reward model
# 3. Create chosen/rejected pairs
#
# This script assumes the pairs already exist at data/simpo-pairs/

def generate_preference_pairs():
    """Generate preference pairs from existing SFT model."""
    print("TODO: Generate preference pairs")
    print("1. Load SFT model with adapters")
    print("2. Generate K=4 responses per prompt")
    print("3. Score with Opus API or heuristic")
    print("4. Save best as chosen, worst as rejected")
    print("5. Output to data/simpo-pairs/train.jsonl")

def train_simpo():
    """Train SimPO alignment."""
    # Check if mlx-tune supports SimPO
    try:
        from mlx_tune import SimPOConfig, SimPOTrainer
        print("SimPO support: OK")
    except ImportError:
        print("SimPO not in mlx-tune, falling back to DPO")
        from mlx_tune import DPOConfig as SimPOConfig, DPOTrainer as SimPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/Qwen3.5-122B-A10B-BF16",
        max_seq_length=1280,
        load_in_4bit=False,
    )

    # Load adapters from Phase 1
    adapter_path = "output/qwen35-122b-opus-v3-curriculum/adapters"
    if Path(adapter_path).exists():
        print(f"Loading Phase 1 adapters from {adapter_path}")
        model.load_weights(str(Path(adapter_path) / "adapters.safetensors"))

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.01,
        target_modules=[
            "in_proj_qkv", "in_proj_z", "in_proj_a",
            "in_proj_b", "out_proj",
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )

    # Dataset
    dataset = load_dataset("json", data_files={
        "train": "data/simpo-pairs/train.jsonl",
    })

    config = SimPOConfig(
        output_dir="output/qwen35-122b-opus-v3-simpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=1,
        max_seq_length=1280,
        logging_steps=5,
        save_steps=50,
    )

    trainer = SimPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=config,
    )

    trainer.train()
    model.save_pretrained("output/qwen35-122b-opus-v3-simpo/final")

if __name__ == "__main__":
    import sys
    if "--generate-pairs" in sys.argv:
        generate_preference_pairs()
    else:
        train_simpo()
