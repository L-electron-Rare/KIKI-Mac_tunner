#!/usr/bin/env python3
"""Training Devstral 2 123B — Sonnet-style fast coding via mlx-tune.

Devstral 2 est un modèle DENSE 123B (pas MoE comme Qwen3.5-122B).
Architecture Mistral — plus simple, pas de routage d'experts.
LoRA sur toutes les couches d'attention.
~250 Go BF16, tient dans 512 Go de mémoire unifiée.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer
from datasets import load_dataset

# 1. Charger le modèle avec LoRA
print("Chargement Devstral-2-123B-Instruct (dense, ~250 Go BF16)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/Devstral-2-123B-Instruct",
    max_seq_length=4096,
    dtype=None,  # auto-detect bf16
    load_in_4bit=False,  # bf16 complet — on a 512 Go
)

# 2. Appliquer LoRA — architecture Mistral dense
# Pas de DeltaNet ni MoE, juste de l'attention standard
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.01,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # MLP (FFN)
    ],
    use_gradient_checkpointing="unsloth",
)

# 3. Dataset — format chat JSONL préparé par prepare_coding_dataset.py
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/sonnet-coding/train.jsonl",
        "validation": "data/sonnet-coding/valid.jsonl",
    },
)


def format_example(example):
    """Applique le chat template Mistral aux messages."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


dataset = dataset.map(format_example)

# 4. Configuration training
# Dense 123B → plus de mémoire que le MoE 122B (10B actifs)
# Garder batch_size=1, compenser avec grad_accumulation_steps=8
config = SFTConfig(
    output_dir="output/devstral2-sonnet",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    num_train_epochs=2,
    max_seq_length=4096,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    optim="adamw_8bit",
    dataset_text_field="text",
    packing=False,
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=config,
)

# 6. Lancer le training
print(f"Dataset : {len(dataset['train'])} train, {len(dataset['validation'])} valid")
print("Lancement training Devstral 2 Sonnet-coding...")
trainer.train()

# 7. Sauvegarder
model.save_pretrained("output/devstral2-sonnet/final")
tokenizer.save_pretrained("output/devstral2-sonnet/final")
print("Training Devstral 2 Sonnet terminé.")
