#!/usr/bin/env python3
"""Training 122B Opus-v3 via mlx-tune (MoE-aware LoRA)."""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer
from datasets import load_dataset

# 1. Charger le modele avec LoRA
print("Chargement Qwen3.5-122B-A10B-BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/Qwen3.5-122B-A10B-BF16",
    max_seq_length=1280,
    dtype=None,  # auto-detect bf16
    load_in_4bit=False,  # bf16 complet
)

# 2. Appliquer LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.01,
    target_modules=[
        "in_proj_qkv", "in_proj_z", "in_proj_a",
        "in_proj_b", "out_proj",  # DeltaNet layers
        "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention
    ],
    use_gradient_checkpointing="unsloth",
)

# 3. Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/final-opus-v3-1/train.jsonl",
        "validation": "data/final-opus-v3-1/valid.jsonl",
    },
)

def format_example(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# 4. Training config
config = SFTConfig(
    output_dir="output/qwen35-122b-opus-v3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=2,
    max_seq_length=1280,
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

# 6. Train
print("Lancement training...")
trainer.train()

# 7. Save
model.save_pretrained("output/qwen35-122b-opus-v3/final")
tokenizer.save_pretrained("output/qwen35-122b-opus-v3/final")
print("Training termine.")
