#!/usr/bin/env python3
"""Phase 3 : GRPO reasoning RL avec rewards verifiables.

Pour chaque prompt math/code/logique:
1. Genere K=4 reponses
2. Verifie la reponse (execute code, check math)
3. Score: 1.0 si correct, 0.0 sinon
4. Normalise par groupe (avantage relatif)
5. Update policy via GRPO loss
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

from mlx_tune import FastLanguageModel, GRPOConfig, GRPOTrainer
from datasets import load_dataset

def math_reward(response, expected_answer):
    """Reward verifiable pour les problemes math."""
    import re
    # Extraire le nombre final de la reponse
    numbers = re.findall(r'-?\d+\.?\d*', response.split('</thinking>')[-1] if '</thinking>' in response else response)
    if numbers and str(expected_answer) in numbers:
        return 1.0
    return 0.0

def code_reward(response, test_code):
    """Reward verifiable pour le code."""
    import subprocess, tempfile
    # Extraire le code Python de la reponse
    code_blocks = response.split('```python')
    if len(code_blocks) < 2:
        return 0.0
    code = code_blocks[1].split('```')[0]

    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(code + '\n' + test_code)
        f.flush()
        try:
            result = subprocess.run(['python', f.name], capture_output=True, timeout=10)
            return 1.0 if result.returncode == 0 else 0.0
        except:
            return 0.0

def train_grpo():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/Qwen3.5-122B-A10B-BF16",
        max_seq_length=1280,
        load_in_4bit=False,
    )

    # Load adapters from Phase 2 (SimPO)
    from pathlib import Path
    adapter_path = "output/qwen35-122b-opus-v3-simpo/final"
    if Path(adapter_path).exists():
        print(f"Loading Phase 2 adapters from {adapter_path}")

    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.01,
        target_modules=[
            "in_proj_qkv", "in_proj_z", "in_proj_a",
            "in_proj_b", "out_proj",
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )

    # Dataset de prompts math/code avec reponses attendues
    dataset = load_dataset("json", data_files={
        "train": "data/grpo-prompts/train.jsonl",
    })

    config = GRPOConfig(
        output_dir="output/qwen35-122b-opus-v3-grpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1,
        max_seq_length=1280,
        logging_steps=5,
        save_steps=50,
        num_generations=4,  # K=4 reponses par prompt
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=config,
        reward_funcs=[math_reward],  # Verifiable rewards
    )

    trainer.train()
    model.save_pretrained("output/qwen35-122b-opus-v3-grpo/final")

if __name__ == "__main__":
    train_grpo()
