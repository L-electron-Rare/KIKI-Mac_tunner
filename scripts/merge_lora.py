#!/usr/bin/env python3
"""
Merge LoRA adapter back into the base model weights.

Usage:
    python scripts/merge_lora.py \
        --model models/Mistral-Large-Instruct-2411 \
        --adapter output/mistral-large-opus/final-lora \
        --output models/Mistral-Large-Opus-Reasoning
"""

import argparse
from pathlib import Path
from mlx_lm import load
from mlx_lm.tuner.lora import dequantize_and_merge


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output merged model path")
    args = parser.parse_args()

    print(f"Base model: {args.model}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading model + adapter...")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    print("Merging LoRA weights into base model...")
    dequantize_and_merge(
        model=args.model,
        adapter_path=args.adapter,
        output_path=str(output_path),
    )

    # Copy tokenizer files
    import shutil
    model_path = Path(args.model)
    for f in model_path.glob("tokenizer*"):
        shutil.copy2(f, output_path)
    for f in model_path.glob("special_tokens*"):
        shutil.copy2(f, output_path)
    if (model_path / "config.json").exists():
        shutil.copy2(model_path / "config.json", output_path)

    print(f"\nMerged model saved to: {output_path}")
    print("Next: python scripts/convert_gguf.py or ./export.sh")


if __name__ == "__main__":
    main()
