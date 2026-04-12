#!/usr/bin/env python3
"""
Merge LoRA adapter back into the base model weights.

Uses mlx_lm.fuse under the hood.

Usage:
    python scripts/merge_lora.py \
        --model models/Mistral-Large-Instruct-2411 \
        --adapter output/mistral-large-opus/final-lora \
        --output models/Mistral-Large-Opus-Reasoning
"""

import argparse
import subprocess
import sys
from pathlib import Path


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

    print("\nFusing LoRA weights into base model via mlx_lm.fuse...")
    result = subprocess.run(
        [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", args.model,
            "--adapter-path", args.adapter,
            "--save-path", str(output_path),
            "--dequantize",
        ],
        check=True,
    )

    print(f"\nMerged model saved to: {output_path}")
    print("Next: python scripts/convert_gguf.py or ./export.sh")


if __name__ == "__main__":
    main()
