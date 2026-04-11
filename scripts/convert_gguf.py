#!/usr/bin/env python3
"""
Convert merged model to GGUF and quantize.

Requires llama.cpp's convert_hf_to_gguf.py.
Either clone llama.cpp locally or specify its path.

Usage:
    python scripts/convert_gguf.py \
        --model models/Mistral-Large-Opus-Reasoning \
        --output models/gguf \
        --quants Q6_K,Q8_0
"""

import argparse
import subprocess
import os
from pathlib import Path


def find_llama_cpp() -> Path | None:
    """Find llama.cpp installation."""
    candidates = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
    ]
    for p in candidates:
        if (p / "convert_hf_to_gguf.py").exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert to GGUF + quantize")
    parser.add_argument("--model", required=True, help="Merged model path (safetensors)")
    parser.add_argument("--output", default="models/gguf", help="Output directory")
    parser.add_argument("--quants", default="Q6_K,Q8_0", help="Comma-separated quant types")
    parser.add_argument("--llama-cpp", default=None, help="Path to llama.cpp directory")
    args = parser.parse_args()

    # Find llama.cpp
    llama_dir = Path(args.llama_cpp) if args.llama_cpp else find_llama_cpp()
    if not llama_dir:
        print("ERROR: llama.cpp not found. Clone it first:")
        print("  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git")
        print("  cd llama.cpp && cmake -B build && cmake --build build -- llama-quantize")
        return 1

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"

    if not quantize_bin.exists():
        print(f"Building llama-quantize...")
        subprocess.run(
            ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(llama_dir), check=True,
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "--", "llama-quantize"],
            cwd=str(llama_dir), check=True,
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model).name
    f16_path = output_dir / f"{model_name}-F16.gguf"

    # Step 1: Convert to GGUF F16
    if not f16_path.exists():
        print(f"\n[1] Converting {args.model} → GGUF F16...")
        subprocess.run(
            [
                "python3", str(convert_script),
                args.model,
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ],
            check=True,
        )
        print(f"  → {f16_path} ({f16_path.stat().st_size / 1024**3:.1f} GB)")
    else:
        print(f"[1] F16 already exists: {f16_path}")

    # Step 2: Quantize
    quants = [q.strip() for q in args.quants.split(",")]
    for quant in quants:
        quant_path = output_dir / f"{model_name}-{quant}.gguf"
        if quant_path.exists():
            print(f"[Q] {quant} already exists: {quant_path}")
            continue

        print(f"\n[Q] Quantizing → {quant}...")
        subprocess.run(
            [str(quantize_bin), str(f16_path), str(quant_path), quant],
            check=True,
        )
        print(f"  → {quant_path} ({quant_path.stat().st_size / 1024**3:.1f} GB)")

    # Optional: remove F16 to save space
    print(f"\nDone! GGUF files in {output_dir}/")
    print(f"F16 intermediate ({f16_path.stat().st_size / 1024**3:.1f} GB) can be deleted with:")
    print(f"  rm {f16_path}")


if __name__ == "__main__":
    main()
