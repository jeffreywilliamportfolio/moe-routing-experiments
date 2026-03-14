#!/usr/bin/env python3
"""
Ling-1T (BailingMoeV2) -- 5-Condition Self-Referential Experiment (Prefill-Only).

CAPTURE ONLY -- no entropy computation on instance.
150 prompts run in batches of 15. Raw .npy files preserved for local analysis.

Binary: capture_activations_ling1t (built from llama.cpp master, BailingMoeV2 merged)
"""
import os
import pathlib
import subprocess
import sys

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/Ling-1T-Q3_K_S/Q3_K_S/Ling-1T-Q3_K_S-00001-of-00009.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations_ling1t",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-bailingmoe2/build-cuda/bin",
)
TSV = "prompts_selfref_5cond.tsv"
OUTPUT_DIR = "output"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))

CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}
BATCH_SIZE = 15


def run_capture(tsv_file=TSV):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", tsv_file,
        "-o", OUTPUT_DIR,
        "-n", str(N_PREDICT),
        "-ngl", str(NGL),
        "-c", str(CTX),
        "-t", str(THREADS),
        "--routing-only",
        "--no-stream",
    ]
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env, check=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Ling-1T (BailingMoeV2) -- 5-Condition Self-Referential Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"150 prompts: 30 pairs x 5 conditions ({', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)})")
    print("Cal-Manip-Cal sandwich, cold KV cache")
    print("Gating: sigmoid (NOT softmax)")
    print("CAPTURE ONLY -- .npy files preserved, analysis runs locally or on-instance")
    print()

    with open(TSV) as f:
        all_lines = f.readlines()
    n_prompts = len(all_lines)
    n_batches = (n_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Loaded {n_prompts} prompts, {n_batches} batches of {BATCH_SIZE}")
    print()

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_prompts)
        batch_lines = all_lines[start:end]

        batch_tsv = f"batch_{batch_idx}.tsv"
        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"=== BATCH {batch_idx+1}/{n_batches}: prompts {start+1}-{end} ===")
        sys.stdout.flush()
        run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)

    output_path = pathlib.Path(OUTPUT_DIR)
    captured = sorted([
        d for d in output_path.iterdir()
        if d.is_dir() and (d / "metadata.txt").exists()
    ])
    print(f"\nRun complete. {len(captured)}/{n_prompts} prompts captured.")
    print()
    print("Run analysis on-instance:")
    print("  python3 analyze_local.py --output-dir output/ --prompt-suite prompt_suite.json")


if __name__ == "__main__":
    main()
