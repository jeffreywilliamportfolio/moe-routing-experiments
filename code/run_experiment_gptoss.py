#!/usr/bin/env python3
"""
GPT-OSS 120B -- Self-Referential Paired Experiment (Prefill-Only).

60 prompts: 30 A (self-referential) + 30 B (matched control).
Cal-Manip-Cal sandwich structure. Preserve router tensors for downstream
pairwise KL / overlap / disagreement analysis.

Architecture: 128 experts, top-4 routing, 36 MoE layers (layer 35 excluded).
Entropy normalized by log2(128).
"""
import glob
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-b8123/build-cuda/bin",
)
TSV = "prompts_selfref_paired.tsv"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_selfref_paired_prefill_gptoss.json"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")

# GPT-OSS-120B: 128 experts, layer 35 excluded (3-row truncation bug)
EXCLUDED_LAYERS = {35}


def softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def run_capture():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", TSV,
        "-o", OUTPUT_DIR,
        "-n", str(N_PREDICT),
        "-ngl", str(NGL),
        "-c", str(CTX),
        "-t", str(THREADS),
        "-fa", FLASH_ATTN,
        "--cache-type-k", CACHE_TYPE_K,
        "--cache-type-v", CACHE_TYPE_V,
        "--routing-only",
        "--no-stream",
    ]
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env, check=False)


def get_metadata(prompt_dir):
    meta = prompt_dir / "metadata.txt"
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
    return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))


def compute_metrics(prompt_dir, n_prompt):
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return None

    files = sorted(
        glob.glob(str(router_dir / "ffn_moe_logits-*.npy")),
        key=lambda file_path: int(pathlib.Path(file_path).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    n_experts = np.load(files[0]).shape[1]
    max_ent = np.log2(n_experts)

    shapes = {}
    for file_path in files:
        layer_index = int(pathlib.Path(file_path).stem.split("-")[1])
        shapes[layer_index] = np.load(file_path).shape[0]
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        layer_index for layer_index in shapes
        if shapes[layer_index] >= median_rows * 0.5
        and layer_index not in EXCLUDED_LAYERS
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))

    per_layer = []
    all_ent = []
    last_token_ents = []

    for layer_index in good_layers:
        file_path = router_dir / f"ffn_moe_logits-{layer_index}.npy"
        logits = np.load(str(file_path))
        n_rows = min(logits.shape[0], n_prompt)
        probs = softmax(logits[:n_rows], axis=-1)
        ent = -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / max_ent
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        per_layer.append({
            "layer": layer_index,
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "last_token_entropy": last_ent,
            "n_rows": int(logits.shape[0]),
        })

        valid = ent > 0
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

    return {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(good_layers),
        "n_layers_excluded": excluded_layers,
        "n_experts": n_experts,
        "per_layer": per_layer,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== GPT-OSS 120B -- Self-Referential Paired Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"flash_attn={FLASH_ATTN}, cache_type_k={CACHE_TYPE_K}, cache_type_v={CACHE_TYPE_V}")
    print("60 prompts: 30 self-ref (A) + 30 control (B)")
    print("Cal-Manip-Cal sandwich, cold KV cache, Harmony template")
    print()

    print("=== PHASE 1: Capture ===")
    sys.stdout.flush()
    run_capture()

    print("\n=== PHASE 2: Compute metrics ===")
    prompt_dirs = sorted(
        [d for d in pathlib.Path(OUTPUT_DIR).iterdir() if d.is_dir() and (d / "metadata.txt").exists()],
        key=lambda d: d.name,
    )

    results = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        n_prompt, _n_gen = get_metadata(prompt_dir)
        metrics = compute_metrics(prompt_dir, n_prompt)
        if metrics is None:
            print(f"  SKIP {prompt_id}: no valid data")
            continue

        prefix, *rest = prompt_id.split("_", 1)
        category = rest[0] if rest else ""
        pair_num = int(prefix[1:3])
        condition = prefix[3]

        row = {
            "id": prompt_id,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)
        print(f"  {prompt_id}: RE={metrics['prefill_re']:.6f} last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}")

    print("\n=== PHASE 3: Paired Analysis ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    print(
        f"\n{'Pair':>4} {'Category':<20} {'A_tok':>5} {'B_tok':>5} {'A_RE':>8} {'B_RE':>8} {'A-B_RE':>8} "
        f"{'A_LT':>8} {'B_LT':>8} {'A-B_LT':>8}"
    )
    print("-" * 105)

    diffs_re = []
    diffs_lt = []
    for pair_num in sorted(pairs.keys()):
        if "A" not in pairs[pair_num] or "B" not in pairs[pair_num]:
            continue
        row_a = pairs[pair_num]["A"]
        row_b = pairs[pair_num]["B"]
        diff_re = row_a["prefill_re"] - row_b["prefill_re"]
        diff_lt = row_a["last_token_re"] - row_b["last_token_re"]
        diffs_re.append(diff_re)
        diffs_lt.append(diff_lt)
        token_status = "OK" if row_a["n_prompt_tokens"] == row_b["n_prompt_tokens"] else "MISMATCH"
        print(
            f"  {pair_num:>3}  {row_a['category']:<20} {row_a['n_prompt_tokens']:>5} {row_b['n_prompt_tokens']:>5} "
            f"{row_a['prefill_re']:>8.6f} {row_b['prefill_re']:>8.6f} {diff_re:>+8.6f} "
            f"{row_a['last_token_re']:>8.6f} {row_b['last_token_re']:>8.6f} {diff_lt:>+8.6f} {token_status}"
        )

    if diffs_lt:
        diffs_re_arr = np.array(diffs_re)
        diffs_lt_arr = np.array(diffs_lt)
        print(f"\n--- Paired Summary (n={len(diffs_lt)} pairs) ---")
        print(f"  All-token RE:  A-B mean = {np.mean(diffs_re_arr):+.6f} +/- {np.std(diffs_re_arr):.6f}")
        print(f"  Last-token RE: A-B mean = {np.mean(diffs_lt_arr):+.6f} +/- {np.std(diffs_lt_arr):.6f}")
        if len(diffs_lt) >= 6 and scipy_wilcoxon is not None:
            w_re, p_re = scipy_wilcoxon(diffs_re_arr)
            w_lt, p_lt = scipy_wilcoxon(diffs_lt_arr)
            print(f"  Wilcoxon all-tok:  W={w_re:.0f}, p={p_re:.4e}")
            print(f"  Wilcoxon last-tok: W={w_lt:.0f}, p={p_lt:.4e}")
        elif len(diffs_lt) >= 6:
            print("  Wilcoxon skipped: scipy not installed on this host")

        print("\n--- Per-Category (last-token RE) ---")
        categories = sorted(set(row["category"] for row in results))
        for category in categories:
            cat_diffs = []
            for pair_num in sorted(pairs.keys()):
                if "A" not in pairs[pair_num] or "B" not in pairs[pair_num]:
                    continue
                if pairs[pair_num]["A"]["category"] != category:
                    continue
                cat_diffs.append(pairs[pair_num]["A"]["last_token_re"] - pairs[pair_num]["B"]["last_token_re"])
            if cat_diffs:
                arr = np.array(cat_diffs)
                print(f"  {category:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    output = {
        "experiment": "gptoss_selfref_paired_1",
        "model": "GPT-OSS-120B mxfp4",
        "architecture": "gpt-oss",
        "n_experts": 128,
        "n_expert_used": 4,
        "n_moe_layers": 36,
        "n_moe_layers_valid": 35,
        "excluded_layers": sorted(EXCLUDED_LAYERS),
        "chat_template": "<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts, cold cache",
        "inference": {
            "n_predict": N_PREDICT,
            "ngl": NGL,
            "ctx": CTX,
            "flash_attn": FLASH_ATTN,
            "cache_type_k": CACHE_TYPE_K,
            "cache_type_v": CACHE_TYPE_V,
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "npy_preserved": True,
        "per_prompt": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts. Results -> {RESULTS_FILE} ===")


if __name__ == "__main__":
    main()
