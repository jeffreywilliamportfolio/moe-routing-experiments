#!/usr/bin/env python3
"""
Qwen3.5-397B-A17B -- 5-Condition Self-Referential Experiment (Prefill-Only).

150 prompts: 30 pairs x 5 conditions.
  A = "this system"   (proximal deictic)
  B = "a system"      (indefinite generic)
  C = "your system"   (2nd-person possessive)
  D = "the system"    (definite article)
  E = "their system"  (3rd-person possessive)

Cal-Manip-Cal sandwich structure. Cold KV cache.

Discriminates:
  - Definiteness: does D ("the") pattern like A ("this") or B ("a")?
  - Addressivity vs possessive: does E ("their") pattern like C ("your") or B ("a")?

Architecture: 512 experts, top-10 routing, 60 MoE layers.
Entropy normalized by log2(512).
"""
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
from itertools import combinations

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/Qwen3.5-397B-A17B-GGUF/UD-IQ3_XXS/Qwen3.5-397B-A17B-UD-IQ3_XXS-00001-of-00004.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-b8123/build-cuda/bin",
)
TSV = "prompts_selfref_5cond.tsv"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_selfref_5cond_prefill_qwen.json"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "16384"))
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


def softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


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
        "-fa", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
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
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))

    if excluded_layers:
        print(f"    Auto-excluded layers: {excluded_layers}")

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

    print("=== Qwen3.5-397B-A17B -- 5-Condition Self-Referential Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"150 prompts: 30 pairs x 5 conditions ({', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)})")
    print("Cal-Manip-Cal sandwich, cold KV cache, Qwen ChatML template")
    print()

    with open(TSV) as f:
        all_lines = f.readlines()
    n_batches = (len(all_lines) + BATCH_SIZE - 1) // BATCH_SIZE

    results = []
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_lines))
        batch_lines = all_lines[start:end]

        batch_tsv = f"batch_{batch_idx}.tsv"
        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"\n=== BATCH {batch_idx+1}/{n_batches}: Capture prompts {start+1}-{end} ===")
        sys.stdout.flush()
        run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)

        print(f"  Computing metrics for batch {batch_idx+1}...")
        prompt_dirs = sorted(
            [d for d in pathlib.Path(OUTPUT_DIR).iterdir() if d.is_dir() and (d / "metadata.txt").exists()
             and not any(r["id"] == d.name for r in results)],
            key=lambda d: d.name,
        )

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

            router_dir = prompt_dir / "router"
            if router_dir.exists():
                shutil.rmtree(router_dir)

    # === Analysis ===
    print("\n=== PHASE 3: 5-Condition Paired Analysis ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    # Token match table
    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c+'_tok':>5}" for c in CONDITIONS) + f" {'Match':>7}")
    print("-" * 80)
    for pair_num in sorted(pairs.keys()):
        p = pairs[pair_num]
        if not all(c in p for c in CONDITIONS):
            continue
        toks = [p[c]["n_prompt_tokens"] for c in CONDITIONS]
        status = "OK" if len(set(toks)) == 1 else "MISMATCH"
        cat = p["A"]["category"]
        tok_str = " ".join(f"{t:>5}" for t in toks)
        print(f"  {pair_num:>3}  {cat:<20} {tok_str} {status:>7}")

    # All 10 pairwise comparisons
    comp_results = {}
    for cond1, cond2 in combinations(CONDITIONS, 2):
        label = f"{cond1} vs {cond2}"
        diffs_re = []
        diffs_lt = []

        print(f"\n--- {label} ({COND_LABELS[cond1]} vs {COND_LABELS[cond2]}) ---")
        print(f"  {'Pair':>4} {'Category':<20} {f'{cond1}_RE':>8} {f'{cond2}_RE':>8} {'Diff_RE':>8} "
              f"{f'{cond1}_LT':>8} {f'{cond2}_LT':>8} {'Diff_LT':>8}")
        print("  " + "-" * 90)

        for pair_num in sorted(pairs.keys()):
            if cond1 not in pairs[pair_num] or cond2 not in pairs[pair_num]:
                continue
            r1 = pairs[pair_num][cond1]
            r2 = pairs[pair_num][cond2]
            d_re = r1["prefill_re"] - r2["prefill_re"]
            d_lt = r1["last_token_re"] - r2["last_token_re"]
            diffs_re.append(d_re)
            diffs_lt.append(d_lt)
            print(f"  {pair_num:>4}  {r1['category']:<20} {r1['prefill_re']:>8.6f} {r2['prefill_re']:>8.6f} {d_re:>+8.6f} "
                  f"{r1['last_token_re']:>8.6f} {r2['last_token_re']:>8.6f} {d_lt:>+8.6f}")

        if diffs_lt:
            dre = np.array(diffs_re)
            dlt = np.array(diffs_lt)
            n_pos_re = int(np.sum(dre > 0))
            n_pos_lt = int(np.sum(dlt > 0))
            print(f"\n  Summary (n={len(dlt)} pairs):")
            print(f"    All-token RE:  mean = {np.mean(dre):+.6f} +/- {np.std(dre):.6f}  ({n_pos_re}/{len(dre)} {cond1}>{cond2})")
            print(f"    Last-token RE: mean = {np.mean(dlt):+.6f} +/- {np.std(dlt):.6f}  ({n_pos_lt}/{len(dlt)} {cond1}>{cond2})")
            if len(dlt) >= 6 and scipy_wilcoxon is not None:
                w_re, p_re = scipy_wilcoxon(dre)
                w_lt, p_lt = scipy_wilcoxon(dlt)
                print(f"    Wilcoxon all-tok:  W={w_re:.0f}, p={p_re:.4e}")
                print(f"    Wilcoxon last-tok: W={w_lt:.0f}, p={p_lt:.4e}")
            elif len(dlt) >= 6:
                print("    Wilcoxon skipped: scipy not installed on this host")

            comp_results[label] = {
                "mean_diff_re": float(np.mean(dre)),
                "std_diff_re": float(np.std(dre)),
                "mean_diff_lt": float(np.mean(dlt)),
                "std_diff_lt": float(np.std(dlt)),
                "n_positive_re": n_pos_re,
                "n_positive_lt": n_pos_lt,
                "n_pairs": len(dlt),
            }

    # Condition means summary
    print("\n--- Condition Means ---")
    print(f"  {'Cond':<6} {'Label':<20} {'All-tok RE':>12} {'Last-tok RE':>12} {'N':>4}")
    print("  " + "-" * 60)
    for c in CONDITIONS:
        cond_rows = [r for r in results if r["condition"] == c]
        if cond_rows:
            mean_re = np.mean([r["prefill_re"] for r in cond_rows])
            mean_lt = np.mean([r["last_token_re"] for r in cond_rows])
            print(f"  {c:<6} {COND_LABELS[c]:<20} {mean_re:>12.6f} {mean_lt:>12.6f} {len(cond_rows):>4}")

    # Per-category breakdown (last-token RE, A vs B)
    print("\n--- Per-Category (last-token RE, A vs B) ---")
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
        "experiment": "qwen_selfref_5cond_1",
        "model": "Qwen3.5-397B-A17B-UD-IQ3_XXS",
        "architecture": "qwen35moe",
        "n_experts": 512,
        "n_expert_used": 10,
        "n_moe_layers": 60,
        "chat_template": "<|im_start|>user {prompt}<|im_end|> <|im_start|>assistant",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions, cold cache",
        "conditions": COND_LABELS,
        "rationale": "Discriminates definiteness (D=the) vs addressivity (E=their) vs possessive (C=your)",
        "inference": {
            "n_predict": N_PREDICT,
            "ngl": NGL,
            "ctx": CTX,
            "flash_attn": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "comparisons": comp_results,
        "npy_preserved": False,
        "per_prompt": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts. Results -> {RESULTS_FILE} ===")


if __name__ == "__main__":
    main()
