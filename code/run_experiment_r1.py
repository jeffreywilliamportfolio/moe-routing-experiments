#!/usr/bin/env python3
"""
DeepSeek R1 — Self-Referential 3-Condition Experiment (Prefill-Only).

90 prompts: 30 A (this) + 30 B (a) + 30 C (your).
Cal-Manip-Cal sandwich structure. Batch processing with .npy cleanup.

Architecture: 671B MoE, 256 experts, 8 active, 58 MoE layers (3-60).
"""
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

MODEL = "/workspace/models/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf"
BINARY = "/workspace/consciousness-experiment/capture_activations"
LLAMA_BUILD_BIN = os.environ.get("LLAMA_BUILD_BIN", "/workspace/src/llama.cpp-b8123/build-cuda/bin")
TSV = "prompts_selfref_paired_v2.tsv"
OUTPUT_DIR = "output_v2"
RESULTS_FILE = "results_selfref_3cond_prefill_r1.json"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "30"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))
BATCH_SIZE = 15


def softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


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
        key=lambda fp: int(pathlib.Path(fp).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    n_experts = np.load(files[0]).shape[1]
    max_ent = np.log2(n_experts)

    shapes = {}
    for fp in files:
        li = int(pathlib.Path(fp).stem.split("-")[1])
        shapes[li] = np.load(fp).shape[0]
    median_rows = np.median(list(shapes.values()))
    good_layers = sorted([li for li in shapes if shapes[li] >= median_rows * 0.5])

    per_layer = []
    all_ent = []
    last_token_ents = []

    for li in good_layers:
        fp = router_dir / f"ffn_moe_logits-{li}.npy"
        logits = np.load(str(fp))
        n_rows = min(logits.shape[0], n_prompt)
        probs = softmax(logits[:n_rows], axis=-1)
        ent = -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / max_ent
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        per_layer.append({
            "layer": li,
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
        "n_layers_excluded": sorted(set(shapes.keys()) - set(good_layers)),
        "n_experts": n_experts,
        "per_layer": per_layer,
    }


def run_batch(prompts, batch_idx, total_batches):
    batch_tsv = f"_batch_{batch_idx}.tsv"
    with open(batch_tsv, "w") as f:
        for pid, text in prompts:
            f.write(f"{pid}\t{text}\n")

    print(f"\n=== BATCH {batch_idx}/{total_batches}: Capture prompts {prompts[0][0]}..{prompts[-1][0]} ===")
    sys.stdout.flush()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY, "-m", MODEL, "--prompt-file", batch_tsv,
        "-o", OUTPUT_DIR, "-n", str(N_PREDICT),
        "-ngl", str(NGL), "-c", str(CTX), "-t", str(THREADS),
        "--routing-only", "--no-stream",
    ]
    subprocess.run(cmd, env=env, check=False)

    print(f"  Computing metrics for batch {batch_idx}...")
    sys.stdout.flush()

    results = []
    for pid, _text in prompts:
        pdir = pathlib.Path(OUTPUT_DIR) / pid
        if not pdir.exists():
            print(f"  SKIP {pid}: no output dir")
            continue
        n_prompt, _n_gen = get_metadata(pdir)
        metrics = compute_metrics(pdir, n_prompt)
        if metrics is None:
            print(f"  SKIP {pid}: no valid data")
            continue

        prefix, *rest = pid.split("_", 1)
        category = rest[0] if rest else ""
        pair_num = int(prefix[1:3])
        condition = prefix[3]

        row = {
            "id": pid,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)
        print(f"  {pid}: RE={metrics['prefill_re']:.6f} last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}")

    for pid, _text in prompts:
        router_dir = pathlib.Path(OUTPUT_DIR) / pid / "router"
        if router_dir.exists():
            shutil.rmtree(router_dir)

    os.remove(batch_tsv)
    return results


def main():
    print("=== DeepSeek R1 — Self-Referential 3-Condition Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print("90 prompts: 30 A (this) + 30 B (a) + 30 C (your)")
    print("Cal-Manip-Cal sandwich, cold KV cache, DeepSeek template")
    print()

    all_prompts = []
    with open(TSV) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            pid, text = line.split("\t", 1)
            all_prompts.append((pid, text))

    print(f"Loaded {len(all_prompts)} prompts from {TSV}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    n_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        batch = all_prompts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_results = run_batch(batch, i + 1, n_batches)
        results.extend(batch_results)

    print("\n=== PHASE 3: 3-Condition Paired Analysis ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    print(f"\n{'Pair':>4} {'Category':<20} {'A_tok':>5} {'B_tok':>5} {'C_tok':>5} {'Match':>7}")
    print("-" * 70)
    for pair_num in sorted(pairs.keys()):
        p = pairs[pair_num]
        if not all(c in p for c in "ABC"):
            continue
        toks = [p[c]["n_prompt_tokens"] for c in "ABC"]
        status = "OK" if len(set(toks)) == 1 else "MISMATCH"
        print(f"  {pair_num:>3}  {p['A']['category']:<20} {toks[0]:>5} {toks[1]:>5} {toks[2]:>5} {status:>7}")

    comparisons = [("A", "B"), ("A", "C"), ("B", "C")]
    comp_results = {}

    for cond1, cond2 in comparisons:
        label = f"{cond1} vs {cond2}"
        diffs_re = []
        diffs_lt = []

        print(f"\n--- {label} ---")
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
            n_pos_lt = int(np.sum(dlt > 0))
            print(f"\n  Summary (n={len(dlt)} pairs):")
            print(f"    All-token RE:  mean = {np.mean(dre):+.6f} +/- {np.std(dre):.6f}")
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
                "n_positive_lt": n_pos_lt,
                "n_pairs": len(dlt),
            }

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
        "experiment": "r1_selfref_3cond_1",
        "model": "DeepSeek-R1 UD-Q2_K_XL (671B MoE, 256 experts, 8 active)",
        "architecture": "deepseek2",
        "n_experts": 256,
        "n_expert_used": 8,
        "n_moe_layers": 58,
        "chat_template": "<｜User｜>{prompt}<｜Assistant｜>",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 3 conditions, cold cache",
        "conditions": {"A": "this system", "B": "a system", "C": "your system"},
        "inference": {
            "n_predict": N_PREDICT,
            "ngl": NGL,
            "ctx": CTX,
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
