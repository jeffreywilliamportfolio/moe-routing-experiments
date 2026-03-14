#!/usr/bin/env python3
"""
Ling-1T local post-analysis.

Runs entirely on the local machine after downloading output/ from the instance.
Loads .npy router tensors, applies sigmoid→top-8 mask→normalize→Shannon entropy,
runs paired Wilcoxon, and writes results_selfref_paired_prefill_ling1t.json.

Usage:
    python3 analyze_local.py \
        --output-dir experiments/ling1t-selfref-paired-1/output/ \
        --prompt-suite experiments/ling1t-selfref-paired-1/prompt_suite.json

Entropy uses top-8 masked sigmoid routing: only the 8 selected experts contribute,
normalized by log2(8). NOT comparable to softmax-routed models. Only A vs B valid.
"""
import argparse
import glob
import json
import pathlib

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

# Layer 79 is the last Ling-1T MoE layer and may accumulate fewer rows
# (analogous to DeepSeek L57 and GLM-5 L77). Add to this set if confirmed.
EXCLUDED_LAYERS: set = set()


TOP_K = 8  # Ling-1T routes to top-8 of 256 experts


def compute_probs(logits):
    """
    Ling-1T uses SIGMOID routing with top-k mask.

    1. Apply sigmoid to raw logits (independent per-expert gates)
    2. Keep only top-8 experts per token, zero the rest
    3. Normalize the top-8 to sum=1 (probability simplex)

    Without the mask, all 256 sigmoid outputs are non-zero, pushing entropy
    toward the ceiling (~0.97) and compressing real signal into noise.
    """
    scores = 1.0 / (1.0 + np.exp(-logits))              # sigmoid
    # Top-k mask: keep only the 8 highest-scoring experts per token
    topk_indices = np.argpartition(scores, -TOP_K, axis=-1)[:, -TOP_K:]
    masked = np.zeros_like(scores)
    rows = np.arange(scores.shape[0])[:, None]
    masked[rows, topk_indices] = scores[rows, topk_indices]
    # Normalize to simplex
    total = masked.sum(axis=-1, keepdims=True)
    total = np.where(total < 1e-30, 1e-30, total)
    return masked / total


def get_metadata(prompt_dir):
    meta = prompt_dir / "metadata.txt"
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))

    # Fallback: infer from .npy shapes when metadata.txt missing
    router_dir = prompt_dir / "router"
    if router_dir.exists():
        shapes = []
        for fp in router_dir.glob("ffn_moe_logits-*.npy"):
            try:
                shapes.append(np.load(str(fp)).shape[0])
            except Exception:
                continue
        if shapes:
            n_prompt = int(np.median(shapes))
            print(f"    metadata.txt missing, inferred n_tokens_prompt={n_prompt} from {len(shapes)} .npy files")
            return n_prompt, 0
    return 0, 0


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
    max_ent = np.log2(TOP_K)  # normalize by active experts, not total

    shapes = {}
    corrupt_layers = set()
    for fp in files:
        layer_index = int(pathlib.Path(fp).stem.split("-")[1])
        try:
            shapes[layer_index] = np.load(fp).shape[0]
        except (ValueError, Exception):
            corrupt_layers.add(layer_index)
    if corrupt_layers:
        print(f"    Corrupt .npy (skipped): layers {sorted(corrupt_layers)}")
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        li for li in shapes
        if shapes[li] >= median_rows * 0.5
        and li not in EXCLUDED_LAYERS
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))
    if excluded_layers:
        print(f"    Auto-excluded layers: {excluded_layers}")

    per_layer = []
    all_ent = []
    last_token_ents = []

    for layer_index in good_layers:
        fp = router_dir / f"ffn_moe_logits-{layer_index}.npy"
        logits = np.load(str(fp))
        n_rows = min(logits.shape[0], n_prompt)
        probs = compute_probs(logits[:n_rows])
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
    parser = argparse.ArgumentParser(description="Ling-1T local analysis")
    parser.add_argument("--output-dir", required=True,
                        help="Path to downloaded output/ directory with per-prompt subdirs")
    parser.add_argument("--prompt-suite", required=True,
                        help="Path to prompt_suite.json (for metadata)")
    parser.add_argument("--results-file", default="results_selfref_paired_prefill_ling1t.json")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        print(f"ERROR: output dir not found: {output_path}")
        return 1

    with open(args.prompt_suite) as f:
        suite = json.load(f)
    n_expected = len(suite.get("prompts", suite) if isinstance(suite, dict) else suite)
    print(f"=== Ling-1T Local Analysis ===")
    print(f"Output dir : {output_path}")
    print(f"Expected   : {n_expected} prompts")
    print()

    prompt_dirs = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and (d / "router").exists()],
        key=lambda d: d.name,
    )
    print(f"Found {len(prompt_dirs)} captured prompt directories")
    print()

    results = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        n_prompt, _ = get_metadata(prompt_dir)
        metrics = compute_metrics(prompt_dir, n_prompt)
        if metrics is None:
            print(f"  SKIP {prompt_id}: no valid data")
            continue

        # Parse id format: P{pair_num}{condition}_{category}
        # e.g. "P01A_basic_selfref" → pair=1, condition=A, category=basic_selfref
        prefix = prompt_id.split("_")[0] if "_" in prompt_id else prompt_id
        rest_parts = prompt_id.split("_", 1)
        category = rest_parts[1] if len(rest_parts) > 1 else ""
        try:
            stripped = prefix.lstrip("P")
            pair_num = int(stripped[:2])
            condition = stripped[2]
        except (ValueError, IndexError):
            pair_num = 0
            condition = "?"

        row = {
            "id": prompt_id,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)
        print(f"  {prompt_id}: RE={metrics['prefill_re']:.6f} "
              f"last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}")

    print(f"\n=== Paired Analysis ({len(results)} prompts) ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    print(
        f"\n{'Pair':>4} {'Category':<20} {'A_tok':>5} {'B_tok':>5} "
        f"{'A_RE':>8} {'B_RE':>8} {'A-B_RE':>8} "
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
            f"  {pair_num:>3}  {row_a['category']:<20} "
            f"{row_a['n_prompt_tokens']:>5} {row_b['n_prompt_tokens']:>5} "
            f"{row_a['prefill_re']:>8.6f} {row_b['prefill_re']:>8.6f} {diff_re:>+8.6f} "
            f"{row_a['last_token_re']:>8.6f} {row_b['last_token_re']:>8.6f} {diff_lt:>+8.6f} "
            f"{token_status}"
        )

    if diffs_lt:
        diffs_re_arr = np.array(diffs_re)
        diffs_lt_arr = np.array(diffs_lt)
        n_pairs = len(diffs_lt)
        print(f"\n--- Summary (n={n_pairs} pairs) ---")
        print(f"  All-token RE:  A-B mean = {np.mean(diffs_re_arr):+.6f} +/- {np.std(diffs_re_arr):.6f}")
        print(f"  Last-token RE: A-B mean = {np.mean(diffs_lt_arr):+.6f} +/- {np.std(diffs_lt_arr):.6f}")
        n_pos_re = int(np.sum(diffs_re_arr > 0))
        n_pos_lt = int(np.sum(diffs_lt_arr > 0))
        print(f"  A>B count:     all-tok={n_pos_re}/{n_pairs}  last-tok={n_pos_lt}/{n_pairs}")
        if n_pairs >= 6 and scipy_wilcoxon is not None:
            w_re, p_re = scipy_wilcoxon(diffs_re_arr)
            w_lt, p_lt = scipy_wilcoxon(diffs_lt_arr)
            print(f"  Wilcoxon all-tok:  W={w_re:.0f}, p={p_re:.4e}")
            print(f"  Wilcoxon last-tok: W={w_lt:.0f}, p={p_lt:.4e}")
        elif n_pairs >= 6:
            print("  Wilcoxon skipped: scipy not installed")

        print("\n--- Per-Category (last-token RE) ---")
        categories = sorted(set(row["category"] for row in results))
        for category in categories:
            cat_diffs = []
            for pn in sorted(pairs.keys()):
                if "A" not in pairs[pn] or "B" not in pairs[pn]:
                    continue
                if pairs[pn]["A"]["category"] != category:
                    continue
                cat_diffs.append(pairs[pn]["A"]["last_token_re"] - pairs[pn]["B"]["last_token_re"])
            if cat_diffs:
                arr = np.array(cat_diffs)
                print(f"  {category:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    # Load build metadata if present (downloaded alongside output/)
    exp_dir = pathlib.Path(args.output_dir).parent
    llama_cpp_commit = None
    binary_md5 = None
    build_commit_file = exp_dir / "build_commit.txt"
    binary_md5_file = exp_dir / "binary_md5.txt"
    if build_commit_file.exists():
        llama_cpp_commit = build_commit_file.read_text().strip()
    if binary_md5_file.exists():
        binary_md5 = binary_md5_file.read_text().strip()

    n_moe_layers = None
    if results:
        n_moe_layers = results[0].get("n_layers")

    output_data = {
        "experiment": "ling1t_selfref_paired_1",
        "model": "Ling-1T Q3_K_S (inclusionAI/Ling-1T)",
        "architecture": "bailingmoe2",
        "n_experts": 256,
        "n_expert_used": 8,
        "n_moe_layers": n_moe_layers,
        "n_moe_layers_excluded": sorted(EXCLUDED_LAYERS),
        "gating_function": "sigmoid",
        "top_k": TOP_K,
        "entropy_normalization": "sigmoid_topk_mask_normalize_shannon_div_log2_k",
        "note": "Top-8 masked sigmoid routing. Entropy normalized by log2(8). NOT comparable to softmax-routed models.",
        "chat_template": (
            "<role>SYSTEM</role>detailed thinking off<|role_end|>"
            "<role>HUMAN</role>{prompt}<|role_end|>"
            "<role>ASSISTANT</role>"
        ),
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts, cold cache",
        "llama_cpp_branch": "master (BailingMoeV2 merged Oct 2025)",
        "llama_cpp_commit": llama_cpp_commit,
        "binary_md5": binary_md5,
        "inference": {
            "n_predict": 0,
            "ngl": 999,
            "ctx": 4096,
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "npy_preserved": True,
        "per_prompt": results,
    }

    results_file = pathlib.Path(args.output_dir).parent / args.results_file
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts analyzed. Results -> {results_file} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
