#!/usr/bin/env python3
"""
Expert decomposition analysis for Ling-1T 5-condition experiments.

For each token at each layer, identifies WHICH 8 experts are active (top-8 sigmoid).
Compares expert sets across conditions to find:
  1. Addressivity experts: fire for "your X" but not "a X" (regardless of X)
  2. Content-specific experts: fire for system but not dog/cat
  3. Self-referential addressivity: fire for "your system" but not "your dog"

Requires output/ directories from all 3 experiments (dog, cat, system).

Usage (on instance):
    python3 expert_decomposition.py \
        --dog-dir /workspace/experiment-ling1t-dog/output \
        --cat-dir /workspace/experiment-ling1t-cat/output \
        --system-dir /workspace/experiment-ling1t-system/output \
        --results expert_decomposition_results.json
"""
import argparse
import glob
import json
import pathlib
from collections import Counter, defaultdict

import numpy as np

TOP_K = 8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_top_k_experts(logits, k=TOP_K):
    """Return top-k expert indices per token. Shape: [n_tokens, k]"""
    scores = sigmoid(logits)
    return np.argpartition(scores, -k, axis=-1)[:, -k:]


def get_metadata(prompt_dir):
    meta = prompt_dir / "metadata.txt"
    if meta.exists():
        info = {}
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        return int(info.get("n_tokens_prompt", 0))
    return 0


def load_expert_sets(prompt_dir, n_prompt):
    """Load all layers, return dict: layer -> [n_tokens, k] expert indices."""
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return {}

    result = {}
    for fp in sorted(router_dir.glob("ffn_moe_logits-*.npy")):
        layer = int(fp.stem.split("-")[1])
        if layer == 79:  # auto-excluded
            continue
        logits = np.load(str(fp))
        n = min(logits.shape[0], n_prompt)
        result[layer] = get_top_k_experts(logits[:n])
    return result


def expert_freq(expert_indices):
    """Count how often each expert appears across all tokens. Returns Counter."""
    c = Counter()
    for row in expert_indices:
        c.update(row.tolist())
    return c


def expert_sets_per_token(expert_indices):
    """Convert [n_tokens, k] to list of frozensets."""
    return [frozenset(row.tolist()) for row in expert_indices]


def diff_experts_paired(sets_a, sets_b, n_tokens):
    """For each token position, find experts in A but not B, and vice versa."""
    n = min(len(sets_a), len(sets_b), n_tokens)
    only_a = Counter()  # experts that fire in A but not B
    only_b = Counter()  # experts that fire in B but not A
    shared = Counter()  # experts that fire in both
    for i in range(n):
        sa, sb = sets_a[i], sets_b[i]
        for e in sa - sb:
            only_a[e] += 1
        for e in sb - sa:
            only_b[e] += 1
        for e in sa & sb:
            shared[e] += 1
    return only_a, only_b, shared, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog-dir", required=True)
    parser.add_argument("--cat-dir", required=True)
    parser.add_argument("--system-dir", required=True)
    parser.add_argument("--results", default="expert_decomposition_results.json")
    args = parser.parse_args()

    dog_dir = pathlib.Path(args.dog_dir)
    cat_dir = pathlib.Path(args.cat_dir)
    sys_dir = pathlib.Path(args.system_dir)

    # =========================================================
    # 1. Per-condition expert frequency profiles (aggregated)
    # =========================================================
    print("=== PHASE 1: Expert frequency profiles per condition ===\n")

    conditions = {"A": "this", "B": "a", "C": "your", "D": "the", "E": "their"}
    domains = {"dog": dog_dir, "cat": cat_dir, "system": sys_dir}

    # Aggregate expert frequencies: domain -> condition -> Counter over 256 experts
    freq = {d: {c: Counter() for c in "ABCDE"} for d in domains}
    token_counts = {d: {c: 0 for c in "ABCDE"} for d in domains}

    for domain, base_dir in domains.items():
        for prompt_dir in sorted(base_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            name = prompt_dir.name
            if len(name) < 4 or name[3] not in "ABCDE":
                continue
            cond = name[3]
            n_prompt = get_metadata(prompt_dir)
            if n_prompt == 0:
                continue

            expert_data = load_expert_sets(prompt_dir, n_prompt)
            for layer, indices in expert_data.items():
                freq[domain][cond] += expert_freq(indices)
                token_counts[domain][cond] += indices.shape[0]

    # Print top-20 experts per condition per domain
    for domain in domains:
        print(f"--- {domain.upper()} ---")
        for c in "ABCDE":
            top20 = freq[domain][c].most_common(20)
            total = token_counts[domain][c]
            top_str = ", ".join(f"E{e}({n/total:.2f})" for e, n in top20[:10])
            print(f"  {c} ({conditions[c]:>5}): total_firings={sum(freq[domain][c].values()):,}  top10: {top_str}")
        print()

    # =========================================================
    # 2. Differential expert analysis: C (your) vs B (a)
    # =========================================================
    print("=== PHASE 2: Differential experts — C (your) vs B (a) ===\n")

    # For each domain, find experts that fire MORE in C than B (normalized)
    your_vs_a = {}
    for domain in domains:
        fc = freq[domain]["C"]
        fb = freq[domain]["B"]
        tc = max(token_counts[domain]["C"], 1)
        tb = max(token_counts[domain]["B"], 1)

        # Normalized rate: firings per token-layer
        diff = {}
        all_experts = set(fc.keys()) | set(fb.keys())
        for e in all_experts:
            rate_c = fc[e] / tc
            rate_b = fb[e] / tb
            diff[e] = rate_c - rate_b

        # Sort by biggest positive diff (more in C=your)
        sorted_diff = sorted(diff.items(), key=lambda x: -x[1])
        your_vs_a[domain] = {e: d for e, d in sorted_diff}

        print(f"--- {domain.upper()}: experts firing MORE for 'your' than 'a' ---")
        print(f"  Top 20 (your > a):")
        for e, d in sorted_diff[:20]:
            rate_c = fc[e] / tc
            rate_b = fb[e] / tb
            print(f"    Expert {e:>3}: your={rate_c:.4f}  a={rate_b:.4f}  diff={d:+.4f}")

        print(f"  Top 20 (a > your):")
        for e, d in sorted_diff[-20:]:
            rate_c = fc[e] / tc
            rate_b = fb[e] / tb
            print(f"    Expert {e:>3}: your={rate_c:.4f}  a={rate_b:.4f}  diff={d:+.4f}")
        print()

    # =========================================================
    # 3. Cross-domain expert overlap
    # =========================================================
    print("=== PHASE 3: Cross-domain expert classification ===\n")

    # Threshold: expert must have diff > 0.001 to count as "your-preferring"
    THRESHOLD = 0.001

    your_experts = {}
    for domain in domains:
        your_experts[domain] = set(
            e for e, d in your_vs_a[domain].items() if d > THRESHOLD
        )
        print(f"  {domain}: {len(your_experts[domain])} experts prefer 'your' (diff > {THRESHOLD})")

    # Universal addressivity: prefer "your" in ALL three domains
    universal_addr = your_experts["dog"] & your_experts["cat"] & your_experts["system"]
    print(f"\n  Universal addressivity experts (your-preferring in dog AND cat AND system): {len(universal_addr)}")
    if universal_addr:
        print(f"    Experts: {sorted(universal_addr)}")

    # Animal-only addressivity: prefer "your" in dog+cat but NOT system
    animal_addr = (your_experts["dog"] & your_experts["cat"]) - your_experts["system"]
    print(f"\n  Animal-only addressivity (dog+cat but not system): {len(animal_addr)}")
    if animal_addr:
        print(f"    Experts: {sorted(animal_addr)}")

    # System-only addressivity: prefer "your" in system but NOT dog or cat
    system_only = your_experts["system"] - your_experts["dog"] - your_experts["cat"]
    print(f"\n  System-only addressivity (system but not dog/cat): {len(system_only)}")
    if system_only:
        print(f"    Experts: {sorted(system_only)}")

    # Dog+system but not cat
    dog_sys = (your_experts["dog"] & your_experts["system"]) - your_experts["cat"]
    print(f"  Dog+system (not cat): {len(dog_sys)}")

    # Cat+system but not dog
    cat_sys = (your_experts["cat"] & your_experts["system"]) - your_experts["dog"]
    print(f"  Cat+system (not dog): {len(cat_sys)}")

    # =========================================================
    # 4. A (this) vs B (a) — same decomposition
    # =========================================================
    print("\n=== PHASE 4: Differential experts — A (this) vs B (a) ===\n")

    this_vs_a = {}
    this_experts = {}
    for domain in domains:
        fa = freq[domain]["A"]
        fb = freq[domain]["B"]
        ta = max(token_counts[domain]["A"], 1)
        tb = max(token_counts[domain]["B"], 1)

        diff = {}
        all_experts = set(fa.keys()) | set(fb.keys())
        for e in all_experts:
            rate_a = fa[e] / ta
            rate_b = fb[e] / tb
            diff[e] = rate_a - rate_b

        sorted_diff = sorted(diff.items(), key=lambda x: -x[1])
        this_vs_a[domain] = {e: d for e, d in sorted_diff}
        this_experts[domain] = set(e for e, d in sorted_diff if d > THRESHOLD)

        print(f"--- {domain.upper()}: experts firing MORE for 'this' than 'a' ---")
        print(f"  {len(this_experts[domain])} experts prefer 'this' (diff > {THRESHOLD})")
        for e, d in sorted_diff[:10]:
            rate_a = fa[e] / ta
            rate_b = fb[e] / tb
            print(f"    Expert {e:>3}: this={rate_a:.4f}  a={rate_b:.4f}  diff={d:+.4f}")
        print()

    universal_this = this_experts["dog"] & this_experts["cat"] & this_experts["system"]
    system_only_this = this_experts["system"] - this_experts["dog"] - this_experts["cat"]
    print(f"  Universal 'this'-preferring: {len(universal_this)}")
    print(f"  System-only 'this'-preferring: {len(system_only_this)}")

    # =========================================================
    # 5. Per-layer expert divergence (C vs B) — find which layers differ most
    # =========================================================
    print("\n=== PHASE 5: Per-layer expert divergence (your vs a) ===\n")

    # For each layer, count how many token positions have different expert sets
    for domain in ["dog", "system"]:
        base = domains[domain]
        print(f"--- {domain.upper()}: Layer-by-layer Jaccard distance (C vs B, averaged over 30 pairs) ---")

        layer_jaccard = defaultdict(list)
        for pair_num in range(1, 31):
            c_dir = None
            b_dir = None
            for d in base.iterdir():
                if d.name.startswith(f"P{pair_num:02d}C_"):
                    c_dir = d
                elif d.name.startswith(f"P{pair_num:02d}B_"):
                    b_dir = d

            if not c_dir or not b_dir:
                continue

            n_c = get_metadata(c_dir)
            n_b = get_metadata(b_dir)
            n = min(n_c, n_b)

            data_c = load_expert_sets(c_dir, n_c)
            data_b = load_expert_sets(b_dir, n_b)

            for layer in sorted(set(data_c.keys()) & set(data_b.keys())):
                sets_c = expert_sets_per_token(data_c[layer])
                sets_b = expert_sets_per_token(data_b[layer])
                n_tok = min(len(sets_c), len(sets_b), n)

                jaccards = []
                for i in range(n_tok):
                    intersection = len(sets_c[i] & sets_b[i])
                    union = len(sets_c[i] | sets_b[i])
                    jaccards.append(1.0 - intersection / union if union > 0 else 0.0)
                layer_jaccard[layer].append(np.mean(jaccards))

        # Print top 10 most divergent layers
        layer_means = {l: np.mean(v) for l, v in layer_jaccard.items()}
        sorted_layers = sorted(layer_means.items(), key=lambda x: -x[1])
        print(f"  {'Layer':>5}  {'Mean Jaccard dist':>17}  (higher = more different expert sets)")
        for layer, jd in sorted_layers[:15]:
            print(f"  {layer:>5}  {jd:>17.4f}")
        print(f"  ...")
        for layer, jd in sorted_layers[-5:]:
            print(f"  {layer:>5}  {jd:>17.4f}")
        print()

    # =========================================================
    # Save results
    # =========================================================
    output = {
        "experiment": "expert_decomposition",
        "threshold": THRESHOLD,
        "n_experts_total": 256,
        "top_k": TOP_K,
        "your_vs_a_experts": {
            domain: {
                "n_preferring": len(your_experts[domain]),
                "experts": sorted(your_experts[domain]),
            }
            for domain in domains
        },
        "universal_addressivity_experts": sorted(universal_addr),
        "animal_only_addressivity_experts": sorted(animal_addr),
        "system_only_addressivity_experts": sorted(system_only),
        "this_vs_a_experts": {
            domain: {
                "n_preferring": len(this_experts[domain]),
                "experts": sorted(this_experts[domain]),
            }
            for domain in domains
        },
        "universal_this_experts": sorted(universal_this),
        "system_only_this_experts": sorted(system_only_this),
        "your_vs_a_rates": {
            domain: {
                str(e): round(d, 6)
                for e, d in sorted(your_vs_a[domain].items(), key=lambda x: -x[1])[:50]
            }
            for domain in domains
        },
    }
    with open(args.results, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
