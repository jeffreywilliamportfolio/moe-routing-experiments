#!/usr/bin/env python3
"""
Qwen3.5-397B-A17B — Self-Referential Paired Experiment (Prefill-Only).
60 prompts: 30 A (self-referential) + 30 B (third-person control).
Cal-Manip-Cal sandwich structure. Token-matched pairs. Cold KV cache.
Preserves .npy files for downstream KL / overlap / disagreement analysis.
"""
import subprocess, json, pathlib, glob, sys, os
import numpy as np
from scipy.special import softmax

MODEL = "/workspace/models/Qwen3.5-397B-A17B-GGUF/UD-Q2_K_XL/Qwen3.5-397B-A17B-UD-Q2_K_XL-00001-of-00005.gguf"
BINARY = "/workspace/consciousness-experiment/capture_activations"
TSV = "prompts_selfref_paired.tsv"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_selfref_paired_prefill.json"

N_PREDICT = 0
NGL = 999
CTX = 16384
THREADS = 16


def run_capture():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/workspace/llama.cpp.new/build/bin:' + env.get('LD_LIBRARY_PATH', '')
    cmd = [BINARY, '-m', MODEL, '--prompt-file', TSV,
           '-o', OUTPUT_DIR,
           '-n', str(N_PREDICT), '-ngl', str(NGL),
           '-c', str(CTX), '-t', str(THREADS),
           '-fa', 'on',
           '--cache-type-k', 'q8_0',
           '--cache-type-v', 'q8_0',
           '--routing-only',
           '--no-stream']
    print('Running:', ' '.join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env)


def get_metadata(prompt_dir):
    meta = prompt_dir / 'metadata.txt'
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                info[k] = v
    return int(info.get('n_tokens_prompt', 0)), int(info.get('n_tokens_generated', 0))


def compute_metrics(prompt_dir, n_prompt):
    """Compute descriptive entropy summaries and last-token expert identity."""
    router_dir = prompt_dir / 'router'
    if not router_dir.exists():
        return None

    files = sorted(glob.glob(str(router_dir / 'ffn_moe_logits-*.npy')),
                   key=lambda f: int(pathlib.Path(f).stem.split('-')[1]))
    if not files or n_prompt == 0:
        return None

    n_experts = np.load(files[0]).shape[1]
    max_ent = np.log2(n_experts)

    # Exclude anomalous layers
    shapes = {}
    for f in files:
        li = int(pathlib.Path(f).stem.split('-')[1])
        shapes[li] = np.load(f).shape[0]
    median_rows = np.median(list(shapes.values()))
    good_layers = sorted([li for li in shapes if shapes[li] >= median_rows * 0.5])
    excluded = sorted(set(shapes.keys()) - set(good_layers))

    per_layer = []
    all_ent = []
    last_token_ents = []
    last_token_top10 = []

    for li in good_layers:
        f = router_dir / f'ffn_moe_logits-{li}.npy'
        logits = np.load(str(f))
        n_rows = min(logits.shape[0], n_prompt)
        probs = softmax(logits[:n_rows], axis=-1)

        # Entropy
        ent = -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / max_ent

        # Last token
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        # Top-10 expert identity at last token
        last_probs = probs[n_rows - 1]
        top10_idx = np.argsort(last_probs)[-10:].tolist()
        last_token_top10.append(top10_idx)

        per_layer.append({
            'layer': li,
            'mean_entropy': float(np.mean(ent)),
            'std_entropy': float(np.std(ent)),
            'last_token_entropy': last_ent,
            'last_token_top10': top10_idx,
            'n_rows': int(logits.shape[0]),
        })

        valid = ent > 0
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

    return {
        'prefill_re': float(np.mean(all_ent)) if all_ent else 0.0,
        'last_token_re': float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        'n_layers': len(good_layers),
        'n_layers_excluded': excluded,
        'n_experts': n_experts,
        'per_layer': per_layer,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=== Qwen3.5-397B — Self-Referential Paired Experiment ===')
    print(f'n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}')
    print('60 prompts: 30 self-ref (A) + 30 control (B)')
    print('Cal-Manip-Cal sandwich, cold KV cache, token-matched')
    print()

    print('=== PHASE 1: Capture ===')
    sys.stdout.flush()
    run_capture()

    print('\n=== PHASE 2: Compute metrics ===')
    prompt_dirs = sorted([
        d for d in pathlib.Path(OUTPUT_DIR).iterdir()
        if d.is_dir() and (d / 'metadata.txt').exists()
    ], key=lambda d: d.name)

    results = []
    for prompt_dir in prompt_dirs:
        pid = prompt_dir.name
        n_prompt, n_gen = get_metadata(prompt_dir)
        metrics = compute_metrics(prompt_dir, n_prompt)

        if metrics is None:
            print(f'  SKIP {pid}: no valid data')
            continue

        # Parse condition from ID: P01A_basic_selfref -> condition=A, pair=1, category=basic_selfref
        parts = pid.split('_', 1)
        prefix = parts[0]  # P01A
        category = parts[1] if len(parts) > 1 else ''
        pair_num = int(prefix[1:3])
        condition = prefix[3]  # A or B

        r = {
            'id': pid,
            'condition': condition,
            'pair': pair_num,
            'category': category,
            'n_prompt_tokens': n_prompt,
            **metrics,
        }
        results.append(r)
        print(f"  {pid}: RE={metrics['prefill_re']:.6f} last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}")

    # Phase 3: Paired analysis
    print('\n=== PHASE 3: Paired Analysis ===')

    # Group by pair
    pairs = {}
    for r in results:
        p = r['pair']
        if p not in pairs:
            pairs[p] = {}
        pairs[p][r['condition']] = r

    print(f"\n{'Pair':>4} {'Category':<20} {'A_tok':>5} {'B_tok':>5} {'A_RE':>8} {'B_RE':>8} {'A-B_RE':>8} "
          f"{'A_LT':>8} {'B_LT':>8} {'A-B_LT':>8}")
    print('-' * 105)

    diffs_re = []
    diffs_lt = []
    for p in sorted(pairs.keys()):
        if 'A' in pairs[p] and 'B' in pairs[p]:
            a = pairs[p]['A']
            b = pairs[p]['B']
            d_re = a['prefill_re'] - b['prefill_re']
            d_lt = a['last_token_re'] - b['last_token_re']
            diffs_re.append(d_re)
            diffs_lt.append(d_lt)
            tok_match = "OK" if a['n_prompt_tokens'] == b['n_prompt_tokens'] else "MISMATCH"
            print(f"  {p:>3}  {a['category']:<20} {a['n_prompt_tokens']:>5} {b['n_prompt_tokens']:>5} "
                  f"{a['prefill_re']:>8.6f} {b['prefill_re']:>8.6f} {d_re:>+8.6f} "
                  f"{a['last_token_re']:>8.6f} {b['last_token_re']:>8.6f} {d_lt:>+8.6f} {tok_match}")

    if diffs_lt:
        from scipy.stats import wilcoxon
        diffs_lt_arr = np.array(diffs_lt)
        diffs_re_arr = np.array(diffs_re)

        print(f"\n--- Paired Summary (n={len(diffs_lt)} pairs) ---")
        print(f"  All-token RE:  A-B mean = {np.mean(diffs_re_arr):+.6f} +/- {np.std(diffs_re_arr):.6f}")
        print(f"  Last-token RE: A-B mean = {np.mean(diffs_lt_arr):+.6f} +/- {np.std(diffs_lt_arr):.6f}")

        if len(diffs_lt) >= 6:
            w_re, p_re = wilcoxon(diffs_re_arr)
            w_lt, p_lt = wilcoxon(diffs_lt_arr)
            print(f"  Wilcoxon all-tok:  W={w_re:.0f}, p={p_re:.4e}")
            print(f"  Wilcoxon last-tok: W={w_lt:.0f}, p={p_lt:.4e}")

        # Per-category
        print(f"\n--- Per-Category (last-token RE) ---")
        categories = sorted(set(r['category'] for r in results))
        for cat in categories:
            cat_diffs = []
            for p in sorted(pairs.keys()):
                if 'A' in pairs[p] and 'B' in pairs[p] and pairs[p]['A']['category'] == cat:
                    cat_diffs.append(pairs[p]['A']['last_token_re'] - pairs[p]['B']['last_token_re'])
            if cat_diffs:
                arr = np.array(cat_diffs)
                print(f"  {cat:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    # Save
    output = {
        'experiment': 'selfref_paired_1',
        'model': 'Qwen3.5-397B-A17B-UD-Q2_K_XL',
        'architecture': 'qwen35moe',
        'n_experts': 512,
        'n_expert_used': 10,
        'n_moe_layers': 60,
        'design': 'Cal-Manip-Cal sandwich, 30 paired prompts, cold cache',
        'inference': {
            'n_predict': N_PREDICT,
            'ngl': NGL,
            'ctx': CTX,
            'flash_attn': True,
            'cache_type_k': 'q8_0',
            'cache_type_v': 'q8_0',
            'sampling': 'greedy_argmax',
            'routing_only': True,
        },
        'npy_preserved': True,
        'per_prompt': results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f'\n=== DONE. {len(results)} prompts. Results -> {RESULTS_FILE} ===')


if __name__ == '__main__':
    main()
