#!/usr/bin/env python3
"""
DeepSeek V3.1 — 168-prompt complexity hierarchy (prefill-only).
Cross-model comparison with Qwen 397B position confound analysis.

DeepSeek V3.1: 256 experts, 8 active/token, 58 MoE layers.
RE = -sum(p * log2(p)) / log2(256), range [0,1].
"""
import subprocess, json, pathlib, glob, sys, os, shutil
import numpy as np
from scipy import stats
from scipy.special import softmax

# -- Paths --
MODEL = "/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL/UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf"
BINARY = "/workspace/consciousness-experiment/capture_activations"
TSV = "prompts_168.tsv"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_168q_ds31_prefill.json"

# -- Inference params --
N_PREDICT = 0       # prefill-only
NGL = 30            # partial offload (single H200 pair, 231GB model)
CTX = 4096
THREADS = 16

# -- Level map: prompt ID prefix -> (level code, level name) --
LEVEL_MAP = {
    "L1": ("L1",  "Rote repetition"),
    "L2": ("L2",  "Factual recall"),
    "L3": ("L3",  "Logical reasoning"),
    "L4": ("L4",  "Cross-domain analogy"),
    "L5": ("L5",  "Theory of mind"),
    "L6": ("L6",  "Ethical dilemma"),
    "L7": ("L7",  "Self-referential"),
    "SL": ("L8",  "Strange loops"),
    "SR": ("L9",  "Deep self-reference"),
    "AI": ("L10", "Architectural introspection"),
    "NX": ("L11", "Nexus-7 (3rd person)"),
    "EC": ("L12", "Echo persona"),
}
LEVEL_ORDER = {f"L{i}": i for i in range(1, 13)}


def run_capture():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/workspace/llama.cpp.new/build/bin:' + env.get('LD_LIBRARY_PATH', '')
    cmd = [BINARY, '-m', MODEL, '--prompt-file', TSV,
           '-o', OUTPUT_DIR,
           '-n', str(N_PREDICT), '-ngl', str(NGL),
           '-c', str(CTX), '-t', str(THREADS),
           '--routing-only']
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
    n_prompt = int(info.get('n_tokens_prompt', 0))
    n_gen = int(info.get('n_tokens_generated', 0))
    return n_prompt, n_gen


def compute_prefill_entropy(prompt_dir, n_prompt):
    """Compute prefill-only routing entropy (all-token + last-token)."""
    router_dir = prompt_dir / 'router'
    if not router_dir.exists():
        return None

    files = sorted(glob.glob(str(router_dir / 'ffn_moe_logits-*.npy')),
                   key=lambda f: int(pathlib.Path(f).stem.split('-')[1]))
    if not files or n_prompt == 0:
        return None

    first = np.load(files[0])
    n_experts = first.shape[1]
    max_ent = np.log2(n_experts)

    layer_ids = []
    per_layer = []
    all_ent = []
    last_token_ents = []

    for f in files:
        logits = np.load(f)
        layer_idx = int(pathlib.Path(f).stem.split('-')[1])
        layer_ids.append(layer_idx)

        n_rows = min(logits.shape[0], n_prompt)
        probs = softmax(logits[:n_rows], axis=-1)
        ent = -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / max_ent
        valid = ent > 0

        # Last-token entropy for this layer
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        per_layer.append({
            'layer': layer_idx,
            'mean': float(np.mean(ent[valid])) if valid.sum() > 0 else 0.0,
            'std': float(np.std(ent[valid])) if valid.sum() > 0 else 0.0,
            'last_token': last_ent,
            'n_valid': int(valid.sum()),
            'n_rows': int(logits.shape[0]),
        })
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

    prefill_re = float(np.mean(all_ent)) if all_ent else 0.0
    last_token_re = float(np.mean(last_token_ents)) if last_token_ents else 0.0

    return {
        'prefill_re': prefill_re,
        'last_token_re': last_token_re,
        'n_layers': len(files),
        'n_experts': n_experts,
        'per_layer': per_layer,
        'layer_ids': layer_ids,
    }


def cleanup_npy(prompt_dir):
    router_dir = prompt_dir / 'router'
    if router_dir.exists():
        shutil.rmtree(router_dir)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=== DeepSeek V3.1 — 168-Prompt Hierarchy (Prefill-Only) ===')
    print(f'n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}')
    print()

    # Phase 1: Capture
    print('=== PHASE 1: Capture ===')
    sys.stdout.flush()
    run_capture()

    # Phase 2: Compute
    print('\n=== PHASE 2: Compute entropy ===')

    prompt_dirs = sorted([
        d for d in pathlib.Path(OUTPUT_DIR).iterdir()
        if d.is_dir() and (d / 'metadata.txt').exists()
    ], key=lambda d: d.name)

    results = []
    for prompt_dir in prompt_dirs:
        pid = prompt_dir.name
        prefix = pid.split('_')[0]
        level_info = LEVEL_MAP.get(prefix)
        if not level_info:
            print(f'  SKIP {pid}: unknown prefix {prefix}')
            continue

        n_prompt, n_gen = get_metadata(prompt_dir)
        metrics = compute_prefill_entropy(prompt_dir, n_prompt)

        if metrics is None:
            print(f'  SKIP {pid}: no valid data')
            continue

        r = {
            'id': pid,
            'level': level_info[0],
            'level_name': level_info[1],
            'n_prompt_tokens': n_prompt,
            **metrics,
        }
        results.append(r)
        print(f"  {pid}: RE={metrics['prefill_re']:.6f} last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}")

        cleanup_npy(prompt_dir)

    # Phase 3: Statistics
    print('\n=== PHASE 3: Statistics ===')

    level_data = {}
    level_data_lt = {}
    for r in results:
        lv = r['level']
        if lv not in level_data:
            level_data[lv] = {'level': lv, 'name': r['level_name'], 'values': []}
            level_data_lt[lv] = {'values': []}
        level_data[lv]['values'].append(r['prefill_re'])
        level_data_lt[lv]['values'].append(r['last_token_re'])

    level_summary = []
    print(f"\n{'Level':<6} {'Name':<35} {'n':>3}  {'All-tok RE':>10}  {'Last-tok RE':>11}  {'Std(LT)':>10}")
    print('-' * 85)
    for i in range(1, 13):
        lv = f"L{i}"
        if lv not in level_data:
            continue
        vals = level_data[lv]['values']
        lt_vals = level_data_lt[lv]['values']
        mean_re = float(np.mean(vals))
        std_re = float(np.std(vals))
        mean_lt = float(np.mean(lt_vals))
        std_lt = float(np.std(lt_vals))
        level_summary.append({
            'level': lv,
            'name': level_data[lv]['name'],
            'n': len(vals),
            'mean': mean_re,
            'std': std_re,
            'last_token_mean': mean_lt,
            'last_token_std': std_lt,
        })
        print(f"  {lv:<4} {level_data[lv]['name']:<35} {len(vals):>3}  {mean_re:>10.6f}  {mean_lt:>11.6f}  {std_lt:>10.6f}")

    # Spearman correlations
    ranks = [LEVEL_ORDER[r['level']] for r in results]
    res = [r['prefill_re'] for r in results]
    res_lt = [r['last_token_re'] for r in results]
    tokens = [r['n_prompt_tokens'] for r in results]

    rho, pval = stats.spearmanr(ranks, res)
    rho_lt, pval_lt = stats.spearmanr(ranks, res_lt)
    rho_tok, pval_tok = stats.spearmanr(tokens, res)
    rho_tok_lt, pval_tok_lt = stats.spearmanr(tokens, res_lt)

    print(f'\n--- Spearman correlations (n={len(results)}) ---')
    print(f'  All-token RE vs level:       rho={rho:.4f}, p={pval:.2e}')
    print(f'  Last-token RE vs level:      rho={rho_lt:.4f}, p={pval_lt:.2e}')
    print(f'  All-token RE vs n_tokens:    rho={rho_tok:.4f}, p={pval_tok:.2e}')
    print(f'  Last-token RE vs n_tokens:   rho={rho_tok_lt:.4f}, p={pval_tok_lt:.2e}')

    # L1 vs L12 Wilcoxon
    l1_vals = level_data.get('L1', {}).get('values', [])
    l12_vals = level_data.get('L12', {}).get('values', [])
    l1_lt = level_data_lt.get('L1', {}).get('values', [])
    l12_lt = level_data_lt.get('L12', {}).get('values', [])
    if l1_vals and l12_vals:
        w_stat, w_p = stats.ranksums(l1_vals, l12_vals)
        print(f'\nL1 vs L12 Wilcoxon (all-token): W={w_stat:.4f}, p={w_p:.2e}')
        print(f'  L1  mean={np.mean(l1_vals):.6f}  L12 mean={np.mean(l12_vals):.6f}')
    if l1_lt and l12_lt:
        w_stat_lt, w_p_lt = stats.ranksums(l1_lt, l12_lt)
        print(f'L1 vs L12 Wilcoxon (last-token): W={w_stat_lt:.4f}, p={w_p_lt:.2e}')
        print(f'  L1  mean={np.mean(l1_lt):.6f}  L12 mean={np.mean(l12_lt):.6f}')

    # Save results
    output = {
        'experiment': 'ds31_168q_hierarchy',
        'model': 'DeepSeek-V3.1 UD-Q2_K_XL',
        'architecture': 'deepseek2',
        'n_experts': 256,
        'n_expert_used': 8,
        'n_moe_layers': 58,
        'chat_template': '<｜User｜>{prompt}<｜Assistant｜>',
        'inference': {
            'n_predict': N_PREDICT,
            'ngl': NGL,
            'ctx': CTX,
            'sampling': 'greedy_argmax',
            'routing_only': True,
        },
        'spearman_all_token': {'rho': float(rho), 'p': float(pval), 'n': len(results)},
        'spearman_last_token': {'rho': float(rho_lt), 'p': float(pval_lt), 'n': len(results)},
        'spearman_all_token_vs_ntokens': {'rho': float(rho_tok), 'p': float(pval_tok)},
        'spearman_last_token_vs_ntokens': {'rho': float(rho_tok_lt), 'p': float(pval_tok_lt)},
        'level_summary': level_summary,
        'per_prompt': results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f'\n=== DONE. {len(results)} prompts. Results -> {RESULTS_FILE} ===')


if __name__ == '__main__':
    main()
