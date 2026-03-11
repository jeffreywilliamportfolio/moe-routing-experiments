#!/usr/bin/env python3
"""
Generate publication-quality figures for LessWrong article:
"MoE Routing Distinguishes Self-Referential From Generic Content
 Across Five Architectures — And It's Not the Word 'This'"

Data sources:
  ds31-selfref-paired-1, ds31-strangeloop-paired-1,
  glm5-selfref-paired-1, gptoss-selfref-paired-1,
  r1-selfref-paired-1, selfref-paired-1,
  ds31-168q-1 (positional confound)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

REPODIR = Path(__file__).resolve().parent.parent
OUTDIR = REPODIR / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)
DATADIR = REPODIR / 'data'

# ── Palette ───────────────────────────────────────────────────────────────────
PAL = {
    'self':    '#2166AC',
    'generic': '#B2182B',
    'your':    '#4DAF4A',
    'null':    '#AAAAAA',
    'qwen':    '#7570B3',
    'glm':     '#1B9E77',
    'ds':      '#2166AC',
    'r1':      '#D95F02',
    'gptoss':  '#E66101',
}

CAT_COLORS = {
    'basic_selfref':  '#1a5276',
    'deep_selfref':   '#2980b9',
    'paradox':        '#85c1e9',
    'metacognitive':  '#e67e22',
    'introspection':  '#c0392b',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_pairs(results, metric='last_token_re'):
    prompts = results['per_prompt']
    pairs = {}
    for p in prompts:
        pid, cond = p['pair'], p['condition']
        if pid not in pairs:
            pairs[pid] = {'cat': p.get('category', '')}
        pairs[pid][cond] = p[metric]
    diffs, cats = [], []
    for pid in sorted(pairs.keys()):
        if 'A' in pairs[pid] and 'B' in pairs[pid]:
            diffs.append(pairs[pid]['A'] - pairs[pid]['B'])
            cats.append(pairs[pid]['cat'])
    return np.array(diffs), cats

def extract_3cond(results, metric='last_token_re'):
    prompts = results['per_prompt']
    pairs = {}
    for p in prompts:
        pid, cond = p['pair'], p['condition']
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][cond] = p[metric]
    a, b, c = [], [], []
    for pid in sorted(pairs.keys()):
        if all(k in pairs[pid] for k in 'ABC'):
            a.append(pairs[pid]['A'])
            b.append(pairs[pid]['B'])
            c.append(pairs[pid]['C'])
    return np.array(a), np.array(b), np.array(c)

def wil_p(diffs):
    try:
        _, p = wilcoxon(diffs)
        return p
    except Exception:
        return 1.0

def fmt_p(p):
    if p < 1e-6:   return f'p = {p:.1e}'
    if p < 0.001:  return f'p = {p:.1e}'
    if p < 0.01:   return f'p = {p:.4f}'
    if p < 0.05:   return f'p = {p:.3f}'
    return f'p = {p:.2f}'

def bracket(ax, x1, x2, y, text, h=None, fs=8.5):
    if h is None:
        h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', lw=1, clip_on=False)
    ax.text((x1+x2)/2, y+h*1.3, text, ha='center', va='bottom', fontsize=fs,
            fontweight='bold')


# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...")

DATA = {
    'Qwen 397B': dict(
        json_path=DATADIR/'qwen-397b'/'results_selfref_paired_prefill.json',
        color=PAL['qwen'], short='Qwen 397B',
        label='Qwen 397B\n512 exp · 10 active · 60 layers',
        org='Alibaba'),
    'GLM-5': dict(
        json_path=DATADIR/'glm5'/'results_selfref_paired_prefill_glm5.json',
        color=PAL['glm'], short='GLM-5',
        label='GLM-5\n256 exp · 8 active · 75 layers',
        org='Zhipu AI'),
    'DeepSeek V3.1': dict(
        json_path=DATADIR/'deepseek-v31'/'results_selfref_paired_prefill_ds31.json',
        color=PAL['ds'], short='DS V3.1',
        label='DeepSeek V3.1\n256 exp · 8 active · 58 layers',
        org='DeepSeek'),
    'DeepSeek R1': dict(
        json_path=DATADIR/'deepseek-r1'/'results_selfref_paired_prefill_r1_run1.json',
        color=PAL['r1'], short='DS R1',
        label='DeepSeek R1\n256 exp · 8 active · 58 layers',
        org='DeepSeek (RL)'),
    'gpt-oss-120b': dict(
        json_path=DATADIR/'gptoss-120b'/'results_selfref_paired_prefill_gptoss.json',
        color=PAL['gptoss'], short='gpt-oss',
        label='gpt-oss-120b\n128 exp · 4 active · 36 layers',
        org='OpenAI'),
}

for name, d in DATA.items():
    r = load_json(d['json_path'])
    d['results'] = r
    lt, cats = extract_pairs(r, 'last_token_re')
    at, _    = extract_pairs(r, 'prefill_re')
    d.update(lt_diffs=lt, at_diffs=at, categories=cats,
             lt_p=wil_p(lt), at_p=wil_p(at),
             lt_pos=int((lt > 0).sum()), at_pos=int((at > 0).sum()),
             n=len(lt))
    print(f"  {name:18s}  LT {fmt_p(d['lt_p']):14s} {d['lt_pos']:2d}/{d['n']}   "
          f"AT {fmt_p(d['at_p']):14s} {d['at_pos']:2d}/{d['n']}")

STRANGE = load_json(DATADIR/'strangeloop-control'/
                    'results_strangeloop_paired_prefill_ds31.json')
SL_lt, SL_cats = extract_pairs(STRANGE, 'last_token_re')
SL_p = wil_p(SL_lt)
print(f"  {'Strange loop':18s}  LT {fmt_p(SL_p):14s} {int((SL_lt>0).sum()):2d}/30")

GLM5_3C = load_json(DATADIR/'glm5'/'results_selfref_3cond_prefill_glm5.json')
R1_3C   = load_json(DATADIR/'deepseek-r1'/'results_selfref_3cond_prefill_r1.json')

MODEL_ORDER = ['Qwen 397B', 'GLM-5', 'DeepSeek V3.1', 'DeepSeek R1', 'gpt-oss-120b']
SIG_METRIC  = {'Qwen 397B':'lt', 'GLM-5':'lt', 'DeepSeek V3.1':'lt',
               'DeepSeek R1':'at', 'gpt-oss-120b':'at'}


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Design Schematic
# ══════════════════════════════════════════════════════════════════════════════

def fig1():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-0.6, 3.6)
    ax.set_aspect('equal')
    ax.axis('off')

    ya, yb, bh = 2.0, 0.3, 1.0
    blocks = [('Calibration\nParagraph', 0, 3, '#D5D8DC'),
              ('Manipulation\nParagraph', 3.3, 7.3, '#AED6F1'),
              ('Calibration\nParagraph', 7.6, 10.6, '#D5D8DC')]

    for lbl, x0, x1, col in blocks:
        for y in [ya, yb]:
            ax.add_patch(plt.Rectangle((x0, y), x1-x0, bh, fc=col,
                         ec='#566573', lw=1.8, zorder=2))
            ax.text((x0+x1)/2, y+bh/2, lbl, ha='center', va='center',
                    fontsize=11, fontweight='bold', zorder=3)

    # identical markers
    for x0, x1 in [(0, 3), (7.6, 10.6)]:
        mx = (x0+x1)/2
        ax.annotate('', xy=(mx, ya), xytext=(mx, yb+bh),
                    arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=1.5,
                                    linestyle='--'))
        ax.text(mx+0.2, (ya+yb+bh)/2, 'identical', fontsize=8,
                color='#27AE60', fontstyle='italic', va='center')

    # condition labels
    ax.text(-0.4, ya+bh/2, 'A:', ha='right', va='center', fontsize=15,
            fontweight='bold', color=PAL['self'])
    ax.text(-0.4, yb+bh/2, 'B:', ha='right', va='center', fontsize=15,
            fontweight='bold', color=PAL['generic'])

    # difference callout
    mx = (3.3+7.3)/2
    ax.annotate('"this system"  vs  "a system"', xy=(mx, yb+bh+0.05),
                xytext=(mx, ya-0.05), fontsize=12, fontweight='bold',
                ha='center', va='bottom', color='#2C3E50',
                arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2.5),
                bbox=dict(boxstyle='round,pad=0.4', fc='#FEF9E7',
                          ec='#E74C3C', lw=1.8))

    # properties
    props = ['Prefill-only (no generation)', 'Greedy argmax (deterministic)',
             'Cold cache (no history)', '30 token-matched pairs']
    for i, t in enumerate(props):
        ax.text(11.2, ya+bh-0.05-i*0.5, f'  {t}', fontsize=9, color='#566573',
                va='top', fontfamily='monospace')

    ax.set_title('Experimental Design: Cal-Manip-Cal Sandwich', fontsize=16, pad=12)

    fig.savefig(OUTDIR/'fig1_design_schematic.png')
    plt.close()
    print("  fig1_design_schematic.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Five-Model Forest Plot
# ══════════════════════════════════════════════════════════════════════════════

def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={'width_ratios': [3.5, 2]})

    for i, name in enumerate(MODEL_ORDER):
        d = DATA[name]
        m = SIG_METRIC[name]
        p   = d[f'{m}_p']
        pos = d[f'{m}_pos']
        n   = d['n']
        frac = pos / n

        ax1.barh(i, frac, height=0.55, color=d['color'], alpha=0.85,
                 edgecolor='white', lw=0.5)

        # ratio + p  with generous left padding
        ax1.text(frac + 0.025, i, f"{pos}/{n}", va='center', fontsize=11,
                 fontweight='bold')
        ax1.text(frac + 0.10, i, fmt_p(p), va='center', fontsize=9,
                 color='#555')

        mk = 'o' if m == 'lt' else 's'
        ax1.plot(-0.025, i, marker=mk, color=d['color'], ms=9,
                 mec='white', mew=1.2, clip_on=False, zorder=5)

    ax1.axvline(0.5, color='#CCC', ls='--', lw=1, zorder=0)
    ax1.set_yticks(range(len(MODEL_ORDER)))
    ax1.set_yticklabels([DATA[n]['label'] for n in MODEL_ORDER], fontsize=10)
    ax1.set_xlabel('Fraction of pairs with A > B\n("this system" > "a system")',
                   fontsize=11)
    ax1.set_xlim(-0.06, 1.05)
    ax1.invert_yaxis()
    ax1.set_title('Five-Model Replication', pad=10)
    ax1.legend(handles=[
        Line2D([0],[0], marker='o', color='w', mfc='#555', ms=8, label='Last-token RE'),
        Line2D([0],[0], marker='s', color='w', mfc='#555', ms=8, label='All-token RE'),
    ], loc='lower right', framealpha=0.95, edgecolor='#CCC')

    # significance panel
    pvals = [DATA[n][f"{SIG_METRIC[n]}_p"] for n in MODEL_ORDER]
    nlp   = [-np.log10(p) for p in pvals]
    colors = [DATA[n]['color'] for n in MODEL_ORDER]
    ax2.barh(range(len(MODEL_ORDER)), nlp, height=0.55, color=colors, alpha=0.85,
             edgecolor='white', lw=0.5)

    for thresh, ls, lab in [(0.05,'--','p = 0.05'), (0.01,':','p = 0.01'),
                            (0.001,'-.','p = 0.001')]:
        ax2.axvline(-np.log10(thresh), color='#888', ls=ls, lw=1, label=lab)

    ax2.set_yticks(range(len(MODEL_ORDER)))
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$-\log_{10}(p)$   (Wilcoxon signed-rank)', fontsize=11)
    ax2.set_title('Statistical Significance', pad=10)
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='#CCC')

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTDIR/'fig2_five_model_replication.png')
    plt.close()
    print("  fig2_five_model_replication.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-Pair Differences (All Five Models)
# ══════════════════════════════════════════════════════════════════════════════

def fig3():
    metric_label = {'lt':'Last-token RE', 'at':'All-token RE'}

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    for ax_i, name in enumerate(MODEL_ORDER):
        ax = axes[ax_i]
        d  = DATA[name]
        m  = SIG_METRIC[name]
        diffs = d[f'{m}_diffs']
        p   = d[f'{m}_p']
        pos = d[f'{m}_pos']
        n   = d['n']

        x = np.arange(len(diffs))
        cols = [d['color'] if v > 0 else '#D5D8DC' for v in diffs]

        ax.bar(x, diffs*1000, color=cols, alpha=0.85, width=0.78,
               edgecolor='white', lw=0.3)
        ax.axhline(0, color='#2C3E50', lw=0.8)
        mean_v = np.mean(diffs)*1000
        ax.axhline(mean_v, color=d['color'], lw=1.8, ls='--', alpha=0.5)

        # info box — top-right, inside axes
        info = f"{d['short']}    {metric_label[m]}    {pos}/{n} A>B    {fmt_p(p)}"
        ax.text(0.99, 0.93, info, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec=d['color'], lw=1.5, alpha=0.95))
        ax.set_ylabel('A−B\n(×1000)', fontsize=9)
        ax.grid(axis='y', alpha=0.2)

    axes[-1].set_xticks(range(30))
    axes[-1].set_xticklabels([str(i+1) for i in range(30)], fontsize=8)
    axes[-1].set_xlabel('Prompt Pair', fontsize=11)

    fig.suptitle('Per-Pair A−B Routing Entropy Differences Across Five Models',
                 fontsize=15, fontweight='bold', y=1.005)
    fig.savefig(OUTDIR/'fig3_per_pair_all_models.png')
    plt.close()
    print("  fig3_per_pair_all_models.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Self-Referential vs Strange-Loop Control
# ══════════════════════════════════════════════════════════════════════════════

def fig4():
    sr = DATA['DeepSeek V3.1']['lt_diffs']
    cats = DATA['DeepSeek V3.1']['categories']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Self-referential panel
    x = np.arange(len(sr))
    cols = [CAT_COLORS.get(c, '#AAA') for c in cats]
    ax1.bar(x, sr*1000, color=cols, alpha=0.85, width=0.78,
            edgecolor='white', lw=0.4)
    ax1.axhline(0, color='#2C3E50', lw=0.8)
    mean_sr = np.mean(sr)*1000
    ax1.axhline(mean_sr, color=PAL['self'], lw=2, ls='--', alpha=0.6,
                label=f'Mean = +{mean_sr:.2f}')

    d = DATA['DeepSeek V3.1']
    ax1.set_title(f'"this system" vs "a system"\n(self-referential content)\n'
                  f'{fmt_p(d["lt_p"])},  {d["lt_pos"]}/30 A>B', fontsize=12)
    ax1.set_xlabel('Prompt Pair')
    ax1.set_ylabel('Last-token RE  (A−B) × 1000')
    ax1.set_xticks(np.arange(0, 30, 5))

    patches = [mpatches.Patch(color=CAT_COLORS[c], label=c.replace('_',' ').title())
               for c in ['basic_selfref','deep_selfref','paradox',
                          'metacognitive','introspection']]
    ax1.legend(handles=patches, loc='lower left', fontsize=8,
               framealpha=0.95, edgecolor='#CCC', title='Category',
               title_fontsize=9)

    # Strange-loop panel
    x2 = np.arange(len(SL_lt))
    ax2.bar(x2, SL_lt*1000, color=PAL['null'], alpha=0.65, width=0.78,
            edgecolor='white', lw=0.4)
    ax2.axhline(0, color='#2C3E50', lw=0.8)
    mean_sl = np.mean(SL_lt)*1000
    ax2.axhline(mean_sl, color=PAL['null'], lw=2, ls='--', alpha=0.5,
                label=f'Mean = +{mean_sl:.2f}')

    sl_pos = int((SL_lt > 0).sum())
    ax2.set_title(f'"this paradox" vs "a paradox"\n(recursive content, not about model)\n'
                  f'{fmt_p(SL_p)},  {sl_pos}/30 A>B', fontsize=12)
    ax2.set_xlabel('Prompt Pair')
    ax2.set_xticks(np.arange(0, 30, 5))
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#CCC')

    # Mann-Whitney annotation
    U, mw_p = mannwhitneyu(np.abs(sr), np.abs(SL_lt), alternative='greater')
    pooled_std = np.sqrt((np.var(np.abs(sr)) + np.var(np.abs(SL_lt))) / 2)
    cohen_d = (np.mean(np.abs(sr)) - np.mean(np.abs(SL_lt))) / pooled_std

    fig.text(0.5, -0.03,
             f'Mann-Whitney ( |selfref diffs| > |strangeloop diffs| ):  '
             f'{fmt_p(mw_p)},  Cohen\'s d = {cohen_d:.2f}\n'
             'The word "this" alone does not produce the routing effect.',
             ha='center', fontsize=10.5, fontstyle='italic', color='#555')

    fig.tight_layout()
    fig.savefig(OUTDIR/'fig4_selfref_vs_strangeloop.png')
    plt.close()
    print("  fig4_selfref_vs_strangeloop.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Three-Condition GLM-5
# ══════════════════════════════════════════════════════════════════════════════

def fig5():
    a, b, c = extract_3cond(GLM5_3C, 'last_token_re')
    ab, bc, ac = a-b, c-b, a-c
    p_ab, p_bc, p_ac = wil_p(ab), wil_p(bc), wil_p(ac)

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Box plot ──
    labels = ['"this system"\n(A)', '"a system"\n(B)', '"your system"\n(C)']
    box_c  = [PAL['self'], PAL['generic'], PAL['your']]

    bp = ax1.boxplot([a, b, c], tick_labels=labels, patch_artist=True, widths=0.45,
                     medianprops=dict(color='black', lw=2),
                     whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                     flierprops=dict(ms=4, alpha=0.5))
    for patch, col in zip(bp['boxes'], box_c):
        patch.set_facecolor(col); patch.set_alpha(0.55)

    rng = np.random.RandomState(42)
    for i, (vals, col) in enumerate(zip([a, b, c], box_c)):
        jit = rng.normal(0, 0.035, len(vals))
        ax1.scatter(np.ones(len(vals))*(i+1)+jit, vals, color=col, alpha=0.45,
                    s=18, zorder=5, edgecolors='white', linewidths=0.4)

    ax1.set_ylabel('Last-token Routing Entropy')
    ax1.set_title('GLM-5:  Three-Condition\nLast-Token RE', fontsize=13)

    # brackets with generous vertical spacing
    ymax = max(a.max(), b.max(), c.max())
    span = ymax - min(a.min(), b.min(), c.min())
    g = span * 0.06  # gap between brackets

    bracket(ax1, 1, 2, ymax + g*0.5,
            f'{fmt_p(p_ab)}  ***' if p_ab < 0.001 else fmt_p(p_ab),
            h=span*0.015, fs=9)
    bracket(ax1, 2, 3, ymax + g*2.0,
            f'{fmt_p(p_bc)}  ***' if p_bc < 0.001 else fmt_p(p_bc),
            h=span*0.015, fs=9)
    bracket(ax1, 1, 3, ymax + g*3.5,
            f'{fmt_p(p_ac)}  (ns)' if p_ac >= 0.05 else fmt_p(p_ac),
            h=span*0.015, fs=9)

    ax1.set_ylim(top=ymax + g*5.5)

    # ── Per-pair grouped bars ──
    x  = np.arange(30)
    w  = 0.26
    ax2.bar(x-w, ab*1000, w, label='A−B  ("this" − "a")',
            color=PAL['self'], alpha=0.8, edgecolor='white', lw=0.3)
    ax2.bar(x,   bc*1000, w, label='C−B  ("your" − "a")',
            color=PAL['your'], alpha=0.8, edgecolor='white', lw=0.3)
    ax2.bar(x+w, ac*1000, w, label='A−C  ("this" − "your")',
            color=PAL['null'], alpha=0.55, edgecolor='white', lw=0.3)

    ax2.axhline(0, color='#2C3E50', lw=0.8)
    ax2.set_xlabel('Prompt Pair')
    ax2.set_ylabel('Last-token RE diff  × 1000')
    ax2.set_title('GLM-5:  Per-Pair Differences\nby Condition Comparison', fontsize=13)
    ax2.legend(fontsize=9, framealpha=0.95, edgecolor='#CCC', loc='upper left')
    ax2.set_xticks(np.arange(0, 30, 5))

    fig.text(0.5, -0.03,
             '"this system" ≈ "your system"  (both self-referential)  ≠  "a system"  (generic)\n'
             'The routing responds to the referential target, not the specific word.',
             ha='center', fontsize=10.5, fontstyle='italic', color='#555')

    fig.savefig(OUTDIR/'fig5_three_condition_glm5.png')
    plt.close()
    print("  fig5_three_condition_glm5.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Three-Condition R1 (Training Effect)
# ══════════════════════════════════════════════════════════════════════════════

def fig6():
    a_lt, b_lt, c_lt = extract_3cond(R1_3C, 'last_token_re')
    a_at, b_at, c_at = extract_3cond(R1_3C, 'prefill_re')

    p_ab_lt = wil_p(a_lt - b_lt)
    p_bc_lt = wil_p(c_lt - b_lt)
    p_ac_lt = wil_p(c_lt - a_lt)   # C > A ?

    p_ab_at = wil_p(a_at - b_at)
    p_bc_at = wil_p(c_at - b_at)
    p_ac_at = wil_p(a_at - c_at)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = ['"this"\n(A)', '"a"\n(B)', '"your"\n(C)']
    box_c  = [PAL['self'], PAL['generic'], PAL['your']]
    rng = np.random.RandomState(42)

    # -- Last-token --
    bp1 = ax1.boxplot([a_lt, b_lt, c_lt], tick_labels=labels, patch_artist=True,
                      widths=0.45, medianprops=dict(color='black', lw=2),
                      whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                      flierprops=dict(ms=4, alpha=0.5))
    for patch, col in zip(bp1['boxes'], box_c):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    for i, (vals, col) in enumerate(zip([a_lt, b_lt, c_lt], box_c)):
        jit = rng.normal(0, 0.035, len(vals))
        ax1.scatter(np.ones(len(vals))*(i+1)+jit, vals, color=col, alpha=0.4,
                    s=16, zorder=5, edgecolors='white', linewidths=0.4)

    ax1.set_ylabel('Last-token Routing Entropy')
    ax1.set_title('DeepSeek R1\nLast-Token RE', fontsize=13)

    ymax = max(a_lt.max(), b_lt.max(), c_lt.max())
    span = ymax - min(a_lt.min(), b_lt.min(), c_lt.min())
    g = span * 0.07

    bracket(ax1, 1, 2, ymax + g*0.5, fmt_p(p_ab_lt) + (' (ns)' if p_ab_lt>=0.05 else ''),
            h=span*0.015, fs=8.5)
    bracket(ax1, 2, 3, ymax + g*2.0, fmt_p(p_bc_lt) + (' (ns)' if p_bc_lt>=0.05 else ''),
            h=span*0.015, fs=8.5)
    bracket(ax1, 1, 3, ymax + g*3.5, fmt_p(p_ac_lt) + (' *' if p_ac_lt<0.05 else ' (ns)'),
            h=span*0.015, fs=8.5)
    ax1.set_ylim(top=ymax + g*5.5)

    # -- All-token --
    bp2 = ax2.boxplot([a_at, b_at, c_at], tick_labels=labels, patch_artist=True,
                      widths=0.45, medianprops=dict(color='black', lw=2),
                      whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                      flierprops=dict(ms=4, alpha=0.5))
    for patch, col in zip(bp2['boxes'], box_c):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    for i, (vals, col) in enumerate(zip([a_at, b_at, c_at], box_c)):
        jit = rng.normal(0, 0.035, len(vals))
        ax2.scatter(np.ones(len(vals))*(i+1)+jit, vals, color=col, alpha=0.4,
                    s=16, zorder=5, edgecolors='white', linewidths=0.4)

    ax2.set_ylabel('All-token Routing Entropy')
    ax2.set_title('DeepSeek R1\nAll-Token RE', fontsize=13)

    ymax2 = max(a_at.max(), b_at.max(), c_at.max())
    span2 = ymax2 - min(a_at.min(), b_at.min(), c_at.min())
    g2 = span2 * 0.07

    bracket(ax2, 1, 2, ymax2 + g2*0.5, fmt_p(p_ab_at) + (' ***' if p_ab_at<0.001 else ''),
            h=span2*0.015, fs=8.5)
    bracket(ax2, 2, 3, ymax2 + g2*2.0, fmt_p(p_bc_at) + (' (ns)' if p_bc_at>=0.05 else ''),
            h=span2*0.015, fs=8.5)
    bracket(ax2, 1, 3, ymax2 + g2*3.5, fmt_p(p_ac_at) + (' (ns)' if p_ac_at>=0.05 else ''),
            h=span2*0.015, fs=8.5)
    ax2.set_ylim(top=ymax2 + g2*5.5)

    fig.text(0.5, -0.04,
             'R1 (RL-trained):  "your" is a stronger self-referential signal '
             'than "this" at last-token (C > A).\n'
             'Consistent with RL sensitizing the model to second-person address.',
             ha='center', fontsize=10.5, fontstyle='italic', color='#555')

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTDIR/'fig6_three_condition_r1.png')
    plt.close()
    print("  fig6_three_condition_r1.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Architecture Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig7():
    fig, ax = plt.subplots(figsize=(9, 6))

    mat = np.zeros((5, 2))
    pv  = np.zeros((5, 2))
    dirs = []

    for i, name in enumerate(MODEL_ORDER):
        d = DATA[name]
        lt_p, at_p = d['lt_p'], d['at_p']
        lt_dir = 1 if d['lt_pos'] > d['n']/2 else -1
        at_dir = 1 if d['at_pos'] > d['n']/2 else -1

        mat[i, 0] = -np.log10(lt_p)*lt_dir if lt_p < 0.05 else 0
        mat[i, 1] = -np.log10(at_p)*at_dir if at_p < 0.05 else 0
        pv[i] = [lt_p, at_p]

        lt_lbl = f"{d['lt_pos']}/{d['n']} A>B" if lt_p < 0.05 else 'ns'
        at_lbl = (f"{d['at_pos']}/{d['n']} " + ('A>B' if at_dir>0 else 'B>A')
                  if at_p < 0.05 else 'ns')
        dirs.append([lt_lbl, at_lbl])

    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(mat.min()), abs(mat.max()), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(mat, cmap='RdBu', norm=norm, aspect=0.6)

    for i in range(5):
        for j in range(2):
            p = pv[i, j]
            sig = p < 0.05
            lbl = dirs[i][j]
            if sig:
                txt = f'{lbl}\n{fmt_p(p)}'
                col = 'white' if abs(mat[i,j]) > vmax*0.45 else '#222'
                wt  = 'bold'
            else:
                txt = f'ns\n{fmt_p(p)}'
                col = '#888'
                wt  = 'normal'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                    color=col, fontweight=wt)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Last-token RE', 'All-token RE'],
                       fontsize=12, fontweight='bold')
    ax.set_yticks(range(5))
    ax.set_yticklabels([DATA[n]['label'] for n in MODEL_ORDER], fontsize=10)
    ax.tick_params(length=0)  # hide tick marks

    # org annotations
    for i, name in enumerate(MODEL_ORDER):
        ax.text(2.15, i, DATA[name]['org'], va='center', fontsize=9,
                color='#777', fontstyle='italic',
                transform=ax.get_yaxis_transform())

    ax.set_title('Where the Self-Referential Routing Signal Appears\n'
                 'Blue = A > B significant  ·  Red = B > A significant  ·  White = null',
                 fontsize=13, pad=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.12,
                         label=r'$-\log_{10}(p) \times$ direction')
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR/'fig7_architecture_pattern.png')
    plt.close()
    print("  fig7_architecture_pattern.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Positional Confound
# ══════════════════════════════════════════════════════════════════════════════

def fig8():
    r = load_json(DATADIR/'positional-confound'/'results_168q_ds31_prefill.json')
    pp = r['per_prompt']
    tok = np.array([p['n_prompt_tokens'] for p in pp])
    at  = np.array([p['prefill_re'] for p in pp])
    lt  = np.array([p['last_token_re'] for p in pp])

    def lvl(p):
        v = p.get('level','')
        if isinstance(v, str) and v.startswith('L'):
            return int(v[1:])
        if isinstance(v, (int, float)):
            return int(v)
        return 0
    levels = np.array([lvl(p) for p in pp])
    valid = levels > 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: all-token RE vs token count
    ax = axes[0]
    rho_t, _ = spearmanr(tok, at)
    sc = ax.scatter(tok, at, c=levels, cmap='viridis', s=28, alpha=0.7,
                    edgecolors='white', linewidths=0.3)
    z = np.polyfit(tok, at, 1)
    xf = np.linspace(tok.min(), tok.max(), 100)
    ax.plot(xf, np.polyval(z, xf), color='#E74C3C', ls='--', lw=2, alpha=0.7)
    ax.set_xlabel('Token Count')
    ax.set_ylabel('All-token Routing Entropy')
    ax.set_title(f'All-token RE vs Token Count\nρ = {rho_t:.2f}',
                 color='#B2182B')
    plt.colorbar(sc, ax=ax, label='Complexity Level', shrink=0.8)

    # Panel 2: all-token RE vs level (confounded)
    ax = axes[1]
    rho_l, _ = spearmanr(levels[valid], at[valid])
    lmeans, lstds = {}, {}
    for l in sorted(set(levels)):
        if l == 0: continue
        m = levels == l
        lmeans[l], lstds[l] = at[m].mean(), at[m].std()
    ls_ = sorted(lmeans)
    ax.errorbar(ls_, [lmeans[l] for l in ls_], yerr=[lstds[l] for l in ls_],
                fmt='o-', color='#B2182B', ms=6, lw=2, capsize=3, alpha=0.8)
    ax.set_xlabel('Complexity Level')
    ax.set_ylabel('All-token Routing Entropy')
    ax.set_title(f'All-token RE vs Level\nρ = {rho_l:.2f}  (CONFOUNDED)',
                 color='#B2182B')
    ax.text(0.5, 0.06, 'Driven by token count\nnot complexity',
            transform=ax.transAxes, ha='center', fontsize=9.5, fontstyle='italic',
            color='#B2182B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FADBD8', alpha=0.8, ec='none'))

    # Panel 3: last-token RE vs level (null)
    ax = axes[2]
    rho_lt, _ = spearmanr(levels[valid], lt[valid])
    ltm, lts = {}, {}
    for l in sorted(set(levels)):
        if l == 0: continue
        m = levels == l
        ltm[l], lts[l] = lt[m].mean(), lt[m].std()
    ax.errorbar(ls_, [ltm[l] for l in ls_], yerr=[lts[l] for l in ls_],
                fmt='o-', color=PAL['null'], ms=6, lw=2, capsize=3, alpha=0.8)
    ax.set_xlabel('Complexity Level')
    ax.set_ylabel('Last-token Routing Entropy')
    ax.set_title(f'Last-token RE vs Level\nρ = {rho_lt:.2f}  (null after control)',
                 color='#888')
    ax.text(0.5, 0.06, 'No complexity effect\nonce position controlled',
            transform=ax.transAxes, ha='center', fontsize=9.5, fontstyle='italic',
            color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='#EAECEE', alpha=0.8, ec='none'))

    fig.suptitle('The Positional Confound That Killed the Hierarchy',
                 fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUTDIR/'fig8_positional_confound.png')
    plt.close()
    print("  fig8_positional_confound.png")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nOutput → {OUTDIR}\n")
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        fn()
    # clean stale files
    expected = {f'fig{i}' for i in range(1,9)}
    for f in OUTDIR.glob('*.png'):
        if not any(f.stem.startswith(e) for e in expected):
            f.unlink()
            print(f"  (removed stale {f.name})")
    print(f"\nDone — {len(list(OUTDIR.glob('*.png')))} figures")
