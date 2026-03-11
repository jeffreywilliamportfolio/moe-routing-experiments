# Qwen 397B MoE Routing Experiments — 2026-03-10

Routing entropy measurements on Qwen3.5-397B-A17B (512 experts, top-10, 60 MoE layers).

## Experiments

| Experiment | Prompts | Conditions | Key Result |
|-----------|---------|------------|------------|
| selfref-3cond | 90 | A=this, B=a, C=your | Last-token: your > this > a (C>A p=4.60e-04) |
| selfref-5cond | 150 | A=this, B=a, C=your, D=the, E=their | Addressivity confirmed, definiteness rejected |
| strangeloop-paired | 60 | A=this, B=a (abstract recursion) | Significant on Qwen (p=3.48e-03 LT) |

### 5-Condition Last-Token RE Ordering

```
C "your system"   0.867792  ← highest (2nd-person possessive)
A "this system"   0.866851  (proximal deictic)
D "the system"    0.865962  (definite article)
E "their system"  0.865680  (3rd-person possessive)
B "a system"      0.864840  ← lowest (indefinite generic)
```

- **Addressivity confirmed**: C > E, 30/30 (p=1.86e-09). Possessive alone insufficient — requires 2nd-person addressee.
- **Definiteness rejected**: D ≠ A for last-token (p=2.83e-04). "The" does not pattern like "this."
- **D ≈ E null**: p=0.280. Definite article and 3rd-person possessive are indistinguishable.

## Model & Infrastructure

- **Model**: Qwen3.5-397B-A17B-UD-IQ3_XXS (4 shards, 131GB)
- **Architecture**: qwen35moe, 512 experts, 10 active/token, 60 MoE layers
- **GPU**: 2x NVIDIA H200 (143GB each), Ubuntu 24.04.3, driver 580.126.09
- **Binary**: capture_activations (md5: `59a5f9952194536747229e033fc93ca5`)
- **Source**: capture_activations.cpp (md5: `490ce890ecdef148c4a92031df6df0f2`)
- **llama.cpp**: commit `f75c4e8bf52ea480ece07fd3d9a292f1d7f04bc5` (b8123)
- **Inference**: `-n 0 -ngl 999 -c 16384 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --routing-only`
- **Sampling**: greedy argmax, cold KV cache between prompts
- **Chat template**: `<|im_start|>user {text}<|im_end|> <|im_start|>assistant`

## Verification

- 1,335 automated checks across all experiments: **0 failures**
- Per-prompt RE and last-token RE verified against experiment logs (ground truth)
- All Wilcoxon W and p-values recomputed from raw data
- 5-condition r2 rerun: **135/135 bit-exact match** (greedy argmax determinism confirmed)
- Cross-experiment: 90 shared prompts between 3-cond and 5-cond are bit-identical

## Files

```
├── data/qwen-397b/
│   ├── results_selfref_3cond_prefill.json
│   ├── results_selfref_5cond_prefill.json
│   ├── results_selfref_5cond_prefill_r2.json    # replication run
│   └── results_strangeloop_paired_prefill.json
├── prompts/
│   ├── selfref_3cond_qwen.json
│   └── selfref_5cond_qwen.json
├── code/
│   ├── shared/
│   │   └── capture_activations.cpp
│   └── qwen-397b/
│       ├── run_3cond.py
│       ├── run_5cond.py
│       ├── run_strangeloop.py
│       ├── generate_tsv_3cond.py
│       ├── generate_tsv_5cond.py
│       ├── generate_suite_5cond.py
│       └── compare_r1_r2.py
├── logs/qwen-397b/
│   ├── selfref_3cond.log
│   ├── selfref_5cond.log
│   ├── selfref_5cond_r2.log
│   └── strangeloop_paired.log
└── results-mds/
    ├── qwen-397b-selfref-3cond.md
    ├── qwen-397b-selfref-5cond.md
    └── qwen-397b-strangeloop-paired.md
```

## Reproducing

1. Build `capture_activations.cpp` against llama.cpp b8123
2. Download `Qwen3.5-397B-A17B-UD-IQ3_XXS` GGUF from HuggingFace
3. Run `python3 run_5cond.py` (adjusting model path)
4. Compare output against `results_selfref_5cond_prefill.json`

Dependencies: Python 3.10+, numpy, scipy.
