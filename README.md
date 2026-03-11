# MoE Routing Distinguishes Self-Referential From Generic Content

Across five architectures, four organizations, and three training methodologies. All GPUs run on vast.ai (Nvidia H200 141GB VRAM. Total cost for all runs in repo: less than $600

**Author:** Jeffrey William Shorthill

**Article:** forthcoming

## What This Is

When a Mixture-of-Experts model processes text about itself ("this system"), its expert routing distributions look measurably different than when it processes identical text about a generic subject ("a system"). This effect replicates across five models from four organizations with three different training methodologies.

All experiments use greedy argmax inference, cold cache, prefill-only capture, and token-matched prompt pairs via a Cal-Manip-Cal (calibration-manipulation-calibration) design inspired by fMRI block design.

## Results Summary

| Model | Organization | Experts | Active | MoE Layers | Training | Last-token p | All-token p |
|-------|-------------|---------|--------|------------|----------|-------------|------------|
| Qwen 397B | Alibaba | 512 | 10 | 60 | Standard | 8.86e-5 | 5.6e-9 |
| GLM-5 | Zhipu AI | 256 | 8 | 75 | Standard | 4.6e-4 | 4.4e-5 |
| DeepSeek V3.1 | DeepSeek | 256 | 8 | 58 | Standard | 0.011 | null |
| DeepSeek R1 | DeepSeek | 256 | 8 | 58 | RL | null | 0.001 |
| gpt-oss-120b | OpenAI | 128 | 4 | 36 | Distilled | null | 0.021 |

All p-values are Wilcoxon signed-rank tests on 30 token-matched prompt pairs.

### 5-Condition Addressivity Experiment (Qwen 397B)

Extends from 2 to 5 conditions to discriminate competing hypotheses:

| Condition | Determiner | Last-token RE | Role |
|-----------|-----------|---------------|------|
| C "your system" | 2nd-person possessive | 0.867792 (highest) | Addressee-directed |
| A "this system" | Proximal deictic | 0.866851 | Deictic reference |
| D "the system" | Definite article | 0.865962 | Definiteness only |
| E "their system" | 3rd-person possessive | 0.865680 | Possessive, no addressee |
| B "a system" | Indefinite generic | 0.864840 (lowest) | Baseline |

Key findings (all 10 pairwise Wilcoxon tests):
- **Addressivity confirmed**: C > E, 30/30 pairs (p=1.86e-09). Possessive structure alone is insufficient — the 2nd-person addressee is the critical variable.
- **Definiteness rejected**: D ≠ A for last-token RE (p=2.83e-04). "The" does not pattern like "this."
- **D ≈ E on last-token**: p=0.280 (null). Definite article and 3rd-person possessive are indistinguishable.
- **Bit-exact replication**: r2 rerun matches r1 on 135/135 overlapping prompts (15 dropped to GPU memory, not data).

## Repository Structure

```
├── README.md
├── article.md                  # Full article text
├── data/                       # Processed results (JSON)
│   ├── qwen-397b/
│   │   ├── results_selfref_paired_prefill.json      # 2-condition (this/a)
│   │   ├── results_selfref_3cond_prefill.json       # 3-condition (this/a/your)
│   │   ├── results_selfref_5cond_prefill.json       # 5-condition (this/a/your/the/their)
│   │   ├── results_selfref_5cond_prefill_r2.json    # 5-condition replication run
│   │   └── results_strangeloop_paired_prefill.json   # Strange loop control on Qwen
│   ├── glm5/                   # GLM-5 paired + three-condition
│   ├── deepseek-v31/           # DeepSeek V3.1 paired results
│   ├── deepseek-r1/            # DeepSeek R1 paired + three-condition
│   ├── gptoss-120b/            # gpt-oss-120b paired results
│   ├── strangeloop-control/    # "this paradox" vs "a paradox" (DS31, null)
│   └── positional-confound/    # 168-prompt hierarchy showing token-count confound
├── prompts/                    # Prompt definitions (JSON)
│   ├── selfref_paired_30.json       # 30 A/B pairs used across most models
│   ├── selfref_3cond_glm5.json      # Three-condition (this/a/your) for GLM-5
│   ├── selfref_3cond_r1.json        # Three-condition for DeepSeek R1
│   ├── selfref_3cond_qwen.json      # Three-condition for Qwen 397B
│   ├── selfref_5cond_qwen.json      # Five-condition for Qwen 397B
│   └── strangeloop_paired_30.json   # Strange loop control pairs
├── figures/                    # Publication figures (PNG)
│   ├── fig1_design_schematic.png
│   ├── fig2_five_model_replication.png
│   ├── fig3_per_pair_all_models.png
│   ├── fig4_selfref_vs_strangeloop.png
│   ├── fig5_three_condition_glm5.png
│   ├── fig6_three_condition_r1.png
│   ├── fig7_architecture_pattern.png
│   └── fig8_positional_confound.png
├── code/                       # All code, organized by model
│   ├── shared/
│   │   ├── capture_activations.cpp      # C++ binary source (llama.cpp b8123 fork)
│   │   └── generate_figures.py          # Generates all 8 figures from data/
│   ├── qwen-397b/
│   │   ├── run_paired.py                # 2-condition (this/a)
│   │   ├── run_3cond.py                 # 3-condition (this/a/your)
│   │   ├── run_5cond.py                 # 5-condition (this/a/your/the/their)
│   │   ├── run_strangeloop.py           # Strange loop control
│   │   ├── generate_tsv_3cond.py
│   │   ├── generate_tsv_5cond.py
│   │   ├── generate_suite_5cond.py      # Derives 5-cond from 3-cond suite
│   │   └── compare_r1_r2.py            # Bit-exact replication verifier
│   ├── deepseek-v31/
│   │   ├── run_paired.py
│   │   ├── run_strangeloop.py
│   │   ├── run_168q_hierarchy.py        # 168-prompt complexity hierarchy
│   │   └── generate_tsv.py
│   ├── deepseek-r1/
│   │   ├── run_paired.py
│   │   └── token_corrections.json
│   ├── glm5/
│   │   ├── run_experiment.py
│   │   ├── generate_tsv.py
│   │   └── token_corrections.json
│   └── gptoss-120b/
│       ├── run_experiment.py
│       ├── generate_tsv.py
│       └── token_corrections.json
└── logs/                       # Raw experiment logs (ground truth)
    ├── qwen-397b/
    │   ├── selfref_3cond.log            # 90-prompt 3-condition
    │   ├── selfref_5cond.log            # 150-prompt 5-condition
    │   ├── selfref_5cond_r2.log         # 5-condition replication
    │   └── strangeloop_paired.log       # Strange loop control
    ├── qwen_397b.log                    # Original paired run
    ├── glm5.log
    ├── deepseek_v31.log
    ├── deepseek_r1.log
    ├── gptoss_120b.log
    ├── strangeloop_control.log
    └── positional_confound_168q.log
```

## How It Works

### Capture (C++)

`code/capture_activations.cpp` is a fork of llama.cpp b8123 that intercepts `ffn_moe_logits` tensors during inference. For each prompt, it saves one `[n_tokens, n_experts]` float32 array per MoE layer as `.npy` files.

Key flags:
- `--routing-only` — captures only router logits (not SwiGLU gates or expert projections)
- `-n 0` — prefill-only (no generation), eliminating token-count confound
- Greedy argmax throughout, cold KV cache between prompts

### Compute (Python)

Each `run_experiment_*.py` script:
1. Invokes the binary with model-specific paths and chat template
2. Loads `.npy` files, applies softmax to get routing probabilities
3. Computes normalized Shannon entropy: `RE = -sum(p * log2(p)) / log2(n_experts)`
4. Runs Wilcoxon signed-rank tests on paired A-B differences
5. Saves per-prompt results with per-layer detail to JSON

### Prompt Design (Cal-Manip-Cal)

Each prompt has three segments:
1. **Calibration** — identical paragraph about transformer routing (same across all prompts)
2. **Manipulation** — experimental content differing by one word ("this system" vs "a system")
3. **Calibration** — same paragraph again

Token counts are verified to match exactly between conditions for every pair. Mismatches are corrected with single-token padding (model-specific; see `token_corrections_*.json`).

## Infrastructure

All experiments ran on rented NVIDIA H200 GPUs (Vast.ai) using quantized GGUF models:
- DeepSeek V3.1: `DeepSeek-V3-0324-UD-Q2_K_XL` (6 shards, 231GB)
- DeepSeek R1: `DeepSeek-R1-UD-Q2_K_XL` (5 shards, 212GB)
- Qwen 397B: `Qwen3.5-397B-A17B-UD-IQ3_XXS` (4 shards, 131GB)
- GLM-5: `GLM-4.7-UD-Q2_K_XL` (3 shards, 128GB)
- gpt-oss-120b: `gpt-oss-120b-UD-Q2_K_XL`

All models used `-ngl 30 -c 4096 -t 16 --routing-only`.

## Reproducing

1. Build `capture_activations.cpp` against llama.cpp b8123
2. Download a GGUF-quantized MoE model
3. Format prompts with the appropriate chat template (`generate_tsv_*.py`)
4. Run the experiment script (`run_experiment_*.py`), adjusting paths
5. Generate figures: `python3 code/generate_figures.py`

Dependencies: Python 3.10+, numpy, scipy, matplotlib, seaborn.

## Verification

All results were verified against raw experiment logs (ground truth) using automated cross-referencing:

| Experiment | Prompts | Checks | Failures |
|-----------|---------|--------|----------|
| Qwen 3-condition | 90 | 307 | 0 |
| Qwen strangeloop | 60 | 382 | 0 |
| Qwen 5-condition | 150 | 556 | 0 |
| Cross-experiment (3-cond vs 5-cond) | 90 | 90 | 0 |
| **Total** | | **1,335** | **0** |

Checks include: per-prompt RE and last-token RE against log values, token counts, all Wilcoxon W and p-values recomputed from raw data, per-category breakdowns, condition means, per-layer detail spot checks (deltas at 1e-16 floating point floor), and cross-experiment identity verification (90 shared prompts between 3-cond and 5-cond are bit-identical).

Determinism confirmed: 5-condition r2 rerun matches r1 on 135/135 overlapping prompts (bit-exact, greedy argmax).

Binary provenance: single `capture_activations.cpp` source (md5: `490ce890ecdef148c4a92031df6df0f2`) compiled to one binary (md5: `59a5f9952194536747229e033fc93ca5`) used across all experiments. Built against llama.cpp commit `f75c4e8` (b8123).

## Data Format

Each results JSON contains a `per_prompt` array where each entry has:
- `id` — prompt identifier (e.g., `BASIC_01_A`)
- `condition` — `A` ("this system"), `B` ("a system"), or `C` ("your system")
- `pair` — pair number (1-30)
- `category` — self-reference category (e.g., `basic_selfref`, `deep_selfref`)
- `n_prompt_tokens` — verified token count
- `prefill_re` — all-token mean routing entropy
- `last_token_re` — last-token routing entropy
- `per_layer` — array of 58-89 objects with layer-level detail (mean, std, min, max, coalition_strength)

## License

MIT
