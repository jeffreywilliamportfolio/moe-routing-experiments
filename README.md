# MoE Routing Distinguishes Self-Referential From Generic Content

Across five architectures, four organizations, and three training methodologies.

**Author:** Jeffrey William Shorthill

**Article:** [LessWrong post](https://www.lesswrong.com/) *(link TBD)*

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

## Repository Structure

```
├── README.md
├── article.md                  # Full article text
├── data/                       # Processed results (JSON)
│   ├── qwen-397b/              # Qwen 397B paired results
│   ├── glm5/                   # GLM-5 paired + three-condition
│   ├── deepseek-v31/           # DeepSeek V3.1 paired results
│   ├── deepseek-r1/            # DeepSeek R1 paired + three-condition
│   ├── gptoss-120b/            # gpt-oss-120b paired results
│   ├── strangeloop-control/    # "this paradox" vs "a paradox" (null control)
│   └── positional-confound/    # 168-prompt hierarchy showing token-count confound
├── prompts/                    # Prompt definitions (JSON)
│   ├── selfref_paired_30.json  # 30 A/B pairs used across most models
│   ├── selfref_3cond_glm5.json # Three-condition (this/a/your) for GLM-5
│   ├── selfref_3cond_r1.json   # Three-condition for DeepSeek R1
│   └── strangeloop_paired_30.json  # Strange loop control pairs
├── figures/                    # Publication figures (PNG)
│   ├── fig1_design_schematic.png
│   ├── fig2_five_model_replication.png
│   ├── fig3_per_pair_all_models.png
│   ├── fig4_selfref_vs_strangeloop.png
│   ├── fig5_three_condition_glm5.png
│   ├── fig6_three_condition_r1.png
│   ├── fig7_architecture_pattern.png
│   └── fig8_positional_confound.png
├── code/                       # All code used
│   ├── capture_activations.cpp # C++ binary source (llama.cpp b8123 fork)
│   ├── generate_figures.py     # Generates all 8 figures from data/
│   ├── run_experiment_*.py     # Per-model orchestrator scripts
│   ├── generate_tsv_*.py       # Prompt formatting per model
│   └── token_corrections_*.json # Token-matching corrections per model
└── logs/                       # Raw experiment logs (ground truth)
    ├── qwen_397b.log
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
- Qwen 397B: `Qwen3-MoE-UD-Q2_K_XL`
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
