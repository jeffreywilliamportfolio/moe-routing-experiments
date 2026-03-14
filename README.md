# MoE Routing Entropy Experiments: Ling-1T

Measuring how a trillion-parameter mixture-of-experts model routes text through its expert network when the only variable is a single determiner word.

## What this is

We fed 510 prompts to **Ling-1T** (inclusionAI, 1T parameters, 256 experts, top-8 sigmoid routing) and captured which 8 of 256 specialist modules the model selected for every token at every layer. The prompts were identical except for one word: the determiner before the noun.

- **"this dog"** vs **"a dog"** vs **"your dog"** vs **"the dog"** vs **"their dog"**
- **"this cat"** vs **"a cat"** vs **"your cat"** vs **"the cat"** vs **"their cat"**
- **"this system"** vs **"a system"** vs **"your system"** vs **"the system"** vs **"their system"**

The system prompts are self-referential -- they describe the model's own routing process. The dog and cat prompts are controls.

## Results

### The determiner changes routing

Changing one word -- "this" to "a" -- changes which experts the model selects. This is statistically significant across all three content domains:

| Content | "this > a" all-token | p-value | Pair consistency |
|---------|---------------------|---------|-----------------|
| Dog | +0.000573 | 8.3e-07 | 26/30 |
| Cat | +0.000876 | 1.9e-09 | 30/30 |
| System | +0.001178 | 4.7e-08 | 28/30 |

### Self-referential content amplifies the effect

The "this vs a" routing difference is **2x larger** when the text describes the model's own processing (system) compared to animal behavior (dog). The model doesn't just process self-referential text -- it routes it differently.

### "Your" is special

"Your X" produces the highest last-token routing entropy across all content types. The 2nd-person possessive creates measurable routing uncertainty at the position where the model must commit to a response.

| Content | "your > a" last-token | p-value |
|---------|----------------------|---------|
| Dog | +0.001897 | 3.2e-03 |
| Cat | +0.001593 | 2.2e-02 |
| System | +0.003006 (your > the) | 1.2e-03 |

### Expert decomposition

Because Ling-1T uses sigmoid gating (not softmax), we can identify exactly which experts fire for each condition:

| Expert class | Count | Description |
|-------------|-------|-------------|
| Universal addressivity | 2 | E95, E185 -- fire more for "your" regardless of noun |
| Animal-only addressivity | 4 | E60, E70, E188, E208 -- "your" effect in dog/cat only |
| System-only addressivity | 15 | Fire more for "your" only when noun is "system" |
| Universal deictic | 1 | E117 -- responds to "this" across all content |
| System-only deictic | 11 | Respond to "this" only for self-referential content |

The self-referential signal recruits **7.5x more dedicated experts** than the pure linguistic addressivity signal.

### Per-layer divergence

Expert set Jaccard distance between "your X" and "a X" at each layer:

| Domain | Peak divergence | Lowest divergence | Peak layers |
|--------|----------------|-------------------|-------------|
| Dog | 0.149 | 0.015 | L50-L75 |
| System | 0.265 | 0.120 | L49-L70 |

System content routes to different experts **1.8x more** than dog content. Even at the earliest MoE layers (L4-L5), "your system" and "a system" differ by 12% in expert selection -- 8x the 1.5% divergence seen for "your dog" vs "a dog."

## Model

| Property | Value |
|----------|-------|
| Model | Ling-1T (inclusionAI/Ling-1T) |
| Architecture | BailingMoeV2 |
| Parameters | 1T total, ~50B active/token |
| Experts | 256 routed + 1 shared |
| Active experts/token | 8 (top-k) |
| Gating | Sigmoid (not softmax) |
| MoE layers | 76 (L4-L79, L79 excluded) |
| Quantization | Q3_K_S (9 shards, 402GB) |
| Inference | Prefill-only, greedy argmax, cold cache |
| Hardware | 8x NVIDIA H200 |

## Experiments

| Folder | Prompts | Content | Design |
|--------|---------|---------|--------|
| `ling1t-selfref-paired-1/` | 60 | Self-referential (this/a system) | A/B paired, Cal-Manip-Cal |
| `ling1t-5cond-dog-1/` | 150 | Dog behavior | 5-condition (this/a/your/the/their), Cal-Manip-Cal |
| `ling1t-5cond-cat-1/` | 150 | Cat behavior | 5-condition, Cal-Manip-Cal |
| `ling1t-5cond-1/` | 150 | Self-referential system | 5-condition, Cal-Manip-Cal |

Each experiment folder contains:
- `prompt_suite.json` -- prompt definitions
- `generate_tsv.py` -- wraps prompts in Ling-1T chat template
- `run_experiment.py` -- orchestrates capture on GPU instance
- `analyze_local.py` -- computes entropy from raw router tensors
- `results_*.json` -- full per-prompt, per-layer results
- `experiment.log` -- raw capture output (ground truth)
- `RESULTS.md` -- human-readable results

Expert decomposition analysis: `ling1t-5cond-dog-1/expert_decomposition.py` and `RESULTS-EXPERTS.md`.

## Methodology

**Cal-Manip-Cal sandwich**: Each prompt is structured as [calibration paragraph] + [manipulation paragraph] + [calibration paragraph]. The calibration paragraph is identical across all prompts. Only the manipulation paragraph varies, and only in the determiner word(s).

**Token matching**: All 5 conditions within each pair produce identical token counts on Ling-1T's tokenizer (verified). The only variable is the determiner.

**Entropy computation**: Sigmoid routing requires special handling. Raw logits are passed through sigmoid, then a top-8 mask zeros non-selected experts, then the 8 remaining scores are normalized to sum to 1. Shannon entropy is computed on this masked distribution and normalized by log2(8).

**Capture binary**: Modified llama.cpp (`capture_activations.cpp`) intercepts `ffn_moe_logits` tensors at each MoE layer during prefill. No generation tokens are produced (`n_predict=0`).

## Reproducing

```bash
# On a multi-GPU instance with Ling-1T GGUF loaded:
cd experiments/ling1t-5cond-dog-1/
python3 generate_tsv.py
python3 run_experiment.py 2>&1 | tee experiment.log
python3 analyze_local.py --output-dir output/ --prompt-suite prompt_suite.json
```

The `setup_instance.sh` script automates the full pipeline: model download, binary compilation, and experiment execution.
