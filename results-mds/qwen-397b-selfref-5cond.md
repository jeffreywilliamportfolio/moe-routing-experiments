# qwen_selfref_5cond_1

## Model

- **Model**: Qwen3.5-397B-A17B-UD-IQ3_XXS
- **Architecture**: qwen35moe
- **Experts**: 512 total, 10 active/token
- **MoE Layers**: 60
- **Chat Template**: `<|im_start|>user {prompt}<|im_end|> <|im_start|>assistant`

## Infrastructure

- **GPU**: 2x NVIDIA H200 (143GB each), Ubuntu 24.04.3, driver 580.126.09, CUDA 12.9
- **Binary**: capture_activations (md5: `59a5f9952194536747229e033fc93ca5`)
- **Source**: capture_activations.cpp (md5: `490ce890ecdef148c4a92031df6df0f2`)
- **llama.cpp**: commit `f75c4e8bf52ea480ece07fd3d9a292f1d7f04bc5 (b8123)`

## Inference Parameters

- **n_predict**: 0
- **ngl**: 999
- **ctx**: 16384
- **flash_attn**: True
- **cache_type_k**: q8_0
- **cache_type_v**: q8_0
- **sampling**: greedy_argmax
- **routing_only**: True

## Design

- **Design**: Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions, cold cache
- **A**: "this system"
- **B**: "a system"
- **C**: "your system"
- **D**: "the system"
- **E**: "their system"
- **Prompts**: 150
- **Rationale**: Discriminates definiteness (D=the) vs addressivity (E=their) vs possessive (C=your)

## Condition Means

| Condition | N | All-token RE | Last-token RE | Token Range |
|-----------|---|-------------|--------------|-------------|
| A | 30 | 0.901962 | 0.866851 | 331-362 |
| B | 30 | 0.901083 | 0.864840 | 331-363 |
| C | 30 | 0.899884 | 0.867792 | 331-362 |
| D | 30 | 0.901963 | 0.865962 | 331-362 |
| E | 30 | 0.901712 | 0.865680 | 331-362 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000878 | 1.8626e-09 | +0.002011 | 1.8626e-09 | 30/30 A>B |
| A vs C | +0.002078 | 1.8626e-09 | -0.000941 | 4.6011e-04 | 8/30 A>C |
| A vs D | -0.000002 | 8.0783e-01 | +0.000889 | 2.8326e-04 | 23/30 A>D |
| A vs E | +0.000250 | 7.9790e-04 | +0.001171 | 3.5390e-08 | 29/30 A>E |
| B vs C | +0.001199 | 1.8626e-09 | -0.002952 | 1.8626e-09 | 0/30 B>C |
| B vs D | -0.000880 | 1.8626e-09 | -0.001122 | 1.3039e-08 | 1/30 B>D |
| B vs E | -0.000629 | 3.9041e-05 | -0.000840 | 5.1446e-06 | 5/30 B>E |
| C vs D | -0.002080 | 1.8626e-09 | +0.001830 | 3.8557e-07 | 26/30 C>D |
| C vs E | -0.001828 | 1.8626e-09 | +0.002112 | 1.8626e-09 | 30/30 C>E |
| D vs E | +0.000252 | 1.2334e-04 | +0.000282 | 2.8009e-01 | 14/30 D>E |

## Token Matching

- **Matched**: 24/30 pairs
- **Mismatched**: 6 pairs (off by 1 token)

## Code

- `code/qwen-397b/run_5cond.py`
- `code/qwen-397b/generate_tsv_5cond.py`
- `code/qwen-397b/generate_suite_5cond.py`
- `code/qwen-397b/compare_r1_r2.py`

## Notes

Last-token RE ordering: C (your) > A (this) > D (the) ≈ E (their) > B (a).

- **Definiteness rejected**: D ≠ A for last-token (p=2.83e-04). "The" does not pattern like "this."
- **Addressivity confirmed**: C > E, 30/30 (p=1.86e-09). Possessive structure alone is insufficient — the 2nd-person addressee is critical.
- **D ≈ E on last-token**: p=0.280 (null). Definite article and 3rd-person possessive are indistinguishable.
- **Bit-exact replication**: r2 matches r1 on 135/135 overlapping prompts. 15 dropped due to GPU memory on batch 2 reload.
