# qwen_selfref_3cond_1

## Model

- **Model**: Qwen3.5-397B-A17B-UD-Q2_K_XL
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

- **Design**: Cal-Manip-Cal sandwich, 30 paired prompts x 3 conditions, cold cache
- **A**: "this system"
- **B**: "a system"
- **C**: "your system"
- **Prompts**: 90

## Condition Means

| Condition | N | All-token RE | Last-token RE | Token Range |
|-----------|---|-------------|--------------|-------------|
| A | 30 | 0.901962 | 0.866851 | 331-362 |
| B | 30 | 0.901083 | 0.864840 | 331-363 |
| C | 30 | 0.899884 | 0.867792 | 331-362 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000878 | 1.8626e-09 | +0.002011 | 1.8626e-09 | 30/30 A>B |
| A vs C | +0.002078 | 1.8626e-09 | -0.000941 | 4.6011e-04 | 8/30 A>C |
| B vs C | +0.001199 | 1.8626e-09 | -0.002952 | 1.8626e-09 | 0/30 B>C |

## Token Matching

- **Matched**: 24/30 pairs
- **Mismatched**: 6 pairs (off by 1 token)

## Code

- `code/qwen-397b/run_3cond.py`
- `code/qwen-397b/generate_tsv_3cond.py`

## Notes

All-token RE: A > C > B. Last-token RE: C > A > B. The dissociation between all-token and last-token metrics for condition C ("your system") motivated the 5-condition follow-up.
