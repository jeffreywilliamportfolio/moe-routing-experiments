# selfref_paired_1

## Model

- **Model**: Qwen3.5-397B-A17B-UD-Q2_K_XL
- **Architecture**: qwen35moe
- **Experts**: 512 total, 10 active/token
- **MoE Layers**: 60
- **Chat Template**: `?`

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

- **Design**: Cal-Manip-Cal sandwich, 30 paired prompts, cold cache
- **A**: "this system" (proximal deictic)
- **B**: "a system" (indefinite generic)
- **Prompts**: 60

## Condition Means

| Condition | N | All-token RE | Last-token RE | Token Range |
|-----------|---|-------------|--------------|-------------|
| A | 30 | 0.902203 | 0.872521 | 331-364 |
| B | 30 | 0.901421 | 0.871599 | 331-363 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000782 | 5.5879e-09 | +0.000922 | 8.8565e-05 | 25/30 A>B |

## Token Matching

- **Matched**: 24/30 pairs
- **Mismatched**: 6 pairs (off by 1 token)

## Code

- `code/qwen-397b/run_paired.py`
