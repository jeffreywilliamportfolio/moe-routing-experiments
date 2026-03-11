# qwen_strangeloop_paired_1

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

- **Design**: Cal-Manip-Cal sandwich, 30 paired prompts, cold cache
- **A**: "this system" (proximal deictic)
- **B**: "a system" (indefinite generic)
- **Prompts**: 60
- **Rationale**: Control for selfref-paired. Recursive content (Godel, Escher, bootstrap) but not about the model.

## Condition Means

| Condition | N | All-token RE | Last-token RE | Token Range |
|-----------|---|-------------|--------------|-------------|
| A | 30 | 0.899695 | 0.867111 | 356-374 |
| B | 30 | 0.898673 | 0.866778 | 356-374 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.001021 | 1.8626e-09 | +0.000333 | 3.4752e-03 | 24/30 A>B |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/qwen-397b/run_strangeloop.py`

## Notes

Significant on Qwen (p=1.86e-09 all-token, p=3.48e-03 last-token). Content is abstract recursion (Gödel, Escher, bootstrap paradoxes) — NOT about the model. Opposite of DS31 where strangeloop was null (p=0.685).
