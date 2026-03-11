# glm5_selfref_3cond_1

## Model

- **Model**: GLM-5 UD-Q2_K_XL (745B MoE, 44B active)
- **Architecture**: glm_moe_dsa
- **Experts**: 256 total, 8 active/token
- **MoE Layers**: 75
- **Chat Template**: `[gMASK]<sop><|user|>{prompt}<|assistant|>`

## Infrastructure

- **GPU**: 2x NVIDIA H200 (143GB each), Ubuntu 24.04, driver 580.126.09
- **Binary**: capture_activations (md5: `59a5f9952194536747229e033fc93ca5`)
- **Source**: capture_activations.cpp (md5: `490ce890ecdef148c4a92031df6df0f2`)
- **llama.cpp**: commit `f75c4e8bf52ea480ece07fd3d9a292f1d7f04bc5 (b8123)`

## Inference Parameters

- **n_predict**: 0
- **ngl**: 999
- **ctx**: 2048
- **flash_attn**: off
- **cache_type_k**: f16
- **cache_type_v**: f16
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
| A | 30 | 0.899699 | 0.857583 | 325-357 |
| B | 30 | 0.900752 | 0.851175 | 325-357 |
| C | 30 | 0.899275 | 0.856497 | 325-357 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | -0.001053 | 4.4076e-05 | +0.006408 | 4.6011e-04 | 23/30 A>B |
| A vs C | +0.000424 | 6.9893e-02 | +0.001087 | 4.0449e-01 | 17/30 A>C |
| B vs C | +0.001477 | 6.9179e-06 | -0.005322 | 7.2960e-04 | 6/30 B>C |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/glm5/run_experiment.py`
- `code/glm5/generate_tsv.py`
