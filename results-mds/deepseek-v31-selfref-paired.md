# ds31_selfref_paired_1

## Model

- **Model**: DeepSeek-V3.1 UD-Q2_K_XL
- **Architecture**: deepseek2
- **Experts**: 256 total, 8 active/token
- **MoE Layers**: 58
- **Chat Template**: `<｜User｜>{prompt}<｜Assistant｜>`

## Infrastructure

- **GPU**: 1x NVIDIA H200 (143GB), Ubuntu 22.04, driver 565.57.01
- **Binary**: capture_activations (md5: `59a5f9952194536747229e033fc93ca5`)
- **Source**: capture_activations.cpp (md5: `490ce890ecdef148c4a92031df6df0f2`)
- **llama.cpp**: commit `f75c4e8bf52ea480ece07fd3d9a292f1d7f04bc5 (b8123)`

## Inference Parameters

- **n_predict**: 0
- **ngl**: 30
- **ctx**: 4096
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
| A | 30 | 0.895755 | 0.859592 | 332-364 |
| B | 30 | 0.895631 | 0.857840 | 332-364 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000124 | 5.8376e-01 | +0.001751 | 1.1303e-02 | 22/30 A>B |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/deepseek-v31/run_paired.py`
- `code/deepseek-v31/generate_tsv.py`
