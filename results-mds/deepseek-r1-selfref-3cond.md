# r1_selfref_3cond_1

## Model

- **Model**: DeepSeek-R1 UD-Q2_K_XL (671B MoE, 256 experts, 8 active)
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
- **ngl**: 999
- **ctx**: 4096
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
| A | 30 | 0.888333 | 0.835800 | 332-364 |
| B | 30 | 0.887856 | 0.835252 | 332-364 |
| C | 30 | 0.888140 | 0.838073 | 332-364 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000477 | 1.0382e-03 | +0.000548 | 7.3034e-01 | 17/30 A>B |
| A vs C | +0.000193 | 6.6665e-02 | -0.002273 | 3.0985e-02 | 10/30 A>C |
| B vs C | -0.000284 | 5.7688e-02 | -0.002821 | 9.1950e-02 | 12/30 B>C |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/deepseek-r1/run_paired.py`
