# ds31_strangeloop_paired_1

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
- **Rationale**: Control for selfref-paired. Recursive content (Godel, Escher, bootstrap) but not about the model.

## Condition Means

| Condition | N | All-token RE | Last-token RE | Token Range |
|-----------|---|-------------|--------------|-------------|
| A | 30 | 0.888722 | 0.864797 | 355-372 |
| B | 30 | 0.888593 | 0.864468 | 355-372 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000129 | 4.7711e-01 | +0.000329 | 6.8505e-01 | 14/30 A>B |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/deepseek-v31/run_strangeloop.py`

## Notes

Null result on DS31 — abstract recursion does NOT trigger the self-referential routing shift.
