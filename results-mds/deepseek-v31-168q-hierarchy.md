# ds31_168q_hierarchy

## Model

- **Model**: DeepSeek-V3.1 UD-Q2_K_XL
- **Architecture**: deepseek2
- **Experts**: 256 total, 8 active/token
- **MoE Layers**: 58

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

- **Design**: 168-prompt cognitive complexity hierarchy (L1-L12)
- **Prompts**: 168
- **Purpose**: Tests whether routing entropy correlates with prompt complexity level

## Per-Level Means

| Level | Name | N | All-token RE | Last-token RE | Token Range |
|-------|------|---|-------------|--------------|-------------|
| L1 | Rote repetition | 14 | 0.837028 | 0.869591 | 17-93 |
| L2 | Factual recall | 14 | 0.829977 | 0.844566 | 17-30 |
| L3 | Logical reasoning | 14 | 0.833659 | 0.836919 | 26-93 |
| L4 | Cross-domain analogy | 14 | 0.843093 | 0.853888 | 32-49 |
| L5 | Theory of mind | 14 | 0.865266 | 0.846452 | 58-86 |
| L6 | Ethical dilemma | 14 | 0.855243 | 0.857573 | 59-81 |
| L7 | Self-referential | 14 | 0.850803 | 0.842124 | 39-81 |
| L8 | Strange loops | 14 | 0.879876 | 0.860547 | 137-172 |
| L9 | Deep self-reference | 14 | 0.876095 | 0.847248 | 81-127 |
| L10 | Architectural introspection | 14 | 0.866580 | 0.854846 | 95-126 |
| L11 | Nexus-7 (3rd person) | 14 | 0.882910 | 0.854795 | 94-145 |
| L12 | Echo persona | 14 | 0.885707 | 0.847595 | 120-162 |

## Notes

This experiment demonstrates the positional confound: RE correlates with both complexity level AND token count. The correlation with token count (rho >= correlation with level) means the hierarchy results cannot be cleanly attributed to cognitive complexity alone. This motivated the switch to paired Cal-Manip-Cal designs with token-matched prompts.

## Code

- `code/deepseek-v31/run_168q_hierarchy.py`
