# gptoss_selfref_paired_1

## Model

- **Model**: GPT-OSS-120B mxfp4
- **Architecture**: gpt-oss
- **Experts**: 128 total, 4 active/token
- **MoE Layers**: 36
- **Chat Template**: `<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>`

## Infrastructure

- **GPU**: 2x NVIDIA H200 (143GB each), Ubuntu 24.04, driver 580.126.09
- **Binary**: capture_activations (md5: `59a5f9952194536747229e033fc93ca5`)
- **Source**: capture_activations.cpp (md5: `490ce890ecdef148c4a92031df6df0f2`)
- **llama.cpp**: commit `f75c4e8bf52ea480ece07fd3d9a292f1d7f04bc5 (b8123)`

## Inference Parameters

- **n_predict**: 0
- **ngl**: 999
- **ctx**: 4096
- **flash_attn**: off
- **cache_type_k**: f16
- **cache_type_v**: f16
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
| A | 30 | 0.952254 | 0.938323 | 320-352 |
| B | 30 | 0.952134 | 0.938379 | 320-352 |

## Pairwise Comparisons (Wilcoxon signed-rank)

| Comparison | All-tok mean diff | All-tok p | Last-tok mean diff | Last-tok p | Direction (LT) |
|-----------|------------------|----------|-------------------|----------|---------------|
| A vs B | +0.000120 | 2.0850e-02 | -0.000056 | 6.7018e-01 | 11/30 A>B |

## Token Matching

- **Matched**: 30/30 pairs

## Code

- `code/gptoss-120b/run_experiment.py`
- `code/gptoss-120b/generate_tsv.py`
