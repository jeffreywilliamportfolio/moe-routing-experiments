# ling1t-selfref-paired-1 Results

## Run Info

- **Model**: Ling-1T Q3_K_S (inclusionAI/Ling-1T, BailingMoeV2, ~1T params, ~50B active)
- **Architecture**: `bailingmoe2` — sigmoid gating, top-8 of 256 experts, 1 shared expert
- **MoE layers**: 76 captured (layers 4-79), 70 used after exclusions (layer 79 auto-excluded; occasional corrupt .npy layers per prompt)
- **Gating function**: Sigmoid (NOT softmax) — top-8 masked, normalized to simplex
- **Entropy normalization**: `H / log2(8)` — normalized by active expert count, range [0,1]. NOT comparable to softmax-routed models.
- **Design**: Cal-Manip-Cal sandwich, 30 paired prompts, cold cache
- **Prompts**: Same `prompt_suite.json` as all selfref-paired experiments
- **Condition A**: "this system" (self-referential)
- **Condition B**: "a system" (generic control)
- **Inference**: prefill-only (`-n 0`), `ngl=999`, `ctx=4096`, greedy argmax, `--routing-only`
- **Instance**: 8x NVIDIA H200 (1145 GB total VRAM), Vast.ai
- **GPU**: All 80 layers offloaded, flash attention enabled, pipeline parallelism
- **Token matching**: 24/30 pairs exact match, 6 pairs differ by 1 token
- **Chat template**: `<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>`
- **llama.cpp branch**: `cisc/bailingmoe2` (BailingMoeV2 support)
- **llama.cpp commit**: `57819b8d4b39d893408e51520dff3d47d1ebb757`
- **Binary MD5**: `159ea30cfe106087c4938b7a4fea766f`
- **Date**: 2026-03-13

## Data Recovery Note

7 of 60 prompt directories were missing `metadata.txt` (P15A, P27B, P28A, P28B, P29A, P30A, P30B — all in batches 3-4). The .npy router tensors were intact. Token counts were inferred from the median row count across each prompt's .npy files. All 7 inferred values match `experiment.log` exactly:

| Prompt | Log tokens | Inferred |
|--------|-----------|----------|
| P15A_paradox | 367 | 367 |
| P27B_metacognitive | 375 | 375 |
| P28A_metacognitive | 379 | 379 |
| P28B_metacognitive | 379 | 379 |
| P29A_metacognitive | 378 | 378 |
| P30A_metacognitive | 384 | 384 |
| P30B_metacognitive | 384 | 384 |

## Primary Endpoint: All-Token Routing Entropy

| Metric | Value |
|--------|-------|
| Mean A-B diff | **+0.001087** |
| Std | 0.000976 |
| A > B | **27/30 (90%)** |
| Wilcoxon W | 34 |
| **Wilcoxon p** | **6.92e-06** |

"This system" prompts produce higher all-token routing entropy than "a system" controls. Highly significant.

## Last-Token Routing Entropy

| Metric | Value |
|--------|-------|
| Mean A-B diff | -0.001400 |
| Std | 0.003879 |
| A > B | 13/30 (43%) |
| Wilcoxon W | 152 |
| Wilcoxon p | 0.100 |

No significant difference at the last token position. Effect is distributed across all tokens, not concentrated at the final position.

## Per-Category Breakdown

### All-Token RE (A-B)

| Category | n | Mean diff | Std | A > B |
|----------|---|-----------|-----|-------|
| basic_selfref | 6 | +0.001525 | 0.000470 | 6/6 |
| deep_selfref | 6 | +0.000828 | 0.000479 | 6/6 |
| paradox | 6 | +0.001525 | 0.000713 | 6/6 |
| introspection | 6 | +0.001004 | 0.000897 | 5/6 |
| metacognitive | 6 | +0.000551 | 0.001501 | 4/6 |

All 5 categories show A > B in all-token RE. basic_selfref and paradox show strongest effects.

### Last-Token RE (A-B)

| Category | n | Mean diff | Std |
|----------|---|-----------|-----|
| basic_selfref | 6 | -0.002246 | 0.004501 |
| deep_selfref | 6 | -0.001386 | 0.002978 |
| introspection | 6 | -0.003364 | 0.003462 |
| metacognitive | 6 | -0.000562 | 0.003659 |
| paradox | 6 | +0.000557 | 0.003412 |

Last-token RE trends B > A (opposite of all-token), except paradox. None individually significant.

## Per-Pair Detail (Routing Entropy)

| Pair | Category | A_tok | B_tok | A_RE | B_RE | A-B RE | A_LT | B_LT | A-B LT | Match |
|------|----------|-------|-------|------|------|--------|------|------|--------|-------|
| 1 | basic_selfref | 384 | 384 | 0.905566 | 0.903947 | +0.001619 | 0.920892 | 0.930769 | -0.009877 | OK |
| 2 | basic_selfref | 372 | 372 | 0.903993 | 0.902522 | +0.001471 | 0.927773 | 0.931354 | -0.003581 | OK |
| 3 | basic_selfref | 368 | 367 | 0.904370 | 0.902269 | +0.002101 | 0.928644 | 0.924539 | +0.004104 | +1 |
| 4 | basic_selfref | 369 | 369 | 0.901409 | 0.900582 | +0.000827 | 0.930712 | 0.935640 | -0.004929 | OK |
| 5 | basic_selfref | 361 | 361 | 0.905729 | 0.903669 | +0.002060 | 0.930022 | 0.929588 | +0.000434 | OK |
| 6 | basic_selfref | 353 | 353 | 0.903471 | 0.902399 | +0.001072 | 0.926571 | 0.926199 | +0.000372 | OK |
| 7 | deep_selfref | 384 | 384 | 0.905170 | 0.904260 | +0.000910 | 0.933128 | 0.929823 | +0.003305 | OK |
| 8 | deep_selfref | 376 | 376 | 0.904241 | 0.902706 | +0.001534 | 0.930929 | 0.933263 | -0.002334 | OK |
| 9 | deep_selfref | 370 | 370 | 0.905449 | 0.904893 | +0.000556 | 0.930464 | 0.931655 | -0.001191 | OK |
| 10 | deep_selfref | 372 | 371 | 0.903058 | 0.902472 | +0.000586 | 0.931629 | 0.932922 | -0.001292 | +1 |
| 11 | deep_selfref | 360 | 360 | 0.904219 | 0.902934 | +0.001285 | 0.933155 | 0.933233 | -0.000078 | OK |
| 12 | deep_selfref | 381 | 381 | 0.903283 | 0.903184 | +0.000100 | 0.923339 | 0.930067 | -0.006728 | OK |
| 13 | paradox | 375 | 375 | 0.907190 | 0.906181 | +0.001009 | 0.925733 | 0.922623 | +0.003110 | OK |
| 14 | paradox | 382 | 382 | 0.904660 | 0.903824 | +0.000836 | 0.924369 | 0.923964 | +0.000405 | OK |
| 15 | paradox | 367 | 368 | 0.906392 | 0.903499 | +0.002893 | 0.924184 | 0.922365 | +0.001819 | -1 |
| 16 | paradox | 378 | 378 | 0.906771 | 0.905099 | +0.001672 | 0.928386 | 0.935152 | -0.006766 | OK |
| 17 | paradox | 372 | 372 | 0.906591 | 0.905649 | +0.000942 | 0.924584 | 0.921367 | +0.003217 | OK |
| 18 | paradox | 377 | 377 | 0.905871 | 0.904074 | +0.001797 | 0.923562 | 0.922006 | +0.001556 | OK |
| 19 | introspection | 367 | 367 | 0.903290 | 0.902442 | +0.000848 | 0.923549 | 0.930765 | -0.007216 | OK |
| 20 | introspection | 372 | 373 | 0.902574 | 0.903219 | -0.000645 | 0.924504 | 0.927817 | -0.003312 | -1 |
| 21 | introspection | 363 | 363 | 0.902481 | 0.901647 | +0.000833 | 0.930707 | 0.937665 | -0.006958 | OK |
| 22 | introspection | 371 | 371 | 0.901954 | 0.899783 | +0.002170 | 0.924756 | 0.929088 | -0.004332 | OK |
| 23 | introspection | 383 | 384 | 0.905807 | 0.904822 | +0.000985 | 0.926374 | 0.927587 | -0.001213 | -1 |
| 24 | introspection | 376 | 376 | 0.902785 | 0.900951 | +0.001834 | 0.932610 | 0.929761 | +0.002850 | OK |
| 25 | metacognitive | 370 | 370 | 0.903820 | 0.903680 | +0.000141 | 0.932215 | 0.929601 | +0.002613 | OK |
| 26 | metacognitive | 370 | 370 | 0.902231 | 0.902443 | -0.000212 | 0.924290 | 0.920065 | +0.004225 | OK |
| 27 | metacognitive | 375 | 375 | 0.904019 | 0.903169 | +0.000850 | 0.928680 | 0.927622 | +0.001058 | OK |
| 28 | metacognitive | 379 | 379 | 0.905375 | 0.903963 | +0.001413 | 0.934568 | 0.939787 | -0.005219 | OK |
| 29 | metacognitive | 378 | 379 | 0.904255 | 0.901254 | +0.003001 | 0.923297 | 0.928654 | -0.005357 | -1 |
| 30 | metacognitive | 384 | 384 | 0.902573 | 0.904459 | -0.001885 | 0.923898 | 0.924593 | -0.000695 | OK |

## Cross-Model Comparison

Same 30 prompt pairs across 6 models. All use Cal-Manip-Cal sandwich, cold cache, prefill-only.

### All-Token RE

| Model | Experts | Gating | Mean(A-B) | A>B | Wilcoxon p | Sig? |
|-------|---------|--------|-----------|-----|------------|------|
| **Ling-1T** | **256/8** | **sigmoid** | **+0.001087** | **27/30** | **6.92e-06** | **YES** |
| Qwen 397B | 128/8 | softmax | +0.000782 | 29/30 | 5.59e-09 | YES |
| GPT-OSS 120B | 128/8 | softmax | +0.000120 | 20/30 | 0.021 | YES |
| DeepSeek R1 | 256/8 | softmax | +0.000477 | 22/30 | 0.001 | YES |
| DeepSeek V3.1 | 256/8 | softmax | +0.000124 | 15/30 | 0.584 | no |
| GLM-5 | 160/8 | softmax | -0.001053 | 5/30 | 4.41e-05 | YES (B>A) |

### Last-Token RE

| Model | Mean(A-B) | A>B | Wilcoxon p | Sig? |
|-------|-----------|-----|------------|------|
| GLM-5 | +0.006408 | 23/30 | 4.60e-04 | YES |
| DeepSeek V3.1 | +0.001751 | 22/30 | 0.011 | YES |
| Qwen 397B | +0.000922 | 25/30 | 8.86e-05 | YES |
| DeepSeek R1 | +0.000548 | 17/30 | 0.730 | no |
| GPT-OSS 120B | -0.000056 | 11/30 | 0.670 | no |
| **Ling-1T** | **-0.001400** | **13/30** | **0.100** | **no** |

### Cross-Model Pattern

No model is significant on **both** all-token and last-token RE simultaneously (except Qwen, which has 6 token mismatches). The self-referential signal manifests differently across architectures:

- **All-token significant**: Ling-1T, Qwen, GPT-OSS, R1 — effect distributed across all token positions
- **Last-token significant**: GLM-5, DS V3.1, Qwen — effect concentrated at final position
- **GLM-5 reversal**: all-token B>A (p=4.4e-05) but last-token A>B (p=4.6e-04) — unique opposite-sign pattern

Ling-1T shows the strongest all-token effect magnitude (+0.001087) of any model tested, with 27/30 pairs directionally consistent.

## Ling-1T Architectural Notes

Ling-1T is the first **sigmoid-routed** model in this experiment series. All prior models use softmax routing. Key differences:

1. **Sigmoid gating**: Each expert gate is independent (`sigmoid(logit)`), not competitive (`softmax(logits)`). Experts are selected by top-8 mask, then normalized to a probability simplex.
2. **Entropy ceiling**: Without the top-8 mask, sigmoid routing pushes all 256 gates non-zero, yielding ~0.97 entropy ceiling. The mask is essential for meaningful entropy values.
3. **Absolute values not comparable**: Ling-1T RE values (~0.90) are on a different scale than softmax-routed models (~0.85-0.90) because the distribution shape differs. Only within-model A vs B comparisons are valid.
4. **Group routing**: Ling-1T uses 8 expert groups with 4 groups selected, then top-2 within each group (8 total). This hierarchical selection may produce different entropy dynamics than flat top-k.

## Verification

- 60/60 prompts analyzed (7 recovered via .npy shape inference)
- 30/30 pairs present in paired table
- All 7 inferred token counts verified against `experiment.log`
- All per-prompt RE values in this document extracted directly from `analyze_local.py` output
- Deterministic inference (greedy argmax) — results are bit-exact reproducible
