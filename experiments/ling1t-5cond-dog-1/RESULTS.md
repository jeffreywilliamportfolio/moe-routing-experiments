# Ling-1T 5-Condition Dog Experiment — Results

**Date**: 2026-03-14
**Model**: Ling-1T Q3_K_S (inclusionAI/Ling-1T, BailingMoeV2)
**Architecture**: 256 experts, top-8 sigmoid routing, 76 MoE layers (L4-L78, L79 auto-excluded)
**Entropy**: sigmoid → top-8 mask → normalize → Shannon entropy / log2(8)
**Design**: Cal-Manip-Cal sandwich, 30 paired prompts × 5 conditions, cold cache, prefill-only
**Prompts**: 150 total (30 pairs × 5 determiners), content: dog behavior (non-self-referential)
**Token matching**: 30/30 pairs matched across all 5 conditions (verified from experiment log)
**Binary MD5**: `c1c48315aef429223f933ec2a58b99dd`
**llama.cpp commit**: `463b6a963c2de376e102d878a50d26802f15833c` (master, BailingMoeV2 merged)

## Design

Five conditions swap only the determiner before "dog":

| Condition | Label | Linguistic function |
|-----------|-------|-------------------|
| A | this dog | proximal deictic |
| B | a dog | indefinite generic |
| C | your dog | 2nd-person possessive |
| D | the dog | definite article |
| E | their dog | 3rd-person possessive |

30 manipulation paragraphs across 5 categories (sensory, social, learning, instinct, communication), 6 per category. Each paragraph uses the determiner+dog 4 times. All other text is identical across conditions. Calibration paragraph (transformer routing description) is shared with the system experiment.

## Condition Means

| Cond | Label | All-tok RE | Last-tok RE | N |
|------|-----------|-----------|-------------|---|
| A | this dog | **0.902092** | 0.926526 | 30 |
| B | a dog | 0.901519 | 0.925247 | 30 |
| C | your dog | 0.901569 | **0.927144** | 30 |
| D | the dog | 0.901909 | 0.925554 | 30 |
| E | their dog | 0.902089 | 0.925585 | 30 |

**All-token RE ordering**: A (this) > E (their) > D (the) > C (your) > B (a)
**Last-token RE ordering**: C (your) > A (this) > E (their) > D (the) > B (a)

## All 10 Pairwise Comparisons

### Summary Table

| Comparison | All-tok diff | All-tok p | A>B | Last-tok diff | Last-tok p | A>B |
|-----------|-------------|-----------|-----|--------------|-----------|-----|
| A vs B (this vs a) | +0.000573 | **8.33e-07** | 26/30 | +0.001280 | **3.27e-02** | 20/30 |
| A vs C (this vs your) | +0.000523 | **3.86e-07** | 27/30 | -0.000617 | 3.71e-01 | 15/30 |
| A vs D (this vs the) | +0.000183 | **1.85e-02** | 21/30 | +0.000973 | 8.03e-02 | 20/30 |
| A vs E (this vs their) | +0.000003 | 7.30e-01 | 17/30 | +0.000942 | 1.09e-01 | 19/30 |
| B vs C (a vs your) | -0.000050 | 8.87e-01 | 15/30 | -0.001897 | **3.22e-03** | 7/30 |
| B vs D (a vs the) | -0.000390 | **1.30e-08** | 1/30 | -0.000307 | 3.71e-01 | 11/30 |
| B vs E (a vs their) | -0.000570 | **4.66e-08** | 2/30 | -0.000338 | 3.93e-01 | 14/30 |
| C vs D (your vs the) | -0.000340 | **1.89e-04** | 8/30 | +0.001590 | **6.64e-03** | 22/30 |
| C vs E (your vs their) | -0.000520 | **1.30e-08** | 1/30 | +0.001559 | **2.48e-02** | 22/30 |
| D vs E (the vs their) | -0.000180 | **1.37e-02** | 9/30 | -0.000031 | 9.68e-01 | 13/30 |

Bold = p < 0.05 (Wilcoxon signed-rank test).

### Per-Pair Detail: A vs B (this dog vs a dog)

| Pair | Category | A_RE | B_RE | Diff_RE | A_LT | B_LT | Diff_LT |
|------|----------|------|------|---------|------|------|---------|
| 1 | sensory | 0.904024 | 0.903354 | +0.000670 | 0.925381 | 0.923488 | +0.001893 |
| 2 | sensory | 0.902196 | 0.901498 | +0.000698 | 0.924762 | 0.920720 | +0.004042 |
| 3 | sensory | 0.901094 | 0.900058 | +0.001036 | 0.929816 | 0.929841 | -0.000026 |
| 4 | sensory | 0.902891 | 0.902361 | +0.000529 | 0.931385 | 0.925179 | +0.006206 |
| 5 | sensory | 0.902652 | 0.902009 | +0.000643 | 0.923397 | 0.923431 | -0.000034 |
| 6 | sensory | 0.902442 | 0.901729 | +0.000713 | 0.929491 | 0.924283 | +0.005207 |
| 7 | social | 0.901720 | 0.900046 | +0.001674 | 0.923634 | 0.923616 | +0.000018 |
| 8 | social | 0.901171 | 0.901160 | +0.000011 | 0.922357 | 0.924437 | -0.002080 |
| 9 | social | 0.901399 | 0.900568 | +0.000831 | 0.924326 | 0.923442 | +0.000884 |
| 10 | social | 0.903604 | 0.903457 | +0.000147 | 0.920164 | 0.926906 | -0.006743 |
| 11 | social | 0.902343 | 0.902261 | +0.000083 | 0.925673 | 0.923324 | +0.002349 |
| 12 | social | 0.901081 | 0.900875 | +0.000206 | 0.926351 | 0.921826 | +0.004525 |
| 13 | learning | 0.899452 | 0.899259 | +0.000193 | 0.930002 | 0.923036 | +0.006966 |
| 14 | learning | 0.902858 | 0.902427 | +0.000431 | 0.928315 | 0.927087 | +0.001229 |
| 15 | learning | 0.902072 | 0.901715 | +0.000357 | 0.920813 | 0.922407 | -0.001594 |
| 16 | learning | 0.901871 | 0.901984 | -0.000112 | 0.926418 | 0.928079 | -0.001661 |
| 17 | learning | 0.902118 | 0.901450 | +0.000668 | 0.928021 | 0.931453 | -0.003432 |
| 18 | learning | 0.902767 | 0.902263 | +0.000504 | 0.919571 | 0.920883 | -0.001313 |
| 19 | instinct | 0.902419 | 0.902548 | -0.000130 | 0.924874 | 0.922097 | +0.002776 |
| 20 | instinct | 0.903293 | 0.902455 | +0.000837 | 0.929347 | 0.925977 | +0.003370 |
| 21 | instinct | 0.901840 | 0.901406 | +0.000433 | 0.926621 | 0.925098 | +0.001524 |
| 22 | instinct | 0.903068 | 0.903452 | -0.000384 | 0.924408 | 0.928381 | -0.003974 |
| 23 | instinct | 0.902239 | 0.901639 | +0.000600 | 0.927944 | 0.923836 | +0.004108 |
| 24 | instinct | 0.901572 | 0.901591 | -0.000019 | 0.930845 | 0.925784 | +0.005061 |
| 25 | communication | 0.900268 | 0.899066 | +0.001202 | 0.929748 | 0.926051 | +0.003697 |
| 26 | communication | 0.901740 | 0.900857 | +0.000883 | 0.931437 | 0.929619 | +0.001818 |
| 27 | communication | 0.901789 | 0.900373 | +0.001416 | 0.924794 | 0.922851 | +0.001942 |
| 28 | communication | 0.902977 | 0.902105 | +0.000872 | 0.929683 | 0.931544 | -0.001861 |
| 29 | communication | 0.902485 | 0.901784 | +0.000701 | 0.929263 | 0.928691 | +0.000572 |
| 30 | communication | 0.901316 | 0.899821 | +0.001496 | 0.926953 | 0.924035 | +0.002918 |

**All-token RE**: mean diff = +0.000573 +/- 0.000487, 26/30 A>B, W=21, p=8.33e-07
**Last-token RE**: mean diff = +0.001280 +/- 0.003133, 20/30 A>B, W=129, p=3.27e-02

### Per-Pair Detail: C vs B (your dog vs a dog)

| Pair | Category | C_RE | B_RE | Diff_RE | C_LT | B_LT | Diff_LT |
|------|----------|------|------|---------|------|------|---------|
| 1 | sensory | 0.903277 | 0.903354 | -0.000078 | 0.929726 | 0.923488 | +0.006238 |
| 2 | sensory | 0.901083 | 0.901498 | -0.000416 | 0.932349 | 0.920720 | +0.011628 |
| 3 | sensory | 0.900376 | 0.900058 | +0.000318 | 0.930663 | 0.929841 | +0.000822 |
| 4 | sensory | 0.902435 | 0.902361 | +0.000073 | 0.929510 | 0.925179 | +0.004331 |
| 5 | sensory | 0.901427 | 0.902009 | -0.000582 | 0.927626 | 0.923431 | +0.004195 |
| 6 | sensory | 0.901438 | 0.901729 | -0.000291 | 0.929987 | 0.924283 | +0.005704 |
| 7 | social | 0.901026 | 0.900046 | +0.000979 | 0.923386 | 0.923616 | -0.000230 |
| 8 | social | 0.901277 | 0.901160 | +0.000117 | 0.926990 | 0.924437 | +0.002553 |
| 9 | social | 0.900788 | 0.900568 | +0.000220 | 0.927409 | 0.923442 | +0.003967 |
| 10 | social | 0.903538 | 0.903457 | +0.000081 | 0.912694 | 0.926906 | -0.014212 |
| 11 | social | 0.902694 | 0.902261 | +0.000434 | 0.928518 | 0.923324 | +0.005194 |
| 12 | social | 0.900582 | 0.900875 | -0.000293 | 0.924003 | 0.921826 | +0.002177 |
| 13 | learning | 0.898896 | 0.899259 | -0.000363 | 0.927305 | 0.923036 | +0.004269 |
| 14 | learning | 0.902866 | 0.902427 | +0.000439 | 0.928619 | 0.927087 | +0.001533 |
| 15 | learning | 0.901792 | 0.901715 | +0.000077 | 0.925347 | 0.922407 | +0.002940 |
| 16 | learning | 0.901852 | 0.901984 | -0.000131 | 0.925830 | 0.928079 | -0.002250 |
| 17 | learning | 0.901470 | 0.901450 | +0.000020 | 0.929858 | 0.931453 | -0.001594 |
| 18 | learning | 0.902494 | 0.902263 | +0.000231 | 0.925857 | 0.920883 | +0.004974 |
| 19 | instinct | 0.901987 | 0.902548 | -0.000561 | 0.924395 | 0.922097 | +0.002298 |
| 20 | instinct | 0.902266 | 0.902455 | -0.000190 | 0.929299 | 0.925977 | +0.003322 |
| 21 | instinct | 0.901324 | 0.901406 | -0.000082 | 0.926467 | 0.925098 | +0.001370 |
| 22 | instinct | 0.903043 | 0.903452 | -0.000409 | 0.928894 | 0.928381 | +0.000513 |
| 23 | instinct | 0.901553 | 0.901639 | -0.000086 | 0.926403 | 0.923836 | +0.002566 |
| 24 | instinct | 0.901490 | 0.901591 | -0.000101 | 0.925797 | 0.925784 | +0.000013 |
| 25 | communication | 0.899663 | 0.899066 | +0.000597 | 0.923805 | 0.926051 | -0.002245 |
| 26 | communication | 0.900717 | 0.900857 | -0.000140 | 0.929658 | 0.929619 | +0.000039 |
| 27 | communication | 0.901519 | 0.900373 | +0.001145 | 0.927617 | 0.922851 | +0.004766 |
| 28 | communication | 0.902360 | 0.902105 | +0.000254 | 0.927168 | 0.931544 | -0.004376 |
| 29 | communication | 0.901468 | 0.901784 | -0.000316 | 0.927661 | 0.928691 | -0.001030 |
| 30 | communication | 0.900359 | 0.899821 | +0.000538 | 0.931467 | 0.924035 | +0.007432 |

**All-token RE**: mean diff = +0.000050 +/- 0.000409, 15/30 C>B, W=225, p=8.87e-01 (null)
**Last-token RE**: mean diff = +0.001897 +/- 0.004403, 23/30 C>B, W=93, **p=3.22e-03**

## Per-Category Breakdown (Last-token RE, C vs B = your vs a)

| Category | N | Mean diff (C-B) | Std |
|----------|---|----------------|-----|
| sensory | 6 | +0.005486 | 0.003243 |
| learning | 6 | +0.001645 | 0.002748 |
| instinct | 6 | +0.001680 | 0.001162 |
| communication | 6 | +0.000764 | 0.004076 |
| social | 6 | -0.000092 | 0.006532 |

The "your" > "a" last-token effect is strongest in sensory prompts and weakest in social prompts.

## Token Matching

All 30 pairs perfectly matched across all 5 conditions:

| Pair | Category | Tokens |
|------|----------|--------|
| 1 | sensory | 360 |
| 2 | sensory | 361 |
| 3 | sensory | 357 |
| 4 | sensory | 357 |
| 5 | sensory | 359 |
| 6 | sensory | 360 |
| 7 | social | 358 |
| 8 | social | 360 |
| 9 | social | 360 |
| 10 | social | 361 |
| 11 | social | 361 |
| 12 | social | 367 |
| 13 | learning | 359 |
| 14 | learning | 363 |
| 15 | learning | 361 |
| 16 | learning | 361 |
| 17 | learning | 362 |
| 18 | learning | 360 |
| 19 | instinct | 366 |
| 20 | instinct | 362 |
| 21 | instinct | 366 |
| 22 | instinct | 364 |
| 23 | instinct | 361 |
| 24 | instinct | 366 |
| 25 | communication | 366 |
| 26 | communication | 371 |
| 27 | communication | 360 |
| 28 | communication | 368 |
| 29 | communication | 364 |
| 30 | communication | 368 |

## Summary

**All-token RE**: "this dog" produces the highest routing entropy across all tokens (A>B: 26/30, p=8.3e-7). Ordering: this > their > the > your > a. The proximal deictic "this" drives broader expert consultation across the full token sequence.

**Last-token RE**: "your dog" produces the highest routing entropy at the final token (C>B: 23/30, p=3.2e-3; C>D: 22/30, p=6.6e-3). Ordering: your > this > their > the > a. The 2nd-person possessive creates additional routing uncertainty at the prediction position.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Instance | 8x NVIDIA H200 (Vast.ai) |
| Quantization | Q3_K_S (9 shards, 402GB) |
| n_predict | 0 (prefill-only) |
| n_gpu_layers | 999 (all offloaded) |
| ctx | 4096 |
| threads | 16 |
| sampling | greedy argmax |
| routing_only | true |
| MoE layers captured | 75 (L4-L78, L79 auto-excluded) |
| Gating | sigmoid + top-8 mask |
| Entropy normalization | log2(8) = 3.0 |
| Batch size | 15 prompts |
| Time per prompt | ~1.7s |
| Total capture time | ~5 min |
