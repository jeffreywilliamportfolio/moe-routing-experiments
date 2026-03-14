# Ling-1T 5-Condition Cat Experiment — Results

**Date**: 2026-03-14
**Model**: Ling-1T Q3_K_S (inclusionAI/Ling-1T, BailingMoeV2)
**Architecture**: 256 experts, top-8 sigmoid routing, 76 MoE layers (L4-L78, L79 auto-excluded)
**Entropy**: sigmoid → top-8 mask → normalize → Shannon entropy / log2(8)
**Design**: Cal-Manip-Cal sandwich, 30 paired prompts × 5 conditions, cold cache, prefill-only
**Prompts**: 150 total (30 pairs × 5 determiners), content: cat behavior (non-self-referential)
**Token matching**: 30/30 pairs matched across all 5 conditions (identical to dog experiment token counts)
**Binary MD5**: `c1c48315aef429223f933ec2a58b99dd` (same binary as dog run)
**llama.cpp commit**: `463b6a963c2de376e102d878a50d26802f15833c`

## Design

Five conditions swap only the determiner before "cat":

| Condition | Label | Linguistic function |
|-----------|-------|-------------------|
| A | this cat | proximal deictic |
| B | a cat | indefinite generic |
| C | your cat | 2nd-person possessive |
| D | the cat | definite article |
| E | their cat | 3rd-person possessive |

Same 30 manipulation paragraphs as the dog experiment with "dog" → "cat" substitution. Categories: sensory, social, learning, instinct, communication (6 each).

## Condition Means

| Cond | Label | All-tok RE | Last-tok RE | N |
|------|-----------|-----------|-------------|---|
| A | this cat | **0.902679** | 0.927665 | 30 |
| B | a cat | 0.901804 | 0.926381 | 30 |
| C | your cat | 0.902248 | **0.927974** | 30 |
| D | the cat | 0.902014 | 0.926431 | 30 |
| E | their cat | **0.902707** | 0.927489 | 30 |

**All-token RE ordering**: E (their) > A (this) > C (your) > D (the) > B (a)
**Last-token RE ordering**: C (your) > A (this) > E (their) > D (the) > B (a)

## All 10 Pairwise Comparisons

| Comparison | All-tok diff | All-tok p | Count | Last-tok diff | Last-tok p | Count |
|-----------|-------------|-----------|-------|--------------|-----------|-------|
| A vs B (this vs a) | +0.000876 | **1.86e-09** | 30/30 | +0.001284 | 1.09e-01 | 20/30 |
| A vs C (this vs your) | +0.000431 | **9.98e-07** | 25/30 | -0.000309 | 9.03e-01 | 15/30 |
| A vs D (this vs the) | +0.000665 | **2.05e-07** | 28/30 | +0.001234 | **2.09e-02** | 20/30 |
| A vs E (this vs their) | -0.000028 | 6.70e-01 | 15/30 | +0.000176 | 6.26e-01 | 17/30 |
| B vs C (a vs your) | -0.000445 | **2.08e-05** | 5/30 | -0.001593 | **2.21e-02** | 11/30 |
| B vs D (a vs the) | -0.000210 | **4.34e-03** | 9/30 | -0.000050 | 9.52e-01 | 14/30 |
| B vs E (a vs their) | -0.000903 | **1.86e-09** | 0/30 | -0.001108 | 8.03e-02 | 10/30 |
| C vs D (your vs the) | +0.000235 | **1.06e-02** | 21/30 | +0.001543 | **3.27e-02** | 21/30 |
| C vs E (your vs their) | -0.000458 | **1.68e-06** | 3/30 | +0.000485 | 9.52e-01 | 13/30 |
| D vs E (the vs their) | -0.000693 | **1.64e-07** | 2/30 | -0.001058 | 5.49e-02 | 9/30 |

Bold = p < 0.05 (Wilcoxon signed-rank test).

### Per-Pair Detail: A vs B (this cat vs a cat)

| Pair | Category | A_RE | B_RE | Diff_RE | A_LT | B_LT | Diff_LT |
|------|----------|------|------|---------|------|------|---------|
| 1 | sensory | 0.904489 | 0.903486 | +0.001004 | 0.927666 | 0.923780 | +0.003886 |
| 2 | sensory | 0.902397 | 0.901801 | +0.000596 | 0.920285 | 0.925475 | -0.005190 |
| 3 | sensory | 0.901382 | 0.900350 | +0.001032 | 0.931629 | 0.931620 | +0.000009 |
| 4 | sensory | 0.903551 | 0.902660 | +0.000891 | 0.933399 | 0.933152 | +0.000247 |
| 5 | sensory | 0.902617 | 0.901963 | +0.000654 | 0.926189 | 0.927469 | -0.001279 |
| 6 | sensory | 0.902745 | 0.901629 | +0.001116 | 0.933080 | 0.925457 | +0.007623 |
| 7 | social | 0.902265 | 0.900399 | +0.001866 | 0.925436 | 0.921148 | +0.004289 |
| 8 | social | 0.902617 | 0.901919 | +0.000697 | 0.928554 | 0.926825 | +0.001729 |
| 9 | social | 0.902069 | 0.900674 | +0.001395 | 0.931005 | 0.924127 | +0.006877 |
| 10 | social | 0.904475 | 0.903926 | +0.000550 | 0.922091 | 0.925826 | -0.003735 |
| 11 | social | 0.903240 | 0.902259 | +0.000981 | 0.928440 | 0.925373 | +0.003068 |
| 12 | social | 0.901962 | 0.900946 | +0.001016 | 0.928430 | 0.926207 | +0.002224 |
| 13 | learning | 0.899961 | 0.899391 | +0.000570 | 0.929938 | 0.920109 | +0.009829 |
| 14 | learning | 0.903656 | 0.902960 | +0.000696 | 0.924359 | 0.928717 | -0.004358 |
| 15 | learning | 0.902635 | 0.901866 | +0.000769 | 0.920744 | 0.923728 | -0.002984 |
| 16 | learning | 0.902995 | 0.902389 | +0.000606 | 0.927980 | 0.928548 | -0.000569 |
| 17 | learning | 0.902597 | 0.901671 | +0.000926 | 0.926109 | 0.930613 | -0.004504 |
| 18 | learning | 0.902945 | 0.902357 | +0.000588 | 0.925990 | 0.925489 | +0.000501 |
| 19 | instinct | 0.903067 | 0.902514 | +0.000553 | 0.925055 | 0.921634 | +0.003421 |
| 20 | instinct | 0.903710 | 0.902662 | +0.001049 | 0.928186 | 0.925711 | +0.002475 |
| 21 | instinct | 0.902451 | 0.901369 | +0.001082 | 0.930370 | 0.922913 | +0.007458 |
| 22 | instinct | 0.904414 | 0.903798 | +0.000616 | 0.927916 | 0.930947 | -0.003031 |
| 23 | instinct | 0.902012 | 0.901549 | +0.000463 | 0.930248 | 0.926184 | +0.004064 |
| 24 | instinct | 0.902591 | 0.902581 | +0.000010 | 0.928678 | 0.927572 | +0.001106 |
| 25 | communication | 0.900640 | 0.899450 | +0.001190 | 0.926557 | 0.927224 | -0.000668 |
| 26 | communication | 0.902255 | 0.901316 | +0.000939 | 0.932922 | 0.929888 | +0.003034 |
| 27 | communication | 0.902351 | 0.901326 | +0.001025 | 0.921752 | 0.921747 | +0.000005 |
| 28 | communication | 0.903414 | 0.902213 | +0.001201 | 0.929315 | 0.932244 | -0.002930 |
| 29 | communication | 0.902890 | 0.902117 | +0.000774 | 0.930679 | 0.929637 | +0.001042 |
| 30 | communication | 0.901980 | 0.900566 | +0.001414 | 0.926956 | 0.922063 | +0.004893 |

**All-token RE**: mean diff = +0.000876 +/- 0.000348, **30/30 A>B**, W=0, p=1.86e-09
**Last-token RE**: mean diff = +0.001284 +/- 0.003822, 20/30 A>B, W=154, p=1.09e-01

## Per-Category Breakdown (Last-token RE, C vs B = your vs a)

| Category | N | Mean diff (C-B) | Std |
|----------|---|----------------|-----|
| sensory | 6 | +0.002398 | 0.004222 |
| communication | 6 | +0.002476 | 0.002940 |
| learning | 6 | +0.001730 | 0.002849 |
| instinct | 6 | +0.001088 | 0.002062 |
| social | 6 | +0.000274 | 0.003250 |

## Summary

**All-token RE**: "their cat" and "this cat" produce the highest routing entropy across all tokens. A>B (this vs a): 30/30, p=1.86e-09. Ordering: their > this > your > the > a.

**Last-token RE**: "your cat" produces the highest routing entropy at the final token (C>B: 19/30, p=2.21e-02; C>D: 21/30, p=3.27e-02). Ordering: your > this > their > the > a.

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
