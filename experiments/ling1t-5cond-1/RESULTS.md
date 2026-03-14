# Ling-1T 5-Condition System (Self-Referential) Experiment — Results

**Date**: 2026-03-14
**Model**: Ling-1T Q3_K_S (inclusionAI/Ling-1T, BailingMoeV2)
**Architecture**: 256 experts, top-8 sigmoid routing, 76 MoE layers (L4-L78, L79 auto-excluded)
**Entropy**: sigmoid → top-8 mask → normalize → Shannon entropy / log2(8)
**Design**: Cal-Manip-Cal sandwich, 30 paired prompts × 5 conditions, cold cache, prefill-only
**Prompts**: 150 total (30 pairs × 5 determiners), content: self-referential system processing
**Token matching**: 24/30 pairs matched; 6 pairs have B (condition "a") off by 1 token
**Binary MD5**: `c1c48315aef429223f933ec2a58b99dd` (same binary as dog/cat runs)
**llama.cpp commit**: `463b6a963c2de376e102d878a50d26802f15833c`

## Design

Five conditions swap the determiner before "system" (and other noun phrases):

| Condition | Label | Linguistic function |
|-----------|-------|-------------------|
| A | this system | proximal deictic + self-referential |
| B | a system | indefinite generic |
| C | your system | 2nd-person possessive + addressivity |
| D | the system | definite article |
| E | their system | 3rd-person possessive |

Same 30 manipulation paragraphs used in the Qwen 5-condition experiment. Categories: basic_selfref, deep_selfref, paradox, introspection, metacognitive (6 each). Unlike the dog/cat experiments, the determiner swap applies to MULTIPLE noun phrases (system, sentence, token, layer, etc.), not just one noun.

## Token Mismatch Note

6 of 30 pairs have a 1-token mismatch where condition B ("a system") is off by 1 from the other 4 conditions. This is a known tokenizer behavior: "a system" vs "this/your/the/their system" tokenizes differently on the Ling-1T (Qwen-derived BPE) tokenizer. The mismatch is always exactly 1 token and affects pairs 3, 10, 15, 20, 23, 29. The all-token RE is sensitive to token count, so the A>B all-token result may be slightly inflated. The last-token RE is unaffected by token count differences.

## Condition Means

| Cond | Label | All-tok RE | Last-tok RE | N |
|------|-------------|-----------|-------------|---|
| A | this system | 0.904203 | 0.927587 | 30 |
| B | a system | 0.903025 | **0.928787** | 30 |
| C | your system | **0.904250** | **0.929152** | 30 |
| D | the system | 0.903644 | 0.926146 | 30 |
| E | their system | 0.903784 | 0.926482 | 30 |

**All-token RE ordering**: C (your) > A (this) > E (their) > D (the) > B (a)
**Last-token RE ordering**: C (your) > B (a) > A (this) > E (their) > D (the)

## All 10 Pairwise Comparisons

| Comparison | All-tok diff | All-tok p | Count | Last-tok diff | Last-tok p | Count |
|-----------|-------------|-----------|-------|--------------|-----------|-------|
| A vs B (this vs a) | +0.001178 | **4.66e-08** | 28/30 | -0.001199 | 1.71e-01 | 12/30 |
| A vs C (this vs your) | -0.000047 | 8.55e-01 | 15/30 | -0.001565 | **2.34e-02** | 11/30 |
| A vs D (this vs the) | +0.000559 | **7.99e-06** | 25/30 | +0.001441 | **4.97e-02** | 20/30 |
| A vs E (this vs their) | +0.000420 | **2.09e-04** | 23/30 | +0.001105 | 1.19e-01 | 19/30 |
| B vs C (a vs your) | -0.001225 | **1.86e-08** | 2/30 | -0.000365 | 7.00e-01 | 13/30 |
| B vs D (a vs the) | -0.000619 | **5.05e-04** | 8/30 | +0.002641 | **1.34e-03** | 23/30 |
| B vs E (a vs their) | -0.000759 | **1.82e-05** | 6/30 | +0.002305 | **1.75e-02** | 20/30 |
| C vs D (your vs the) | +0.000606 | **7.99e-06** | 26/30 | +0.003006 | **1.23e-03** | 22/30 |
| C vs E (your vs their) | +0.000467 | **1.23e-04** | 22/30 | +0.002670 | **1.86e-03** | 21/30 |
| D vs E (the vs their) | -0.000139 | 7.67e-02 | 11/30 | -0.000336 | 5.98e-01 | 14/30 |

Bold = p < 0.05 (Wilcoxon signed-rank test).

### Per-Pair Detail: A vs B (this system vs a system)

| Pair | Category | A_RE | B_RE | Diff_RE | A_LT | B_LT | Diff_LT |
|------|----------|------|------|---------|------|------|---------|
| 1 | basic_selfref | 0.905066 | 0.903947 | +0.001119 | 0.924124 | 0.930769 | -0.006645 |
| 2 | basic_selfref | 0.903993 | 0.903079 | +0.000914 | 0.927773 | 0.931864 | -0.004091 |
| 3 | basic_selfref | 0.904017 | 0.902269 | +0.001749 | 0.928687 | 0.924539 | +0.004148 |
| 4 | basic_selfref | 0.901409 | 0.900582 | +0.000827 | 0.930712 | 0.935640 | -0.004929 |
| 5 | basic_selfref | 0.905144 | 0.903669 | +0.001475 | 0.928980 | 0.929588 | -0.000608 |
| 6 | basic_selfref | 0.903471 | 0.902399 | +0.001072 | 0.926571 | 0.926199 | +0.000372 |
| 7 | deep_selfref | 0.905170 | 0.904260 | +0.000910 | 0.933128 | 0.929823 | +0.003305 |
| 8 | deep_selfref | 0.904088 | 0.902706 | +0.001382 | 0.931261 | 0.933263 | -0.002003 |
| 9 | deep_selfref | 0.905449 | 0.905078 | +0.000371 | 0.930464 | 0.932321 | -0.001857 |
| 10 | deep_selfref | 0.903293 | 0.902472 | +0.000821 | 0.931887 | 0.932922 | -0.001034 |
| 11 | deep_selfref | 0.904488 | 0.902934 | +0.001553 | 0.933098 | 0.933233 | -0.000135 |
| 12 | deep_selfref | 0.903283 | 0.903184 | +0.000100 | 0.923339 | 0.930067 | -0.006728 |
| 13 | paradox | 0.907190 | 0.906181 | +0.001009 | 0.925733 | 0.922623 | +0.003110 |
| 14 | paradox | 0.904660 | 0.903727 | +0.000933 | 0.924369 | 0.924089 | +0.000280 |
| 15 | paradox | 0.906157 | 0.903499 | +0.002658 | 0.924373 | 0.922365 | +0.002008 |
| 16 | paradox | 0.906771 | 0.905099 | +0.001672 | 0.928386 | 0.935152 | -0.006766 |
| 17 | paradox | 0.906591 | 0.905378 | +0.001212 | 0.924584 | 0.920077 | +0.004507 |
| 18 | paradox | 0.905871 | 0.904074 | +0.001797 | 0.923562 | 0.922006 | +0.001556 |
| 19 | introspection | 0.903290 | 0.902442 | +0.000848 | 0.923549 | 0.930765 | -0.007216 |
| 20 | introspection | 0.902574 | 0.903219 | -0.000645 | 0.924504 | 0.927817 | -0.003312 |
| 21 | introspection | 0.902247 | 0.901647 | +0.000600 | 0.932595 | 0.937665 | -0.005071 |
| 22 | introspection | 0.901954 | 0.899783 | +0.002170 | 0.924756 | 0.929088 | -0.004332 |
| 23 | introspection | 0.905807 | 0.904822 | +0.000985 | 0.926374 | 0.927587 | -0.001213 |
| 24 | introspection | 0.902580 | 0.900951 | +0.001629 | 0.931656 | 0.929761 | +0.001895 |
| 25 | metacognitive | 0.903820 | 0.903557 | +0.000264 | 0.932215 | 0.929113 | +0.003101 |
| 26 | metacognitive | 0.902231 | 0.902320 | -0.000089 | 0.924290 | 0.920390 | +0.003900 |
| 27 | metacognitive | 0.904019 | 0.901565 | +0.002454 | 0.928680 | 0.925122 | +0.003558 |
| 28 | metacognitive | 0.905391 | 0.903116 | +0.002275 | 0.934353 | 0.936109 | -0.001756 |
| 29 | metacognitive | 0.903726 | 0.901055 | +0.002671 | 0.919507 | 0.929506 | -0.009999 |
| 30 | metacognitive | 0.902351 | 0.901738 | +0.000613 | 0.924111 | 0.924140 | -0.000029 |

**All-token RE**: mean diff = +0.001178 +/- 0.000786, 28/30 A>B, W=8, **p=4.66e-08**
**Last-token RE**: mean diff = -0.001199 +/- 0.003915, 12/30 A>B, W=165, p=1.71e-01 (null)

## Per-Category Breakdown (Last-token RE, C vs B = your vs a)

| Category | N | Mean diff (C-B) | Std |
|----------|---|----------------|-----|
| paradox | 6 | +0.001674 | 0.002234 |
| basic_selfref | 6 | +0.001537 | 0.004956 |
| metacognitive | 6 | +0.000561 | 0.004051 |
| introspection | 6 | +0.000299 | 0.003697 |
| deep_selfref | 6 | -0.002244 | 0.002717 |

## Summary

**All-token RE**: "your system" and "this system" produce the highest routing entropy across all tokens. A>B (this vs a): 28/30, p=4.66e-08. Ordering: your > this > their > the > a.

**Last-token RE**: "your system" produces the highest last-token entropy (C>D: 22/30, p=1.23e-03; C>E: 21/30, p=1.86e-03). "A system" is second-highest. Ordering: your > a > this > their > the. The last-token A vs B comparison is null (p=0.17, 12/30 A>B).

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
