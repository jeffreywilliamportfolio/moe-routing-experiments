# Ling-1T Expert Decomposition — Dog / Cat / System

**Date**: 2026-03-14
**Model**: Ling-1T Q3_K_S (BailingMoeV2), 256 experts, top-8 sigmoid routing
**Data**: 3 × 150 prompts (dog, cat, system), 75 MoE layers each (L4-L78)
**Method**: sigmoid(logits) → top-8 mask → identify which 8 of 256 experts fire per token per layer
**Threshold**: expert classified as "condition-preferring" if normalized rate diff > 0.001

## Expert Classification Summary

| Category | Count | Experts |
|----------|-------|---------|
| Universal addressivity (your-preferring in dog + cat + system) | 2 | E95, E185 |
| Animal-only addressivity (your-preferring in dog + cat, NOT system) | 4 | E60, E70, E188, E208 |
| System-only addressivity (your-preferring in system, NOT dog/cat) | 15 | E2, E31, E35, E43, E47, E76, E90, E117, E137, E175, E194, E216, E225, E247, E250 |
| Cat+system addressivity (not dog) | 2 | — |
| Dog+system addressivity (not cat) | 0 | — |
| Universal "this"-preferring (this > a in all 3) | 1 | E117 |
| System-only "this"-preferring | 11 | — |

Total experts showing condition preference at threshold 0.001:
- Dog: 6 prefer "your" over "a"
- Cat: 9 prefer "your" over "a"
- System: 19 prefer "your" over "a"

## Your-Preferring Expert Counts

The number of experts recruited by the "your" determiner scales with content self-relevance:

| Domain | Experts preferring "your" | Ratio to dog |
|--------|--------------------------|-------------|
| Dog | 6 | 1.0x |
| Cat | 9 | 1.5x |
| System | 19 | 3.2x |

## Top "Your > A" Experts by Domain

### Dog (6 experts above threshold)

| Expert | "your" rate | "a" rate | Diff |
|--------|------------|---------|------|
| E185 | 0.0263 | 0.0240 | +0.0023 |
| E208 | 0.0398 | 0.0385 | +0.0013 |
| E95 | 0.0274 | 0.0261 | +0.0013 |
| E60 | 0.0271 | 0.0258 | +0.0013 |
| E188 | 0.0297 | 0.0286 | +0.0012 |
| E70 | 0.0338 | 0.0328 | +0.0010 |

### Cat (9 experts above threshold)

| Expert | "your" rate | "a" rate | Diff |
|--------|------------|---------|------|
| E185 | 0.0262 | 0.0241 | +0.0021 |
| E208 | 0.0400 | 0.0386 | +0.0015 |
| E72 | 0.0333 | 0.0320 | +0.0013 |
| E70 | 0.0334 | 0.0322 | +0.0013 |
| E221 | 0.0448 | 0.0437 | +0.0012 |
| E188 | 0.0296 | 0.0285 | +0.0011 |
| E95 | 0.0272 | 0.0261 | +0.0011 |
| E155 | 0.0313 | 0.0302 | +0.0010 |
| E60 | 0.0271 | 0.0261 | +0.0010 |

### System (19 experts above threshold)

| Expert | "your" rate | "a" rate | Diff |
|--------|------------|---------|------|
| E155 | 0.0311 | 0.0293 | +0.0018 |
| E137 | 0.0267 | 0.0250 | +0.0017 |
| E185 | 0.0257 | 0.0241 | +0.0017 |
| E117 | 0.0556 | 0.0541 | +0.0016 |
| E31 | 0.0308 | 0.0293 | +0.0015 |
| E216 | 0.0280 | 0.0266 | +0.0014 |
| E47 | 0.0317 | 0.0303 | +0.0014 |
| E221 | 0.0438 | 0.0425 | +0.0014 |
| E76 | 0.0203 | 0.0190 | +0.0013 |
| E247 | 0.0287 | 0.0274 | +0.0013 |
| E194 | 0.0359 | 0.0347 | +0.0012 |
| E43 | 0.0258 | 0.0246 | +0.0012 |
| E250 | 0.0343 | 0.0331 | +0.0012 |
| E175 | 0.0418 | 0.0406 | +0.0012 |
| E90 | 0.0359 | 0.0348 | +0.0011 |
| E2 | 0.0296 | 0.0284 | +0.0011 |
| E95 | 0.0292 | 0.0281 | +0.0011 |
| E35 | 0.0243 | 0.0232 | +0.0010 |
| E225 | 0.0333 | 0.0323 | +0.0010 |

## Top "A > Your" Experts (suppressed by addressivity)

Experts that fire LESS when text says "your" — suppressed by the possessive determiner.

### Consistent across all 3 domains

| Expert | Dog diff | Cat diff | System diff |
|--------|---------|---------|------------|
| E89 | -0.0014 | -0.0016 | -0.0014 |
| E50 | -0.0008 | -0.0010 | -0.0016 |
| E213 | -0.0008 | -0.0008 | -0.0013 |
| E183 | -0.0013 | -0.0009 | — |
| E74 | -0.0008 | -0.0011 | — |
| E123 | -0.0006 | -0.0009 | -0.0015 |
| E181 | -0.0007 | -0.0009 | -0.0010 |
| E54 | -0.0009 | -0.0008 | — |

E89 is the strongest "your"-suppressed expert across all domains.

## "This"-Preferring Experts

### Universal (all 3 domains)

| Expert | Dog diff | Cat diff | System diff |
|--------|---------|---------|------------|
| E117 | +0.0013 | +0.0012 | +0.0021 |

E117 is the model's primary deictic-specificity expert. It responds to "this" across all content types, with 2x the effect for system content.

### System-only "this"-preferring (11 experts)

| Expert | System diff |
|--------|------------|
| E31 | +0.0015 |
| E26 | +0.0015 |
| E221 | +0.0015 |
| E188 | +0.0014 |
| E194 | +0.0014 |
| E162 | +0.0012 |
| E46 | +0.0012 |
| E112 | +0.0012 |
| E152 | +0.0011 |

These fire for "this system" but not "this dog" or "this cat." Self-referential deictic experts.

## Per-Layer Expert Divergence (your vs a)

Jaccard distance between the expert sets of "your X" and "a X", averaged across 30 pairs. Higher = more different expert routing.

### Dog — Top 15 most divergent layers

| Layer | Mean Jaccard |
|-------|-------------|
| 70 | 0.1486 |
| 59 | 0.1467 |
| 50 | 0.1460 |
| 74 | 0.1446 |
| 75 | 0.1444 |
| 56 | 0.1435 |
| 54 | 0.1414 |
| 58 | 0.1405 |
| 51 | 0.1405 |
| 72 | 0.1399 |
| 62 | 0.1399 |
| 57 | 0.1398 |
| 71 | 0.1387 |
| 61 | 0.1385 |
| 68 | 0.1385 |

Bottom 5: L4 (0.0165), L5 (0.0146), L6 (0.0311), L7 (0.0361), L8 (0.0286)

### System — Top 15 most divergent layers

| Layer | Mean Jaccard |
|-------|-------------|
| 50 | 0.2646 |
| 58 | 0.2611 |
| 57 | 0.2586 |
| 56 | 0.2575 |
| 51 | 0.2565 |
| 61 | 0.2541 |
| 54 | 0.2535 |
| 53 | 0.2516 |
| 70 | 0.2516 |
| 65 | 0.2511 |
| 59 | 0.2508 |
| 49 | 0.2497 |
| 66 | 0.2495 |
| 62 | 0.2486 |
| 63 | 0.2486 |

Bottom 5: L4 (0.1197), L5 (0.1217), L6 (0.1417), L8 (0.1407), L7 (0.1492)

### Cross-domain divergence comparison

| Metric | Dog | System | Ratio |
|--------|-----|--------|-------|
| Peak Jaccard (most divergent layer) | 0.149 | 0.265 | 1.78x |
| Bottom Jaccard (least divergent layer) | 0.015 | 0.120 | 8.0x |
| Mean across all layers | ~0.12 | ~0.23 | ~1.9x |

System content routes to different experts almost 2x more than dog content at every layer. Even at the least divergent layers (L4-L5), system "your vs a" shows 12% expert set difference — 8x the 1.5% seen for dog at those same layers. The model never treats "your system" and "a system" as equivalent, even in the earliest MoE layers.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Model | Ling-1T Q3_K_S (9 shards, 402GB) |
| Instance | 8x NVIDIA H200 (Vast.ai) |
| Architecture | BailingMoeV2, 256 experts, top-8 sigmoid |
| MoE layers analyzed | 75 (L4-L78, L79 excluded) |
| Prompts per domain | 150 (30 pairs × 5 conditions) |
| Total firings analyzed | ~20M per domain (6.5M per condition × 3 domains) |
