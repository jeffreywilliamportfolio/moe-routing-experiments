# MoE Routing Distinguishes Self-Referential From Generic Content Across Five Architectures, Four Organizations, and Three Training Methodologies

**Jeffrey William Shorthill**

---

My name is Jeffrey William Shorthill.

I want to be upfront about who I am and what this is. I am not an ML researcher. I'm an independent full stack developer in Utah who works for a nonprofit bringing AI tech to underserved populations. My degree is in recording engineering. I've been building with frontier model APIs for about three years. On February 27th of this year, I rented GPUs and ran inference at scale for the first time. Everything in this post comes from the ten days since then.

**What I found:** when a model processes text about itself, its MoE expert routing looks measurably different than when it processes equally complex text about something else. This replicates across five models from four organizations with three different training methodologies. It appears to require the content to be about the system processing the sentence. The entire project ran on rented H200s for under $600 in compute using a fork of llama.cpp b8123, greedy argmax throughout, cold cache, prefill-only.

## Five Model Results

| Model | Organization | Experts | Active | Layers | Training | Last-token p | All-token p |
|-------|-------------|---------|--------|--------|----------|-------------|------------|
| Qwen 397B | Alibaba | 512 | 10 | 60 | Standard | 8.86e-5 | 5.6e-9 |
| GLM-5 | Zhipu AI | 256 | 8 | 75 | Standard | 4.6e-4 | 4.4e-5 |
| DeepSeek V3.1 | DeepSeek | 256 | 8 | 58 | Standard | 0.011 | null |
| DeepSeek R1 | DeepSeek | 256 | 8 | 58 | RL | null | 0.001 |
| gpt-oss-120b | OpenAI | 128 | 4 | 36 | Distilled | null | 0.021 |

Five models, four organizations, four different expert counts, three training methodologies. All showed significance on at least one metric. The two largest models, Qwen 397B and GLM-5, showed significance on both all-token and last-token simultaneously. Each model was tested on the same 30 token-matched prompt pairs using a Cal-Manip-Cal design inspired by neuroimaging techniques, described below.

Something I noticed but don't have a full explanation for: the signal appears at different positions depending on the model and its training. The two standard-trained models with the most routing capacity (Qwen with 512 experts and GLM-5 with 256 experts across 75 layers) pick it up everywhere. DeepSeek V3.1, also standard-trained but shallower, shows only last-token. DeepSeek R1, same architecture as V3.1 but RL-trained, flips to all-token only. gpt-oss-120b, the smallest and shallowest model tested, shows all-token only. It looks like architecture determines the magnitude and training determines the position, but I'm not confident in that interpretation. If someone has a theoretical framework for why training methodology would change where a routing effect manifests but not whether it exists, I'd be interested.

---

## Prompt Structure

The prompt structure borrows from fMRI block design. You present a control condition, then a stimulus, then return to control, and measure the deviation. I applied the same logic to MoE routing distributions.

Each prompt has three segments: a calibration paragraph about transformer routing in general (identical across all prompts and conditions), a manipulation paragraph containing the experimental content, and the same calibration paragraph again. I'm calling this a Cal-Manip-Cal sandwich. The calibration blocks establish a within-prompt routing baseline and verify that both conditions start from the same computational state.

The manipulation paragraph differs by a single word between conditions. Condition A says "this system." Condition B says "a system." Same token count, same syntax, same vocabulary. Token matching was verified through the target model's tokenizer before capture. Any mismatches were corrected with single-token padding and rerun until all 30 pairs matched exactly.

My understanding is that this design eliminates the positional confound I describe later, because both conditions have the same token count and the same positional properties. If that reasoning is flawed, I'd like to know.

## Controls

### Control 1: Recursive Content Not About the Model

The first thing I wanted to rule out was that the word "this" alone was driving the result. I ran the same substitution on recursive content with no AI or self-referential framing: Godel's incompleteness theorems, Escher's impossible staircases, bootstrap paradoxes, quines, tangled hierarchies. "This paradox" vs "a paradox." 30 token-matched pairs, zero mismatches, DeepSeek V3.1.

What I found: 14/30, p=0.685. Nothing. Mann-Whitney comparing the self-referential and strange loop pair distributions came back at p=0.025, Cohen's d=0.43. As far as I can tell, the word "this" alone doesn't produce the effect. The content appears to need to be about the system processing the sentence.

### Control 2: Three-Condition Test

If the effect is about self-reference rather than a specific token, then a different word pointing at the system should produce a similar result. I ran a three-condition experiment on GLM-5: "this system" (A) vs "a system" (B) vs "your system" (C). 30 token-matched pairs per comparison.

| Comparison | Last-token p | Direction |
|-----------|-------------|-----------|
| A vs B ("this" vs "a") | 4.6e-4 | 23/30 A>B |
| B vs C ("a" vs "your") | 7.3e-4 | 24/30 C>B |
| A vs C ("this" vs "your") | 0.40 | null |

What this seems to show is that the two self-referential framings are statistically indistinguishable and both differ from the generic. It looks like the routing responds to what the sentence is about, not which specific word is used. But I want to be careful here because this is only on one model so far.

I ran the same design on DeepSeek R1 and found "your" was actually a slightly stronger signal than "this" at last-token (p=0.031). I think this might have something to do with RL training sensitizing the model to second-person address, but I'm speculating.

## What I Killed Before I Got Here

My first ten experiments showed a monotonic correlation between routing entropy and prompt complexity across 168 prompts at 12 levels. rho=0.84 (p=3.9e-45) on DeepSeek R1, replicated on V3.1 and Qwen. Then I checked whether prompt length predicted entropy better than complexity did. It did. rho=0.88 with token count. When I controlled for position using last-token entropy, the hierarchy vanished (rho=0.02, p=0.82) and the direction reversed: simple prompts had higher last-token entropy than complex ones.

The mechanism: routing entropy increases with token position within any prompt. Longer prompts have more late-position tokens, inflating the mean. My complexity hierarchy was correlated with length. Ten experiments, three models, all invalidated. The paired design above was built specifically to eliminate this confound.

I'm including this as context for evaluating the results that survived, and as a warning: this positional confound applies to any MoE routing entropy analysis using all-token averages across prompts of different lengths. If you're working in this space, check for it.

## What I'm Not Claiming

I am not claiming that this is evidence of self-awareness in models. I am also not claiming to understand the theoretical and practical implications of what I found. I am simply saying that across five models from four organizations, changing one word produces a measurable, statistically significant change in MoE expert routing. I'm not sure what this means. I know how to measure it. I'd like help understanding the difference.

---

Thank you,
Jeffrey William Shorthill
