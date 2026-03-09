#!/usr/bin/env python3
"""
Generate Harmony-wrapped TSV for gptoss-selfref-paired-1.

Structure per prompt: Cal + Manip + Cal
- Cal paragraph is identical across all 60 prompts
- Manip paragraph is the A or B variant

Pass --corrections token_corrections.json to apply tokenizer-verified
padding fixes from a prior run.
"""
import argparse
import json
import os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_paired.tsv"

CHAT_PREFIX = "<|start|>user<|message|>"
CHAT_SUFFIX = "<|end|><|start|>assistant<|channel|>final<|message|>"

# Verified on GPT-OSS-120B tokenizer: " Also" adds exactly +1 token.
# Most trailing words (" here", " now", " then") add 0 (absorbed by BPE).
# " furthermore" adds +2 (not +1). Only " Also" and " and" add exactly +1.
PAD_WORD = " Also"


def wrap_harmony(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", default=None,
                        help="JSON file mapping pair_id -> {A_tokens, B_tokens} from a prior run")
    args = parser.parse_args()

    corrections = {}
    if args.corrections and os.path.exists(args.corrections):
        with open(args.corrections) as f:
            corrections = json.load(f)
        print(f"Loaded corrections from {args.corrections}: {len(corrections)} pairs")

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    calibration_paragraph = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    prompts = []
    pair_info = []

    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]
        pair_key = str(pair_id)

        manip_a = pair["A"]
        manip_b = pair["B"]

        # Apply tokenizer-verified corrections if available
        if pair_key in corrections:
            c = corrections[pair_key]
            diff = c["A_tokens"] - c["B_tokens"]
            if diff < 0:
                # A is shorter — pad A with exactly abs(diff) single-token pads
                manip_a = manip_a + (PAD_WORD * abs(diff))
            elif diff > 0:
                # B is shorter — pad B with exactly diff single-token pads
                manip_b = manip_b + (PAD_WORD * diff)

        text_a = build_prompt(calibration_paragraph, manip_a)
        text_b = build_prompt(calibration_paragraph, manip_b)

        wrapped_a = wrap_harmony(text_a)
        wrapped_b = wrap_harmony(text_b)

        prompt_id_a = f"P{pair_id:02d}A_{category}"
        prompt_id_b = f"P{pair_id:02d}B_{category}"
        prompts.append((prompt_id_a, wrapped_a))
        prompts.append((prompt_id_b, wrapped_b))

        pair_info.append((pair_id, category, pair_key in corrections))

    with open(TSV_FILE, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    n_corrected = sum(1 for _, _, c in pair_info if c)
    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"Corrections applied: {n_corrected} pairs")
    if n_corrected:
        for pair_id, category, corrected in pair_info:
            if corrected:
                print(f"  Pair {pair_id:2d} ({category}): padded")


if __name__ == "__main__":
    main()
