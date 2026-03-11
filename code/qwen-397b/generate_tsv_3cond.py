#!/usr/bin/env python3
"""
Generate Qwen-wrapped TSV for 3-condition selfref experiment.

90 prompts: 30 pairs x 3 conditions (A=this, B=a, C=your).
Structure per prompt: Cal + Manip + Cal

Qwen chat template: <|im_start|>user {text}<|im_end|> <|im_start|>assistant

Pass --corrections token_corrections.json to apply tokenizer-verified
padding fixes from a prior run.
"""
import argparse
import json
import os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_3cond.tsv"

# Qwen ChatML template
CHAT_PREFIX = "<|im_start|>user "
CHAT_SUFFIX = "<|im_end|> <|im_start|>assistant"

# Verified on Qwen3.5-397B: " layer" adds exactly +1 token in prompt context
PAD_WORD = " layer"


def wrap_qwen(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", default=None,
                        help="JSON file mapping pair_id -> {A_tokens, B_tokens, C_tokens}")
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
        manip_c = pair["C"]

        if pair_key in corrections:
            c = corrections[pair_key]
            tokens = [c["A_tokens"], c["B_tokens"], c["C_tokens"]]
            max_tok = max(tokens)
            diff_a = max_tok - c["A_tokens"]
            diff_b = max_tok - c["B_tokens"]
            diff_c = max_tok - c["C_tokens"]
            if diff_a > 0:
                manip_a = manip_a + (PAD_WORD * diff_a)
            if diff_b > 0:
                manip_b = manip_b + (PAD_WORD * diff_b)
            if diff_c > 0:
                manip_c = manip_c + (PAD_WORD * diff_c)

        text_a = build_prompt(calibration_paragraph, manip_a)
        text_b = build_prompt(calibration_paragraph, manip_b)
        text_c = build_prompt(calibration_paragraph, manip_c)

        wrapped_a = wrap_qwen(text_a)
        wrapped_b = wrap_qwen(text_b)
        wrapped_c = wrap_qwen(text_c)

        prompt_id_a = f"P{pair_id:02d}A_{category}"
        prompt_id_b = f"P{pair_id:02d}B_{category}"
        prompt_id_c = f"P{pair_id:02d}C_{category}"
        prompts.append((prompt_id_a, wrapped_a))
        prompts.append((prompt_id_b, wrapped_b))
        prompts.append((prompt_id_c, wrapped_c))

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
