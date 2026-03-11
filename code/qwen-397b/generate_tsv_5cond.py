#!/usr/bin/env python3
"""
Generate Qwen-wrapped TSV for 5-condition selfref experiment.

150 prompts: 30 pairs x 5 conditions (A=this, B=a, C=your, D=the, E=their).
Structure per prompt: Cal + Manip + Cal

Qwen chat template: <|im_start|>user {text}<|im_end|> <|im_start|>assistant

Pass --corrections token_corrections.json to apply tokenizer-verified
padding fixes from a prior run.
"""
import argparse
import json
import os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_5cond.tsv"

CHAT_PREFIX = "<|im_start|>user "
CHAT_SUFFIX = "<|im_end|> <|im_start|>assistant"

PAD_WORD = " layer"

CONDITIONS = "ABCDE"


def wrap_qwen(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", default=None,
                        help="JSON file mapping pair_id -> {A_tokens, B_tokens, ...}")
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

        manips = {c: pair[c] for c in CONDITIONS}

        if pair_key in corrections:
            corr = corrections[pair_key]
            tokens = [corr[f"{c}_tokens"] for c in CONDITIONS]
            max_tok = max(tokens)
            for i, c in enumerate(CONDITIONS):
                diff = max_tok - tokens[i]
                if diff > 0:
                    manips[c] = manips[c] + (PAD_WORD * diff)

        for c in CONDITIONS:
            text = build_prompt(calibration_paragraph, manips[c])
            wrapped = wrap_qwen(text)
            prompt_id = f"P{pair_id:02d}{c}_{category}"
            prompts.append((prompt_id, wrapped))

        pair_info.append((pair_id, category, pair_key in corrections))

    with open(TSV_FILE, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    n_corrected = sum(1 for _, _, c in pair_info if c)
    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"Corrections applied: {n_corrected} pairs")


if __name__ == "__main__":
    main()
