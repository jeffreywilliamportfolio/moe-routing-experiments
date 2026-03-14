#!/usr/bin/env python3
"""
Generate Ling-1T-wrapped TSV for ling1t-selfref-paired-1.

Structure per prompt: Cal + Manip + Cal
- Cal paragraph is identical across all 60 prompts
- Manip paragraph is the A or B variant

Chat template (BailingMoeV2 / Ling-1T):
  <role>SYSTEM</role>detailed thinking off<|role_end|>
  <role>HUMAN</role>{text}<|role_end|>
  <role>ASSISTANT</role>

add_bos_token: false (Ling-1T tokenizer does not prepend BOS)

Pass --corrections token_corrections.json to apply tokenizer-verified
padding fixes from token_verify output.
"""
import argparse
import json
import os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_paired.tsv"

CHAT_PREFIX = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>"
CHAT_SUFFIX = "<|role_end|><role>ASSISTANT</role>"

# PAD_WORD: single-token padding for Ling-1T tokenizer.
# MUST be verified on the instance before use. " Also" works for GPT-OSS;
# may differ for Ling-1T's 157K-vocab Qwen-derived BPE.
# TODO: run token_verify.py to confirm pad word.
PAD_WORD = " Also"


def wrap_ling1t(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", default=None,
                        help="JSON file mapping pair_id -> {A_tokens, B_tokens} from token_verify")
    parser.add_argument("--suite", default=PROMPT_SUITE,
                        help="Prompt suite JSON (default: prompt_suite.json)")
    args = parser.parse_args()

    corrections = {}
    if args.corrections and os.path.exists(args.corrections):
        with open(args.corrections) as f:
            corrections = json.load(f)
        print(f"Loaded corrections: {len(corrections)} pairs")

    with open(args.suite) as f:
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

        if pair_key in corrections:
            c = corrections[pair_key]
            diff = c["A_tokens"] - c["B_tokens"]
            if diff < 0:
                manip_a = manip_a + (PAD_WORD * abs(diff))
            elif diff > 0:
                manip_b = manip_b + (PAD_WORD * diff)

        text_a = build_prompt(calibration_paragraph, manip_a)
        text_b = build_prompt(calibration_paragraph, manip_b)

        wrapped_a = wrap_ling1t(text_a)
        wrapped_b = wrap_ling1t(text_b)

        prompt_id_a = f"P{pair_id:02d}A_{category}"
        prompt_id_b = f"P{pair_id:02d}B_{category}"
        prompts.append((prompt_id_a, wrapped_a))
        prompts.append((prompt_id_b, wrapped_b))

        pair_info.append((pair_id, category, pair_key in corrections))

    with open(TSV_FILE, "w") as f:
        for pid, text in prompts:
            f.write(f"{pid}\t{text}\n")

    n_corrected = sum(1 for _, _, c in pair_info if c)
    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"Corrections applied: {n_corrected}/{len(pairs)} pairs")
    if n_corrected < len(pairs):
        print(f"\nWARNING: {len(pairs) - n_corrected} pairs have NO tokenizer-verified corrections.")
        print("Run token_verify.py on the instance before the main experiment.")
    print(f"\nTemplate: {CHAT_PREFIX[:40]}...{CHAT_SUFFIX}")
    print(f"PAD_WORD: '{PAD_WORD}' (verify +1 token on Ling-1T tokenizer before use)")


if __name__ == "__main__":
    main()
