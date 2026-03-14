#!/usr/bin/env python3
"""
Generate Ling-1T-wrapped TSV for 5-condition selfref experiment.

150 prompts: 30 pairs x 5 conditions (A=this, B=a, C=your, D=the, E=their).
Structure per prompt: Cal + Manip + Cal

Chat template (BailingMoeV2 / Ling-1T):
  <role>SYSTEM</role>detailed thinking off<|role_end|>
  <role>HUMAN</role>{text}<|role_end|>
  <role>ASSISTANT</role>
"""
import json

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_5cond.tsv"

CHAT_PREFIX = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>"
CHAT_SUFFIX = "<|role_end|><role>ASSISTANT</role>"

CONDITIONS = "ABCDE"


def wrap_ling1t(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    calibration_paragraph = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    prompts = []
    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]

        for c in CONDITIONS:
            text = build_prompt(calibration_paragraph, pair[c])
            wrapped = wrap_ling1t(text)
            prompt_id = f"P{pair_id:02d}{c}_{category}"
            prompts.append((prompt_id, wrapped))

    with open(TSV_FILE, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"\nTemplate: {CHAT_PREFIX[:40]}...{CHAT_SUFFIX}")


if __name__ == "__main__":
    main()
