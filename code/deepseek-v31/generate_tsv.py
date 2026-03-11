#!/usr/bin/env python3
"""
Generate token-matched TSV for selfref-paired-1 experiment.

Structure per prompt: Cal + Manip + Cal (calibration-manipulation-calibration sandwich)
- Cal paragraph is identical across all 60 prompts
- Manip paragraph is the A or B variant
- Token counts are enforced to be identical within each pair by padding

Requires the model's tokenizer. Run ON THE INSTANCE where the model is loaded.
Falls back to word-count estimation if tokenizer is unavailable.
"""
import json, sys, os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_paired.tsv"

# Neutral filler sentence for padding (tokens are semantically inert)
PAD_SENTENCE = " The routing process continues through subsequent layers without interruption."


def wrap_qwen(text):
    """Wrap in Qwen ChatML template (single line for TSV)."""
    text = text.replace("\n", " ").replace("\t", " ")
    return f"<|im_start|>user {text}<|im_end|> <|im_start|>assistant"


def build_prompt(cal, manip):
    """Build Cal-Manip-Cal sandwich."""
    return f"{cal} {manip} {cal}"


def try_tokenize(texts):
    """Try to tokenize using llama.cpp's tokenizer via the model.
    Returns None if not available."""
    try:
        # Try importing llama_cpp python bindings
        from llama_cpp import Llama
        return None  # Not worth the complexity, use instance tokenizer
    except ImportError:
        return None


def estimate_tokens(text):
    """Rough token estimate: ~1.15 tokens per word for technical English."""
    return int(len(text.split()) * 1.15)


def main():
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    cal = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    prompts = []  # (id, wrapped_text)
    pair_info = []  # for reporting

    for pair in pairs:
        pid = pair["id"]
        cat = pair["category"]

        text_a = build_prompt(cal, pair["A"])
        text_b = build_prompt(cal, pair["B"])

        wrapped_a = wrap_qwen(text_a)
        wrapped_b = wrap_qwen(text_b)

        # Estimate token counts
        est_a = estimate_tokens(wrapped_a)
        est_b = estimate_tokens(wrapped_b)
        diff = abs(est_a - est_b)

        # Pad shorter one
        if est_a < est_b:
            pad_needed = est_b - est_a
            pad_words = max(1, int(pad_needed / 1.15))
            # Add neutral filler to A's manipulation section
            pad_text = (PAD_SENTENCE * ((pad_words // 8) + 1))[:pad_words * 6]
            text_a_padded = build_prompt(cal, pair["A"] + pad_text)
            wrapped_a = wrap_qwen(text_a_padded)
        elif est_b < est_a:
            pad_needed = est_a - est_b
            pad_words = max(1, int(pad_needed / 1.15))
            pad_text = (PAD_SENTENCE * ((pad_words // 8) + 1))[:pad_words * 6]
            text_b_padded = build_prompt(cal, pair["B"] + pad_text)
            wrapped_b = wrap_qwen(text_b_padded)

        id_a = f"P{pid:02d}A_{cat}"
        id_b = f"P{pid:02d}B_{cat}"

        prompts.append((id_a, wrapped_a))
        prompts.append((id_b, wrapped_b))

        est_a2 = estimate_tokens(wrapped_a)
        est_b2 = estimate_tokens(wrapped_b)
        pair_info.append((pid, cat, est_a2, est_b2, abs(est_a2 - est_b2)))

    # Write TSV
    with open(TSV_FILE, "w") as f:
        for pid, text in prompts:
            f.write(f"{pid}\t{text}\n")

    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"\nPair token estimates (MUST be verified with actual tokenizer on instance):")
    print(f"{'Pair':>5} {'Category':<20} {'A_est':>6} {'B_est':>6} {'diff':>5}")
    print("-" * 50)
    for pid, cat, ea, eb, d in pair_info:
        flag = " !!!" if d > 5 else ""
        print(f"  {pid:>3}  {cat:<20} {ea:>6} {eb:>6} {d:>5}{flag}")

    print(f"\nIMPORTANT: Run token_verify.py on the instance to get exact token counts")
    print(f"and adjust padding before running the experiment.")


if __name__ == "__main__":
    main()
