#!/usr/bin/env python3
"""
Token count verification for 5-condition dog experiment on Ling-1T.

Run ON THE INSTANCE where the model is loaded. Verifies that all 5 conditions
produce the same token count for each pair.

Usage (on instance):
    python3 token_verify.py

Requires llama_cpp (pip install llama-cpp-python) or the tokenizer from
the GGUF metadata.
"""
import json
import sys

PROMPT_SUITE = "prompt_suite.json"
CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this dog",
    "B": "a dog",
    "C": "your dog",
    "D": "the dog",
    "E": "their dog",
}

CHAT_PREFIX = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>"
CHAT_SUFFIX = "<|role_end|><role>ASSISTANT</role>"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def wrap_ling1t(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def main():
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama_cpp not available. Install with: pip install llama-cpp-python")
        print("Or run this script on the instance where the model is loaded.")
        return 1

    import os
    model_path = os.environ.get(
        "MODEL_PATH",
        "/workspace/models/Ling-1T-Q3_K_S/Q3_K_S/Ling-1T-Q3_K_S-00001-of-00009.gguf",
    )

    print(f"Loading tokenizer from {model_path}...")
    llm = Llama(model_path=model_path, n_ctx=128, n_gpu_layers=0, verbose=False)

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    cal = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    mismatches = 0
    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c:>5}" for c in CONDITIONS) + "  Status")
    print("-" * 70)

    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]
        counts = {}
        for c in CONDITIONS:
            text = build_prompt(cal, pair[c])
            wrapped = wrap_ling1t(text)
            tokens = llm.tokenize(wrapped.encode("utf-8"), add_bos=False)
            counts[c] = len(tokens)

        values = list(counts.values())
        status = "OK" if len(set(values)) == 1 else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1

        tok_str = " ".join(f"{counts[c]:>5}" for c in CONDITIONS)
        print(f"  {pair_id:>3}  {category:<20} {tok_str}  {status}")

    print(f"\n{mismatches}/{len(pairs)} pairs have token count mismatches.")

    if mismatches > 0:
        print("\nToken mismatches detected. Determiners may tokenize differently.")
        print("Consider padding shorter variants with a verified single-token word.")
        print("Run: python3 -c \"from llama_cpp import Llama; m=Llama('<model>', n_ctx=128, n_gpu_layers=0, verbose=False); print(len(m.tokenize(b' Also', add_bos=False)))\"")
        return 1

    print("\nAll pairs token-matched across 5 conditions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
