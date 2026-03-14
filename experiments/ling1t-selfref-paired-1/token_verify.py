#!/usr/bin/env python3
"""
Verify and repair exact token matching for ling1t-selfref-paired-1.

Run this on the instance with the Ling-1T model available.

Usage:
  python3 token_verify.py
  python3 token_verify.py --fix

The script runs the capture binary on the current TSV (routing-only, n=0),
reads back n_tokens_prompt from each prompt's metadata.txt, then checks that
every A/B pair has identical token counts.

With --fix it inserts PAD_WORD tokens before the final <|role_end|><role>ASSISTANT</role>
suffix to equalize mismatched pairs.
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/Ling-1T-Q3_K_S/Ling-1T-Q3_K_S-00001-of-00001.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations_ling1t",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/llama.cpp-bailingmoe2/build/bin",
)
TSV_IN = "prompts_selfref_paired.tsv"
TSV_OUT = "prompts_selfref_paired.tsv"

# Insert pad words immediately before the final assistant turn marker.
CHAT_SUFFIX = "<|role_end|><role>ASSISTANT</role>"

# PAD_WORD: must add exactly +1 token on the Ling-1T tokenizer.
# Verify before use: run a 2-prompt TSV where prompt B = prompt A + PAD_WORD
# and confirm metadata.txt shows n_tokens_prompt differs by exactly 1.
# " Also" works for GPT-OSS (Qwen BPE base). Ling-1T uses a 157K vocab;
# this may or may not hold. Check and update if needed.
PAD_WORD = " Also"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))


def get_token_counts():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            BINARY,
            "-m", MODEL,
            "--prompt-file", TSV_IN,
            "-o", tmpdir,
            "-n", str(N_PREDICT),
            "-ngl", str(NGL),
            "-c", str(CTX),
            "-t", str(THREADS),
            "--routing-only",
            "--no-stream",
        ]
        print("Running Ling-1T tokenization check...")
        result = subprocess.run(cmd, env=env, capture_output=True)
        output = (result.stdout.decode("utf-8", errors="replace") +
                  result.stderr.decode("utf-8", errors="replace"))

        counts = {}
        for prompt_dir in pathlib.Path(tmpdir).iterdir():
            if not prompt_dir.is_dir():
                continue
            meta = prompt_dir / "metadata.txt"
            if not meta.exists():
                continue
            for line in meta.read_text().strip().splitlines():
                if line.startswith("n_tokens_prompt="):
                    counts[prompt_dir.name] = int(line.split("=", 1)[1])

        if not counts:
            # Fallback: parse from binary log output
            current_id = None
            for line in output.splitlines():
                m = re.match(r"\[(\d+)/\d+\]\s+(\S+)\s+:", line)
                if m:
                    current_id = m.group(2)
                    continue
                m = re.search(r"tokens:\s+(\d+)\s+prompt", line)
                if m and current_id:
                    counts[current_id] = int(m.group(1))
                    current_id = None

    return counts


def load_prompts():
    prompts = []
    with open(TSV_IN) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            pid, text = line.split("\t", 1)
            prompts.append((pid, text))
    return prompts


def save_prompts(prompts):
    with open(TSV_OUT, "w") as f:
        for pid, text in prompts:
            f.write(f"{pid}\t{text}\n")


def insert_padding(text, n_tokens):
    insert_pos = text.rfind(CHAT_SUFFIX)
    if insert_pos < 0:
        raise ValueError(f"Could not find chat suffix {CHAT_SUFFIX!r} in prompt")
    return text[:insert_pos] + (PAD_WORD * n_tokens) + text[insert_pos:]


def main():
    fix_mode = "--fix" in sys.argv
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from {TSV_IN}")

    counts = get_token_counts()
    if not counts:
        print("ERROR: no token counts recovered. Check BINARY and MODEL paths.")
        sys.exit(1)

    mismatches = []
    print(f"\n{'ID_A':<30} {'tok_A':>6} {'ID_B':<30} {'tok_B':>6} {'diff':>5} {'status':>9}")
    print("-" * 98)
    for idx in range(0, len(prompts), 2):
        id_a, _ = prompts[idx]
        id_b, _ = prompts[idx + 1]
        tok_a = counts.get(id_a, -1)
        tok_b = counts.get(id_b, -1)
        diff = abs(tok_a - tok_b)
        status = "OK" if diff == 0 else "MISMATCH"
        if diff:
            mismatches.append((idx, id_a, id_b, tok_a, tok_b, diff))
        print(f"  {id_a:<28} {tok_a:>6} {id_b:<28} {tok_b:>6} {diff:>5} {status:>9}")

    if not mismatches:
        print(f"\nAll {len(prompts) // 2} pairs are token-matched.")
        return

    print(f"\n{len(mismatches)} mismatched pair(s) found.")
    if not fix_mode:
        print("Run with --fix to pad the shorter member of each pair.")
        print(f"PAD_WORD='{PAD_WORD}' — verify this adds exactly +1 token before using --fix.")
        return

    print("\nRepairing mismatches...")
    for idx, id_a, id_b, tok_a, tok_b, diff in mismatches:
        if tok_a < tok_b:
            prompts[idx] = (id_a, insert_padding(prompts[idx][1], diff))
            print(f"  Padded {id_a} by {diff} × '{PAD_WORD}'")
        else:
            prompts[idx + 1] = (id_b, insert_padding(prompts[idx + 1][1], diff))
            print(f"  Padded {id_b} by {diff} × '{PAD_WORD}'")

    save_prompts(prompts)
    print(f"\nWrote repaired TSV to {TSV_OUT}")
    print("Re-run token_verify.py (without --fix) to confirm all pairs match.")


if __name__ == "__main__":
    main()
