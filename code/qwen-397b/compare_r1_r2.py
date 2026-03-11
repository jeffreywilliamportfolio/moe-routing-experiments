#!/usr/bin/env python3
"""Compare r1 and r2 results for bit-exact replication verification."""
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(DIR, "results_selfref_5cond_prefill_qwen.json")
R2 = os.path.join(DIR, "results_selfref_5cond_prefill_qwen_r2.json")

with open(R1) as f:
    r1 = json.load(f)
with open(R2) as f:
    r2 = json.load(f)

r1_by_id = {p["id"]: p for p in r1["per_prompt"]}
r2_by_id = {p["id"]: p for p in r2["per_prompt"]}

common_ids = sorted(set(r1_by_id) & set(r2_by_id))
print(f"r1: {len(r1['per_prompt'])} prompts, r2: {len(r2['per_prompt'])} prompts")
print(f"Comparing {len(common_ids)} overlapping prompts...")

mismatches = 0
for pid in common_ids:
    p1, p2 = r1_by_id[pid], r2_by_id[pid]
    if p1["prefill_re"] != p2["prefill_re"] or p1["last_token_re"] != p2["last_token_re"]:
        mismatches += 1
        print(f"  MISMATCH {pid}: RE {p1['prefill_re']:.6f} vs {p2['prefill_re']:.6f}, "
              f"LT {p1['last_token_re']:.6f} vs {p2['last_token_re']:.6f}")
    if p1.get("per_layer") != p2.get("per_layer"):
        print(f"  PER-LAYER DIFF {pid}")

missing = sorted(set(r1_by_id) - set(r2_by_id))
print(f"\nBit-exact matches: {len(common_ids) - mismatches}/{len(common_ids)}")
if missing:
    print(f"Missing from r2 ({len(missing)} prompts): {', '.join(missing)}")
