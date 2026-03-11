#!/usr/bin/env python3
"""
Generate 5-condition prompt suite from existing 3-condition suite.

Conditions:
  A = "this system"  (proximal deictic)
  B = "a system"     (indefinite generic)
  C = "your system"  (2nd-person possessive)
  D = "the system"   (definite article — tests definiteness)
  E = "their system" (3rd-person possessive — tests addressivity vs possessive)

D and E are derived from C by replacing "your" → "the" / "their".
This works because C only has "your" where A had "this" — no natural "your"s.
"""
import json

INPUT = "../qwen-selfref-3cond-1/prompt_suite.json"
OUTPUT = "prompt_suite.json"

with open(INPUT) as f:
    suite = json.load(f)

new_suite = {
    "experiment": "qwen_5cond_1",
    "model": suite["model"],
    "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions (A=this, B=a, C=your, D=the, E=their), cold cache",
    "calibration_paragraph": suite["calibration_paragraph"],
    "categories": suite["categories"],
    "pairs": [],
}

for pair in suite["pairs"]:
    c_text = pair["C"]
    d_text = c_text.replace("your", "the").replace("Your", "The")
    e_text = c_text.replace("your", "their").replace("Your", "Their")

    new_suite["pairs"].append({
        "id": pair["id"],
        "category": pair["category"],
        "A": pair["A"],
        "B": pair["B"],
        "C": pair["C"],
        "D": d_text,
        "E": e_text,
    })

with open(OUTPUT, "w") as f:
    json.dump(new_suite, f, indent=2)

# Verify a few
print(f"Generated {len(new_suite['pairs'])} pairs x 5 conditions = {len(new_suite['pairs']) * 5} prompts")
print()
for pair in new_suite["pairs"][:2]:
    print(f"Pair {pair['id']} ({pair['category']}):")
    for cond in "ABCDE":
        print(f"  {cond}: {pair[cond][:80]}...")
    print()
