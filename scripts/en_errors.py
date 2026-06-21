#!/usr/bin/env python3
"""Error analysis: which comp/mod cases does qwen3:8b (fewshot12_def) get wrong?
Stays on qwen3:8b; output drives targeted contrastive examples."""
import importlib.util
from collections import Counter

_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)
d = e.d


def main():
    gold = e.load_gold("gold_udep.jsonl")
    test = e.balanced_sample(gold, 200, 0)
    prefix = e.PREFIXES["fewshot12_def"]
    miss_c, miss_m, prep_err = [], [], Counter()
    for c in test:
        p = d.query(prefix + e.suffix(c))
        if p == c["gold"]:
            continue
        prep = c["prep_phrase"].split()[0].lower()
        prep_err[(c["gold"], prep)] += 1
        (miss_c if c["gold"] == "complement" else miss_m).append(c)
    print(f"missed complements (gold=comp, pred=mod): {len(miss_c)}")
    print(f"missed modifiers   (gold=mod,  pred=comp): {len(miss_m)}")
    print(f"\nerrors by (gold, leading prep):")
    for k, n in prep_err.most_common(20):
        print(f"  {k}: {n}")
    print("\n--- sample missed COMPLEMENTS (verb | phrase) ---")
    for c in miss_c[:25]:
        print(f"  {c['verb']} | {c['prep_phrase'][:55]}")


if __name__ == "__main__":
    main()
