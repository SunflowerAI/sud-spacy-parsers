#!/usr/bin/env python3
"""Test whether a bigger model breaks the English comp/mod prompting plateau (~0.912).
Runs the best prompt (fewshot12_def) on several local models over a balanced gold subset.
"""
import importlib.util, json, urllib.request
from collections import Counter

_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)

MODELS = ["qwen3:8b", "glm-4.7-flash:latest"]


def query(prompt, model):
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "options": {"temperature": 0}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.load(r)["response"].strip().lower()
    # robust: a model may emit reasoning; take the LAST comp/mod mention
    ci, mi = raw.rfind("complement"), raw.rfind("modifier")
    if ci == mi == -1:
        return "?"
    return "complement" if ci > mi else "modifier"


def warm(model):
    try:
        query("Reply with one word: hello.", model)
    except Exception as ex:
        print(f"  (warm {model}: {ex})")


def main():
    gold = e.load_gold("gold_udep.jsonl")
    test = e.balanced_sample(gold, 50, 0)   # 100 items
    prefix = e.PREFIXES["fewshot12_def"]
    print(f"en test {len(test)} (50/class), prompt=fewshot12_def\n", flush=True)
    for m in MODELS:
        warm(m)
        preds = [query(prefix + e.suffix(c), m) for c in test]
        acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
        rc = {c: sum(preds[i] == c for i, t in enumerate(test) if t["gold"] == c)
              / max(1, sum(t["gold"] == c for t in test)) for c in ("complement", "modifier")}
        print(f"  {m:26} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
              f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}")


if __name__ == "__main__":
    main()
