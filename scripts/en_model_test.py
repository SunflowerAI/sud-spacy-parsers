#!/usr/bin/env python3
"""Test whether a bigger model breaks the English comp/mod prompting plateau (~0.912).
Runs the best prompt (fewshot12_def) on several local models over a balanced gold subset.
"""
import argparse, importlib.util, json, urllib.request
from collections import Counter


def _load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


e = _load("e", "scripts/eval_prompts.py")
# Endpoint only -- so a remote-GPU run configures client and server with one OLLAMA_HOST.
d = _load("d", "scripts/disambiguate_pp.py")

MODELS = ["qwen3:8b", "glm-4.7-flash:latest"]


def query(prompt, model):
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "options": {"temperature": 0}}
    req = urllib.request.Request(d.OLLAMA_URL,
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--per-class", type=int, default=50, help="items per class (balanced)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", help="write the per-model table here as JSON")
    args = ap.parse_args()

    gold = e.load_gold("gold/gold_udep.jsonl")
    test = e.balanced_sample(gold, args.per_class, args.seed)
    prefix = e.PREFIXES["fewshot12_def"]
    print(f"en test {len(test)} ({args.per_class}/class), prompt=fewshot12_def, "
          f"endpoint={d.OLLAMA_URL}\n", flush=True)
    rows = []
    for m in args.models:
        warm(m)
        preds = [query(prefix + e.suffix(c), m) for c in test]
        acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
        rc = {c: sum(preds[i] == c for i, t in enumerate(test) if t["gold"] == c)
              / max(1, sum(t["gold"] == c for t in test)) for c in ("complement", "modifier")}
        print(f"  {m:26} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
              f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}", flush=True)
        rows.append({"model": m, "n": len(test), "acc": acc,
                     "recall_complement": rc["complement"], "recall_modifier": rc["modifier"],
                     "preds": dict(Counter(preds))})
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"prompt": "fewshot12_def", "per_class": args.per_class,
                       "seed": args.seed, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
