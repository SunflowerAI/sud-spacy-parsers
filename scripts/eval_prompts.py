#!/usr/bin/env python3
"""Benchmark comp-vs-mod prompt variants against the confident udep-derived gold.

Prompts are organized as a STATIC PREFIX (instructions / few-shot examples) followed by a
short VARIABLE SUFFIX (the sentence to classify). Because we query items sequentially per
variant, Ollama reuses the cached KV of the shared prefix across calls — only the suffix is
re-processed. Build the gold first with scripts/build_gold.py.
"""
import argparse, importlib.util, json, random, time
from collections import Counter

_spec = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(d)


def load_gold(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def balanced_sample(gold, per_class, seed):
    rng = random.Random(seed)
    out = []
    for cls in ("complement", "modifier"):
        pool = [g for g in gold if g["gold"] == cls]
        out += rng.sample(pool, min(per_class, len(pool)))
    rng.shuffle(out)
    return out


# ---- variable suffix (the only part that changes per item; kept short) -------------
def suffix(item):
    return (f'Now classify this one.\nIn "{item["verb_phrase"]}", is "{item["prep_phrase"]}" '
            f'a complement or a modifier of "{item["verb"]}"?\nAnswer:')


# ---- static prefixes (identical across all items -> cached) -------------------------
SUD_DEFS = (
    "You will be given a verb and a prepositional phrase that depends on it. In "
    "Surface-Syntactic Universal Dependencies (SUD), decide whether the phrase is a "
    "COMPLEMENT or a MODIFIER of the verb.\n"
    "- COMPLEMENT (comp): an argument of the verb — an obligatory participant in the "
    "meaning of the verb, subcategorized (selected) by the verb. The verb requires or "
    "licenses it and typically fixes the choice of preposition (e.g. depend ON, talk "
    "ABOUT, refer TO, consist OF, accuse OF). Only certain verbs admit it.\n"
    "- MODIFIER (mod): an optional attribute that is not part of the verb's argument "
    "structure and can be added freely to a wide range of verbs (e.g. circumstances of "
    "time, place, manner, reason, or accompaniment: in the morning, at home, with a "
    "smile, for that reason).\n"
    "Answer with exactly one word: complement or modifier.\n\n")

FEWSHOT_HEAD = (
    "Decide whether a prepositional phrase is a COMPLEMENT (an argument the verb selects — "
    "the verb requires or licenses it and usually fixes its preposition) or a MODIFIER (an "
    "optional circumstantial of time, place, manner, reason, or accompaniment that attaches "
    "freely to many verbs). The preposition alone does not decide it; the same preposition "
    "can head a complement of one verb and a modifier of another. Answer with exactly one "
    "word: complement or modifier.\n\n")

# 6 generic examples (the previous few-shot)
SHOTS6 = [
    ("The result depends on the weather.", "on the weather", "depends", "complement"),
    ("She sang in the kitchen.", "in the kitchen", "sang", "modifier"),
    ("They accused him of fraud.", "of fraud", "accused", "complement"),
    ("He left early for safety.", "for safety", "left", "modifier"),
    ("The book consists of ten chapters.", "of ten chapters", "consists", "complement"),
    ("We met on Tuesday.", "on Tuesday", "met", "modifier"),
]

# 12 tuned examples: balanced, with same-preposition contrasts (on/for/at/in appear in
# both roles) so the model learns to key on verb selection, not the preposition.
SHOTS12 = [
    ("The result depends on the weather.", "on the weather", "depends", "complement"),
    ("She arrived on Friday.", "on Friday", "arrived", "modifier"),
    ("They accused him of fraud.", "of fraud", "accused", "complement"),
    ("He read the news in the morning.", "in the morning", "read", "modifier"),
    ("We waited for the bus.", "for the bus", "waited", "complement"),
    ("She left early for safety.", "for safety", "left", "modifier"),
    ("He looked at the painting.", "at the painting", "looked", "complement"),
    ("They met at noon.", "at noon", "met", "modifier"),
    ("The book consists of ten chapters.", "of ten chapters", "consists", "complement"),
    ("It happened during the war.", "during the war", "happened", "modifier"),
    ("She referred to the manual.", "to the manual", "referred", "complement"),
    ("He sang in the kitchen.", "in the kitchen", "sang", "modifier"),
]


def render_shots(shots):
    return "".join(
        f'In "{vp}", is "{pp}" a complement or a modifier of "{v}"?\nAnswer: {a}\n\n'
        for vp, pp, v, a in shots)


PREFIXES = {
    "sud_defs":       SUD_DEFS,
    "fewshot6":       FEWSHOT_HEAD + render_shots(SHOTS6),
    "fewshot12":      FEWSHOT_HEAD + render_shots(SHOTS12),
    "fewshot12_def":  SUD_DEFS + "Examples:\n\n" + render_shots(SHOTS12),
}


def score(preds, items):
    ok = sum(p == it["gold"] for p, it in zip(preds, items))
    per_class = {}
    for cls in ("complement", "modifier"):
        idx = [i for i, it in enumerate(items) if it["gold"] == cls]
        per_class[cls] = sum(preds[i] == cls for i in idx) / len(idx) if idx else 0.0
    return ok / len(items), per_class, Counter(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_udep.jsonl")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default=",".join(PREFIXES))
    ap.add_argument("--out", default="prompt_eval.json")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    sample = balanced_sample(gold, args.per_class, args.seed)
    print(f"Benchmark: {len(sample)} items {dict(Counter(g['gold'] for g in sample))} "
          f"from {args.gold}\n")

    results = {}
    for name in args.variants.split(","):
        prefix = PREFIXES[name]
        t0 = time.time()
        preds = [d.query(prefix + suffix(c)) for c in sample]
        dt = time.time() - t0
        preds = [p if p in ("complement", "modifier") else "?" for p in preds]
        acc, per_class, pred_counts = score(preds, sample)
        results[name] = {"acc": acc, "per_class": per_class,
                         "pred_counts": dict(pred_counts), "sec_per_item": dt / len(sample)}
        print(f"{name:14} acc={acc:.3f}  recall[comp]={per_class['complement']:.3f}  "
              f"recall[mod]={per_class['modifier']:.3f}  pred={dict(pred_counts)}  "
              f"({dt/len(sample):.2f}s/item)")

    with open(args.out, "w") as f:
        json.dump({"n": len(sample), "results": results}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
