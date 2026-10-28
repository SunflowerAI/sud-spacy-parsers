#!/usr/bin/env python
"""Re-measure the whole SUD MISC layer (Idiom, InIdiom, Subject, Reported, Shared) on an en_gum arm.

Standing hazard 5: every pipe there reads the base's own predictions, so ANY base retrain needs this. One
harness, one gold file, one scoring convention for all five keys (the one eval_sud_subject.py/eval_sud_shared.py
use: a token is right when the VALUE matches; a wrong or spurious prediction is an FP, a missed gold an FN;
Idiom/InIdiom/Reported are Yes-only), over GOLD TOKENS with everything else predicted -- so two arms differ only
in the base underneath.

    eval_misc_en_gum.py MODEL_DIR [--rules]   # --rules also scores the Subject and Shared RULES attached to the arm

Gold: assets/en_ewtgum-sud-test.relabeled_ext.reported.udep_resolved.conllu (the file the trained pipes' corpus is
hoisted from). sud_idiom and sud_reported_rule are added if the model lacks them (they are rules, not weights).
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: F401
import spacy
from spacy.tokens import Doc
import sud_idiom, sud_reported_rule, sud_subject_rule, sud_shared_rule  # noqa: F401
from sud_misc import get_misc

GOLD = "assets/en_ewtgum-sud-{split}.relabeled_ext.reported.udep_resolved.conllu"


def sents(path):
    block = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if block: yield block
            block = []
        elif not line.startswith("#"):
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]:
                block.append(f)
    if block: yield block


def kv(col, key):
    for it in col.split("|"):
        if it.startswith(key + "="):
            return it.split("=", 1)[1]
    return None


def gold_of(f, key):
    if key == "Shared": return kv(f[5], key) or kv(f[9], key)
    return kv(f[9], key)


def prf(c):
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 100 * p, 100 * r, 100 * (2 * p * r / (p + r) if p + r else 0.0), tp + fn


def score(nlp, rows, keys, src=None):
    cnt = {k: [0, 0, 0] for k in keys}
    for r in rows:
        doc = nlp(Doc(nlp.vocab, words=[f[1] for f in r], spaces=["SpaceAfter=No" not in f[9] for f in r]))
        assert len(doc) == len(r)
        for tok, f in zip(doc, r):
            for k in keys:
                pred = get_misc(tok, k)
                gold = gold_of(f, k)
                if k in ("Idiom", "InIdiom", "Reported"):
                    pred = pred if pred == "Yes" else None
                    gold = gold if gold == "Yes" else None
                c = cnt[k]
                if pred and gold and pred == gold: c[0] += 1
                else:
                    c[1] += bool(pred); c[2] += bool(gold)
    return {k: prf(v) for k, v in cnt.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--split", default="test")
    ap.add_argument("--rules", action="store_true"); a = ap.parse_args()
    rows = list(sents(GOLD.format(split=a.split)))
    print(f"model={a.model}  {a.split}: {len(rows)} sentences, gold tokens, everything else predicted")
    nlp = spacy.load(a.model)
    print("pipeline:", nlp.pipe_names)
    if "sud_idiom" not in nlp.pipe_names: nlp.add_pipe("sud_idiom", last=True)
    if "sud_reported" in nlp.pipe_names:      # the wheel ships the RULE (add_sud_reported_rule.py drops the trained pipe)
        nlp.remove_pipe("sud_reported")
    if "sud_reported_rule" not in nlp.pipe_names:
        nlp.add_pipe("sud_reported_rule", last=True, config={"lang": "en"})
    shipped_reported = True
    res = score(nlp, rows, ["Idiom", "InIdiom", "Subject", "Reported", "Shared"])
    for k, (p, r, f, n) in res.items():
        tag = {"Reported": "rule" if shipped_reported else "trained", "Idiom": "rule", "InIdiom": "rule",
               "Subject": "trained", "Shared": "trained"}[k]
        print(f"  {k:8s} ({tag:7s}) gold={n:5d}  P={p:6.2f} R={r:6.2f} F={f:6.2f}")
    if a.rules:
        for key, comp in (("Subject", "sud_subject_rule"), ("Shared", "sud_shared_rule")):
            n2 = spacy.load(a.model)
            n2.remove_pipe(f"sud_{key.lower()}")
            n2.add_pipe(comp, last=True, config={"lang": "en_gum" if key == "Shared" else "en"})
            p, r, f, n = score(n2, rows, [key])[key]
            print(f"  {key:8s} (rule   ) gold={n:5d}  P={p:6.2f} R={r:6.2f} F={f:6.2f}")
        n3 = spacy.load(a.model)
        if "sud_reported" in n3.pipe_names:
            p, r, f, n = score(n3, rows, ["Reported"])["Reported"]
            print(f"  Reported (trained) gold={n:5d}  P={p:6.2f} R={r:6.2f} F={f:6.2f}")


if __name__ == "__main__":
    main()
