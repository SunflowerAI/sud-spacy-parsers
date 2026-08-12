#!/usr/bin/env python3
"""How much is UPOS+FEATS worth for predicting XPOS -- on GOLD features, and on PREDICTED ones?

The two answers are very different, and the gap between them is the whole finding recorded in
NEGATIVE-RESULTS.md ("Making XPOS downstream of UPOS and FEATS"). Majority-class maps fitted on
train and scored on test say gold UPOS+FEATS is worth up to +19.6 XPOS points on top of the form;
re-key the same maps on what a released arm actually PREDICTS and they land BELOW the tagger they
were meant to improve, on all ten arms. Morphology is predicted at exact-bundle accuracy 0.75-0.99
and its errors fall on the tokens the tagger also finds hard, so the channel is noise correlated
with the target.

Without --model this is treebank-only and instant. With one it loads the arm and re-keys on its
predictions, which is the number that actually decides whether conditioning can pay.

    xpos_headroom.py assets_ar/.../ar_padt-sud-train.relabeled_ext.conllu \\
                     assets_ar/.../ar_padt-sud-test.relabeled_ext.conllu
    xpos_headroom.py <train.conllu> <test.conllu> --model training_ar_lemma/model-best \\
                     --test-spacy corpus_ar_ext/ar_padt-sud-test.relabeled_ext.spacy
"""
import argparse
import collections
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def read_conllu(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or "-" in f[0] or "." in f[0]:
                continue
            rows.append((f[1], f[3], f[4], f[5]))          # FORM UPOS XPOS FEATS
    return rows


# The backoff ladder is load-bearing and easy to get wrong: dropping the form-free (UPOS, FEATS)
# rung costs ar 4.7 points, because that rung is the one that answers UNSEEN forms -- and ar's XPOS
# is 99.9 % determined by UPOS+FEATS alone. An impoverished ladder understates the gold oracle and
# would make the experiment look better-founded than it is.
KEYS = {
    "form":            lambda f, u, m: f.lower(),
    "form+upos":       lambda f, u, m: (f.lower(), u),
    "form+upos+feats": lambda f, u, m: (f.lower(), u, m),
    "upos+feats":      lambda f, u, m: (u, m),
}
LADDER = ["form+upos+feats", "form+upos", "form", "upos+feats"]


def fit(rows, key):
    d = collections.defaultdict(collections.Counter)
    for form, upos, xpos, feats in rows:
        d[key(form, upos, feats)][xpos] += 1
    return {k: c.most_common(1)[0][0] for k, c in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_conllu")
    ap.add_argument("test_conllu")
    ap.add_argument("--model", default=None, help="arm to take PREDICTED upos/feats/xpos from")
    ap.add_argument("--test-spacy", default=None, help="the .spacy test corpus (with --model)")
    a = ap.parse_args()

    tr, te = read_conllu(a.train_conllu), read_conllu(a.test_conllu)
    maps = {name: fit(tr, k) for name, k in KEYS.items()}
    fallback = collections.Counter(x for _, _, x, _ in tr).most_common(1)[0][0]

    def lookup(form, upos, feats, ladder=LADDER):
        for step in ladder:
            k = KEYS[step](form, upos, feats)
            if k in maps[step]:
                return maps[step][k]
        return fallback

    print(f"train {len(tr)} tokens / {len({x for _, _, x, _ in tr})} XPOS types; test {len(te)}")
    print("\ngold-feature maps (majority class, fitted on train, scored on test)")
    for name in ("form", "form+upos", "form+upos+feats", "upos+feats"):
        lad = {"form": ["form"], "form+upos": ["form+upos", "form"],
               "form+upos+feats": LADDER, "upos+feats": ["upos+feats"]}[name]
        hit = sum(lookup(f, u, m, lad) == x for f, u, x, m in te)
        print(f"  {name:18s} {100*hit/len(te):6.2f}")

    if not a.model:
        return
    if not a.test_spacy:
        raise SystemExit("--model needs --test-spacy (the corpus the arm is scored on)")

    import seg_code  # noqa: F401   custom tokenisers, readers and factories
    import spacy
    from spacy.tokens import Doc, DocBin

    nlp = spacy.load(a.model)
    gold = list(DocBin().from_disk(a.test_spacy).get_docs(nlp.vocab))

    n = tagger_ok = gold_ok = pred_ok = union_ok = fixable = 0
    for g in gold:
        # gold tokenisation, predicted annotation -- the same contract as --gold-preproc
        p = nlp(Doc(nlp.vocab, words=[t.text for t in g],
                    spaces=[t.whitespace_ == " " for t in g]))
        for tg, tp in zip(g, p):
            if not tg.tag_:
                continue
            n += 1
            t_ok = tp.tag_ == tg.tag_
            g_ok = lookup(tg.text, tg.pos_, str(tg.morph)) == tg.tag_
            p_ok = lookup(tg.text, tp.pos_, str(tp.morph)) == tg.tag_
            tagger_ok += t_ok; gold_ok += g_ok; pred_ok += p_ok
            union_ok += t_ok or p_ok
            fixable += (not t_ok) and p_ok

    f = lambda v: f"{100*v/n:6.2f}"
    print(f"\nagainst {a.model}  (n={n})")
    print(f"  released tagger                      {f(tagger_ok)}")
    print(f"  map on GOLD upos+feats (+form)       {f(gold_ok)}")
    print(f"  map on PREDICTED upos+feats (+form)  {f(pred_ok)}   <- what conditioning can reach")
    print(f"  either one right (union ceiling)     {f(union_ok)}")
    print(f"  tagger wrong but map right           {f(fixable)}   ({fixable} tokens)")


if __name__ == "__main__":
    main()
