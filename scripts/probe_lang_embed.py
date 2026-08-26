#!/usr/bin/env python3
"""What does the learned language embedding encode? Probe it for Grambank features.

The inverse of `predict_lang_embed.py`. That asked whether Grambank can PRODUCE an embedding, which
is the deployment question and which failed. This asks whether the embedding CONTAINS Grambank,
which is the interpretation question -- and it is the better-posed direction statistically, because
each target is a single binary feature rather than 128 correlated dimensions.

Method, per Grambank feature: leave-one-language-out logistic regression from the 128-d (or 8-d)
embedding to that feature, over the ~51 training languages that have both.

⚠ **THE MAJORITY BASELINE COMES FIRST, PER FEATURE.** Most Grambank features are heavily skewed --
"is there a definite article" is 0 for most languages in any sample -- so a probe that always
predicts the majority class already scores 80 %+ on many of them. This repo has reported 56.5 %
accuracy as a result against a 58.5 % constant once already.

⚠ **AND A PERMUTATION CONTROL, BECAUSE 106 FEATURES ARE TESTED AT ONCE.** With that many tests some
will beat their baseline by chance. The same probe is re-run with the language-to-embedding
assignment shuffled; the number of features that "pass" under shuffling is the number to discount.
A result is only interesting if the real count clears the shuffled count.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
import sud_generic_embed_v2  # noqa: E402,F401
from sklearn.linear_model import LogisticRegression  # noqa: E402

ALIAS = {"ara": "arb", "fas": "pes", "est": "ekk", "ori": "ory", "uzb": "uzn", "yid": "ydd",
         "msa": "zsm", "zho": "cmn", "nor": "nob", "swa": "swh", "aze": "azj", "grn": "gug",
         "kur": "kmr", "que": "quy", "mlg": "plt", "srp": "hbs", "hrv": "hbs", "bos": "hbs"}


def embedding_table(model):
    nlp = spacy.load(model)
    slots = E = None
    for _, proc in nlp.pipeline:
        m = getattr(proc, "model", None)
        if m is None:
            continue
        for nd in m.walk():
            if nd.name == "extract_lang_slot":
                slots = dict(nd.attrs["ls_slots"])
            if nd.name == "embed" and nd.has_param("E"):
                E = np.array(nd.get_param("E"))
    if slots is None or E is None:
        sys.exit("no language-embedding table in this model")
    return E, slots


def loo_accuracy(X, y, C):
    ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        if len(set(y[m])) < 2:
            ok += int(y[i] == y[m][0])
            continue
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X[m], y[m])
        ok += int(clf.predict(X[~m])[0] == y[i])
    return ok / len(y)


def run_probe(X, langs, feats, vals, C, min_minority, rng=None):
    """(passes, rows). `rng` shuffles the embedding-to-language assignment for the control."""
    Xs = X if rng is None else X[rng.permutation(len(X))]
    rows = []
    for f in feats:
        y = np.array([vals[lg][f] for lg in langs])
        n1 = int(y.sum())
        if min(n1, len(y) - n1) < min_minority:
            continue
        base = max(n1, len(y) - n1) / len(y)
        acc = loo_accuracy(Xs, y, C)
        rows.append((f, acc, base, acc - base, len(y)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--cache", default="assets_typ/cldf")
    ap.add_argument("--min-cov", type=float, default=0.90)
    ap.add_argument("--min-minority", type=int, default=10,
                    help="skip a feature unless both classes have at least this many languages")
    ap.add_argument("--C", type=float, default=0.05, help="inverse regularisation; small = strong")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="a feature counts as predicted only this far above its own baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    E, slots = embedding_table(a.model)
    inv = {c["lcode"] or c["lang_name"]: c for c in
           json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))["corpora"]}
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]

    gl = {r["ISO639P3code"]: r["ID"] for r in
          csv.DictReader(open(f"{a.cache}/glottolog_languages.csv", encoding="utf-8"))
          if r.get("ISO639P3code")}
    raw = collections.defaultdict(dict)
    for v in csv.DictReader(open(f"{a.cache}/grambank_values.csv", encoding="utf-8")):
        if v["Value"] in ("0", "1"):
            raw[v["Language_ID"]][v["Parameter_ID"]] = int(v["Value"])
    names = {r["ID"]: r["Name"] for r in
             csv.DictReader(open(f"{a.cache}/grambank_parameters.csv", encoding="utf-8"))} \
        if pathlib.Path(f"{a.cache}/grambank_parameters.csv").exists() else {}

    vals, langs = {}, []
    for lg in sorted(slots):
        if man.get(lg, {}).get("pool") != "train":
            continue
        iso = inv[lg]["iso3"]
        g = gl.get(iso) or gl.get(ALIAS.get(iso, ""))
        if g and g in raw:
            vals[lg] = raw[g]
            langs.append(lg)
    cov = collections.Counter()
    for lg in langs:
        cov.update(vals[lg].keys())
    feats = sorted(f for f, c in cov.items() if c / len(langs) >= a.min_cov)
    # Impute a gap with the feature's majority value over the languages that have it.
    for f in feats:
        present = [vals[lg][f] for lg in langs if f in vals[lg]]
        maj = int(round(sum(present) / len(present)))
        for lg in langs:
            vals[lg].setdefault(f, maj)

    X = np.vstack([E[slots[lg]] for lg in langs])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"{len(langs)} training languages, {len(feats)} Grambank features coded for "
          f">= {a.min_cov:.0%} of them, embedding width {X.shape[1]}")

    rows = run_probe(X, langs, feats, vals, a.C, a.min_minority)
    passes = [r for r in rows if r[3] >= a.margin]
    ctl_counts = []
    for s in range(5):
        rng = np.random.default_rng(a.seed + s)
        c = run_probe(X, langs, feats, vals, a.C, a.min_minority, rng)
        ctl_counts.append(sum(1 for r in c if r[3] >= a.margin))

    print(f"\n{len(rows)} features had both classes represented at least {a.min_minority} times")
    print(f"  beat their own majority baseline by >= {a.margin:.0%}:  {len(passes)}")
    print(f"  same, with the languages SHUFFLED (5 runs): {ctl_counts}  "
          f"mean {np.mean(ctl_counts):.1f}")
    verdict = ("ABOVE chance" if len(passes) > max(ctl_counts)
               else "NOT above chance -- the probe is fitting noise")
    print(f"  -> {verdict}")

    rows.sort(key=lambda r: -r[3])
    print(f"\n{'feature':9s} {'acc':>6s} {'base':>6s} {'gain':>6s}  name")
    for f, acc, base, gain, n in rows[:12]:
        print(f"  {f:7s} {acc:6.2f} {base:6.2f} {gain:+6.2f}  {names.get(f, '')[:58]}")

    if a.json:
        json.dump({"model": a.model, "n_langs": len(langs), "n_feats": len(rows),
                   "n_pass": len(passes), "shuffled_pass": ctl_counts,
                   "rows": [{"feature": f, "acc": acc, "baseline": base, "gain": g,
                             "name": names.get(f, "")} for f, acc, base, g, _ in rows]},
                  open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
