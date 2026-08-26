#!/usr/bin/env python3
"""Can a learned language embedding be predicted from Grambank features?

If it can, the adaptation result becomes deployable with NO annotation at all: look the language up
in Grambank, predict its row, parse. This script asks that question in the only order that can
answer it honestly --

  1. fit on the TRAINING languages' learned rows, with leave-one-out cross-validation;
  2. compare held-out predictions against the MEAN embedding, which is what you get for free;
  3. only if the regression beats that baseline is anything installed in a model and scored.

⚠ **n IS SMALL AND p IS NOT.** 51 training languages have Grambank data; ~106 features are coded for
90 % of them; the target is 128-dimensional. Ridge with a cross-validated penalty is the most this
supports, and the mean-embedding baseline is the number that decides whether it supports even that.
A regression that cannot beat the mean under LOO is not a weak predictor, it is no predictor.
"""
import argparse, collections, csv, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generic_code_v2  # noqa: F401
import spacy

ALIAS = {"ara": "arb", "fas": "pes", "est": "ekk", "ori": "ory", "uzb": "uzn", "yid": "ydd",
         "msa": "zsm", "zho": "cmn", "nor": "nob", "swa": "swh", "aze": "azj", "grn": "gug",
         "kur": "kmr", "que": "quy", "mlg": "plt", "srp": "hbs", "hrv": "hbs", "bos": "hbs"}


def embedding_table(model):
    nlp = spacy.load(model)
    for _, proc in nlp.pipeline:
        m = getattr(proc, "model", None)
        if m is None:
            continue
        slots = None
        for nd in m.walk():
            if nd.name == "extract_lang_slot":
                slots = dict(nd.attrs["ls_slots"])
        for nd in m.walk():
            if nd.name == "embed" and nd.has_param("E") and slots is not None:
                return np.array(nd.get_param("E")), slots
    sys.exit("no language-embedding table in this model")


def grambank(cache, min_cov, langs_iso):
    gl = {r["ISO639P3code"]: r["ID"] for r in
          csv.DictReader(open(f"{cache}/glottolog_languages.csv", encoding="utf-8"))
          if r.get("ISO639P3code")}
    vals = collections.defaultdict(dict)
    for v in csv.DictReader(open(f"{cache}/grambank_values.csv", encoding="utf-8")):
        if v["Value"] in ("0", "1"):
            vals[v["Language_ID"]][v["Parameter_ID"]] = float(v["Value"])
    got = {}
    for lg, iso in langs_iso.items():
        g = gl.get(iso) or gl.get(ALIAS.get(iso, ""))
        if g and g in vals:
            got[lg] = vals[g]
    cov = collections.Counter()
    for d in got.values():
        # `.update(dict)` ADDS THE VALUES, so a feature coded 0 everywhere counted as uncovered and
        # the filter returned nothing. Count KEYS.
        cov.update(d.keys())
    feats = sorted(p for p, c in cov.items() if c / max(len(got), 1) >= min_cov)
    X = {}
    for lg, d in got.items():
        X[lg] = np.array([d.get(f, np.nan) for f in feats], dtype="f8")
    # Impute a missing value with the feature's mean over the languages that have it: a coded 0 and
    # an uncoded cell are different things, and zero-filling would assert the former.
    M = np.vstack(list(X.values()))
    means = np.nanmean(M, axis=0)
    for lg in X:
        m = np.isnan(X[lg])
        X[lg][m] = means[m]
    return X, feats


def ridge_fit(X, Y, alpha):
    n, p = X.shape
    Xc = np.hstack([X, np.ones((n, 1))])
    A = Xc.T @ Xc + alpha * np.eye(p + 1)
    A[-1, -1] -= alpha                      # do not penalise the intercept
    return np.linalg.solve(A, Xc.T @ Y)


def predict(W, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--cache", default="assets_typ/cldf")
    ap.add_argument("--min-cov", type=float, default=0.90)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[1, 3, 10, 30, 100, 300, 1000, 3000])
    ap.add_argument("--out", default="assets_typ/predicted_embeddings.json")
    a = ap.parse_args()

    E, slots = embedding_table(a.model)
    inv = {c["lcode"] or c["lang_name"]: c for c in
           json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))["corpora"]}
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    iso = {k: inv[k]["iso3"] for k in man}
    X, feats = grambank(a.cache, a.min_cov, iso)
    print(f"{len(feats)} Grambank features coded for >= {a.min_cov:.0%} of the languages that have any")

    train = sorted(k for k in slots if man.get(k, {}).get("pool") == "train" and k in X)
    print(f"fitting on {len(train)} training languages with both an embedding and Grambank data")
    Xt = np.vstack([X[k] for k in train])
    Yt = np.vstack([E[slots[k]] for k in train])
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt = (Xt - mu) / sd

    print(f"\n{'alpha':>7s} {'LOO cosine':>11s} {'mean-baseline':>14s} {'gain':>7s}")
    best = (None, -9)
    for al in a.alphas:
        cs, bs = [], []
        for i in range(len(train)):
            m = np.ones(len(train), bool); m[i] = False
            W = ridge_fit(Xt[m], Yt[m], al)
            p = predict(W, Xt[~m])[0]
            cs.append(cos(p, Yt[i]))
            bs.append(cos(Yt[m].mean(0), Yt[i]))
        c, b = float(np.mean(cs)), float(np.mean(bs))
        print(f"{al:7.0f} {c:11.3f} {b:14.3f} {c - b:+7.3f}")
        if c > best[1]:
            best = (al, c, b)
    al, c, b = best
    print(f"\nbest alpha {al:.0f}: LOO cosine {c:.3f} vs mean baseline {b:.3f}  ({c - b:+.3f})")
    if c <= b + 0.01:
        print("\n⚠ THE REGRESSION DOES NOT BEAT THE MEAN. Grambank does not predict these "
              "embeddings, and installing its predictions would only be a noisier way of "
              "installing the mean. Not writing predictions.")
        return

    W = ridge_fit(Xt, Yt, al)
    test = sorted(k for k in man if man[k]["pool"] == "test" and k in X)
    P = predict(W, (np.vstack([X[k] for k in test]) - mu) / sd)
    json.dump({"meta": {"alpha": al, "loo_cosine": c, "mean_baseline": b,
                        "n_features": len(feats), "n_train": len(train),
                        "model": a.model},
               "languages": {k: P[i].tolist() for i, k in enumerate(test)}},
              open(a.out, "w"), indent=1)
    print(f"wrote predictions for {len(test)} test languages -> {a.out}")


if __name__ == "__main__":
    main()
