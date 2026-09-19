#!/usr/bin/env python3
"""Train the DISTILLED direction classifier for sent_join.py's new `classifier_join` rule.

Same in-unit training pairs and symbolic features as scripts/analyze_direction_classifier_errors.py
(imported from there, not reimplemented), but the vector feature is the STATIC per-character-type
table already shipped in the v0.3.3 wheel (vectors_lzh_siku96.vec / vocab.vectors), not a live
SikuBERT forward pass -- so the deployed pipe needs no new runtime dependency and no new bundled
data file (`doc.vocab[char].vector` already resolves it in any loaded lzh model with the SikuBERT
channel). Root-token-only, matching the validated live-SikuBERT methodology (not mean-pooled).

A plain LogisticRegression on [StandardScaler-normalised features] is linear end to end, so the
scaler folds into the raw-space weight vector (w_raw = coef_/scale_, b_raw = intercept_ -
sum(coef_*mean_/scale_)): the shipped artifact is one weight array + one bias, no scaler state to
carry into scripts/sent_join.py's hand-rolled forward pass.
TRAINS ON POPULATION A ONLY (in-unit pairs, gold-labelled, `build_pairs()`) -- see the
`lzh-sentjoin-classifier-glue` memory for the full four-round history of why, but the short version:
an earlier round diagnosed the classifier firing on `pause_join`'s real cross-sentence merges (a
population A never covers) and tried FIXING IT BY RETRAINING on A + two new populations B
(cross-unit, same sent_group, labelled via cross_unit_rules.py's `decide()`) and C (cross-unit,
DIFFERENT sent_group, hard negative, y=0 always -- `scripts/build_sentjoin_pause_pairs.py` builds
B/C). That retrain measurably HURT: it miscalibrated genuine in-unit positives downward badly enough
that the flagship validated example (苟得其養，無物不長 -- textbook 苟-conditional) stopped firing
(p=0.40, below any defensible threshold), while only partially suppressing false positives on
marker-bearing population-C pairs (61%->36%, not clean either way). THE ACTUAL FIX was not a
different training population -- it was gating `classifier_join` in sent_join.py's
`_classifier_backward_proba` on a DECLARED grammatical marker (`GLUE_COND_MARKERS`) being present
before the learned score is even consulted, which structurally excludes most of what B/C were
trying to teach the model to reject statistically. With that gate in place, A's own well-calibrated,
gold-labelled population is the more defensible classifier to ship -- it is right about the
construction it was built for. `build_sentjoin_pause_pairs.py` and the B/C-combining code below are
kept for reference (the CLI still accepts them via `--include-bc`) but are NOT the shipped default.
"""
import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from analyze_direction_classifier_errors import BOOL_COLS, COND_MARKERS, NUM_COLS, build_pairs  # noqa: E402
from build_sentjoin_pause_pairs import build_pause_pairs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
VEC_PATH = ROOT / "vectors_lzh_siku96.vec"
OUT_PATH = ROOT / "scripts" / "lzh_sentjoin_glue.json"
UPOS_MIN_COUNT = 20


def load_static_vectors(path):
    vecs = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        n, dim = (int(x) for x in header.split())
        for line in f:
            parts = line.rstrip("\n").split(" ")
            key = parts[0]
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            assert vals.shape[0] == dim, (key, vals.shape)
            vecs[key] = vals
    return vecs, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-bc", action="store_true",
                    help="also train on populations B/C (NOT the shipped default -- see module "
                         "docstring for why this measurably hurt the flagship validated example)")
    args = ap.parse_args()

    print("=== 1a. building in-unit pairs (population A, disjoint spans) ===")
    pairs_a = build_pairs()
    for p in pairs_a:
        p["config"] = "A"
    print(f"  A: n = {len(pairs_a)}  backward = {sum(p['backward'] for p in pairs_a)}")

    if args.include_bc:
        print("=== 1b. building cross-unit pause-boundary pairs (populations B, C) ===")
        pairs_bc, bc_counts = build_pause_pairs()
        print(f"  B (same sent_group): {bc_counts['B']}  positive = {bc_counts['B_pos']} "
              f"({100*bc_counts['B_pos']/max(bc_counts['B'],1):.1f}%)")
        print(f"  C (different sent_group, hard negative): {bc_counts['C']}  positive = {bc_counts['C_pos']}")
        pairs = pairs_a + pairs_bc
    else:
        pairs = pairs_a
    print(f"  combined: n = {len(pairs)}  backward = {sum(p['backward'] for p in pairs)} "
          f"({100*sum(p['backward'] for p in pairs)/len(pairs):.1f}%)")

    print("=== 2. static SikuBERT vectors (root character only, per-pair diff) ===")
    static_vecs, dim = load_static_vectors(VEC_PATH)
    print(f"  table: {len(static_vecs)} types, dim={dim}")
    zero = np.zeros(dim)
    hit = 0
    for p in pairs:
        a_char = p["_a_chars"][p["_a_root_idx"]]
        b_char = p["_b_chars"][p["_b_root_idx"]]
        a_vec = static_vecs.get(a_char, zero)
        b_vec = static_vecs.get(b_char, zero)
        hit += (a_char in static_vecs) + (b_char in static_vecs)
        p["_diff_vec"] = b_vec - a_vec
    print(f"  root-char vector hit rate: {hit}/{2*len(pairs)} ({100*hit/(2*len(pairs)):.1f}%)")

    print("=== 3. build feature matrix ===")
    upos_counts = Counter(p["b_first_upos"] for p in pairs)
    upos_cats = sorted(u for u, c in upos_counts.items() if c >= UPOS_MIN_COUNT)
    print(f"  b_first_upos categories (count>={UPOS_MIN_COUNT}): {upos_cats}")

    def upos_onehot(u):
        return [1.0 if u == cat else 0.0 for cat in upos_cats]

    feature_names = list(BOOL_COLS) + list(NUM_COLS) + [f"b_first_upos={c}" for c in upos_cats] + \
        [f"sk{i}" for i in range(dim)]

    rows = []
    for p in pairs:
        row = [float(p[c]) for c in BOOL_COLS] + [float(p[c]) for c in NUM_COLS] + \
            upos_onehot(p["b_first_upos"]) + p["_diff_vec"].tolist()
        rows.append(row)
    X = np.array(rows, dtype=np.float64)
    y = np.array([1 if p["backward"] else 0 for p in pairs], dtype=np.int64)
    print(f"  X shape {X.shape}")

    print("=== 4. 5-fold CV, distilled logistic regression ===")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy")
    majority = 100 * max(y.mean(), 1 - y.mean())
    print(f"  CV accuracy: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}  (majority {majority:.2f}%)")
    print("  reference (RF + LIVE SikuBERT, analyze_direction_classifier_errors.py): 96.54% +/- 0.73")

    # symbolic-only ablation, for an honest "what does the vector actually buy" comparison
    n_sym = len(BOOL_COLS) + len(NUM_COLS) + len(upos_cats)
    scores_sym = cross_val_score(LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0),
                                  Xs[:, :n_sym], y, cv=skf, scoring="accuracy")
    print(f"  CV accuracy, symbolic features only (no vector): {scores_sym.mean()*100:.2f}% +/- {scores_sym.std()*100:.2f}")

    print("=== 4b. threshold sweep for the BACKWARD call (out-of-fold probabilities) ===")
    # False positive here (calling backward when it is really forward) actively REVERSES an edge the
    # existing default branch already gets right 95.2% of the time (measured separately: backward=
    # True->mod is a clean 1921/1921, but backward=False->parataxis is only 2461/2586); false negative
    # just leaves the existing default in place. The two error costs are asymmetric, so the shipped
    # threshold should favour PRECISION on the backward call, not accuracy.
    proba = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0),
                               Xs, y, cv=skf, method="predict_proba")[:, 1]
    print(f"  {'thresh':>7} {'n_pred_bwd':>10} {'precision':>9} {'recall':>7}")
    chosen_threshold = None
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
        pred = proba >= t
        n = pred.sum()
        prec = (y[pred] == 1).mean() if n else float("nan")
        rec = ((y == 1) & pred).sum() / max((y == 1).sum(), 1)
        print(f"  {t:7.2f} {n:10d} {prec:9.3f} {rec:7.3f}")
        if prec >= 0.97 and chosen_threshold is None:
            chosen_threshold = t
    if chosen_threshold is None:
        chosen_threshold = 0.9
    print(f"  chosen threshold (first to clear 0.97 precision, else 0.9 fallback): {chosen_threshold}")

    print("=== 4c. per-population breakdown (out-of-fold), the number that actually matters ===")
    configs = np.array([p["config"] for p in pairs])
    pred_at_t = proba >= chosen_threshold
    print(f"  {'config':>8} {'n':>7} {'pos_rate':>9} {'n_fired':>8} {'fired_precision':>16} {'fired_recall':>7}")
    for cfg in ("A", "B", "C"):
        m = configs == cfg
        n = int(m.sum())
        if n == 0:
            continue
        pos_rate = y[m].mean()
        fired = pred_at_t & m
        n_fired = int(fired.sum())
        prec = (y[fired] == 1).mean() if n_fired else float("nan")
        rec = ((y == 1) & fired).sum() / max((y[m] == 1).sum(), 1)
        print(f"  {cfg:>8} {n:>7} {pos_rate:9.3f} {n_fired:>8} {prec:16.3f} {rec:7.3f}")

    print("=== 5. fit on all data, fold scaler into raw-space weights ===")
    clf.fit(Xs, y)
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    w_raw = coef / scaler.scale_
    b_raw = intercept - np.sum(coef * scaler.mean_ / scaler.scale_)
    # sanity: raw-space forward pass must match the fitted pipeline's decision function exactly
    z_pipeline = Xs @ coef + intercept
    z_raw = X @ w_raw + b_raw
    assert np.allclose(z_pipeline, z_raw, atol=1e-6), "scaler fold-in mismatch"
    train_acc = ((1 / (1 + np.exp(-z_raw)) >= 0.5).astype(int) == y).mean()
    print(f"  raw-space fold-in verified (max abs diff {np.max(np.abs(z_pipeline - z_raw)):.2e}); "
          f"train accuracy {train_acc*100:.2f}%")

    out = {
        "feature_names": feature_names,
        "bool_cols": BOOL_COLS,
        "num_cols": NUM_COLS,
        "upos_categories": upos_cats,
        "cond_markers": sorted(COND_MARKERS),
        "vec_dim": dim,
        "weight": w_raw.tolist(),
        "bias": float(b_raw),
        "cv_accuracy": float(scores.mean()),
        "cv_accuracy_symbolic_only": float(scores_sym.mean()),
        "majority_baseline": float(majority) / 100,
        "n_pairs": len(pairs),
        "n_pairs_by_config": {"A": len(pairs_a), "B": bc_counts["B"], "C": bc_counts["C"]}
                             if args.include_bc else {"A": len(pairs_a)},
        "recommended_threshold": chosen_threshold,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
