#!/usr/bin/env python3
"""Lexicon-aware beam decoding for the CSLiser, scored against plain greedy.

WHY A BEAM IS NEEDED AT ALL. `sa_presegment` is non-autoregressive: `Embed -> residual(expand_window
+ Maxout)xdepth -> Softmax` per position, with no recurrence and no label-transition matrix. The
joint distribution therefore factorises into independent per-position softmaxes, and per-position
argmax is the EXACT global MAP — a beam over the same factorised score cannot do better, only
slower. A beam becomes meaningful only once the score stops decomposing per position, which is what
a lexicon supplies: whether a completed member is a real Sanskrit stem is a property of a SPAN, not
of a character.

WHY TRY AGAIN, given the last one was removed. The previous reranker used a corpus-derived form list
and failed where it mattered — on genuinely unseen text it cost 0.32 sentence PM. Its coverage was
the problem in disguise: measured per compound member on classical/epic DCS, the corpus lexicon
covers only 48.0 % of non-final members. Apte's 128 872 stems cover 65.4 % ALONE and 76.2 % combined
— it nearly doubles the corpus figure on the target domain, because the training corpus is
Vedic-heavy and classical vocabulary is simply different.

The score is
    log P(labels | text)  +  lex_weight * (covered members) / (total members)
so the bonus is per-segmentation, normalised, and cannot be gamed by splitting more. Only non-final
members are looked up: they are bare stems, which is what a dictionary lists, whereas final members
carry inflection (corpus+Apte covers 76.2 % of the former against 31.8 % of the latter).

THE HYPERPARAMETER IS THE KNOWN TRAP. Weights tuned for a weak model cost the strong one 3.80 PM
last time, so `--tune` fits `lex_weight` on DEV and the reported figure is always TEST.

    lexicon_decode.py --model models/sa_presegment_dcs --lexicon models/apte_stems.txt \\
        --dev data_samhita/dcs_dev.jsonl --test data_samhita/dcs_test.jsonl --tune
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sa_presegment import Presegmenter          # noqa: E402

BREAKS = ("= ", "=-")


def load_rows(path):
    return [json.loads(l) for l in pathlib.Path(path).open(encoding="utf-8")]


def members(text, labels):
    """Split a chunk into its members by any break label (word or compound)."""
    out, cur = [], ""
    for ch, lb in zip(text, labels):
        cur += ch
        if lb in BREAKS:
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out


def segments(text, labels):
    """(surface, start, break_kind) per member — break_kind is '= ', '=-' or None at chunk end.

    The distinction matters and getting it wrong invalidated two experiments. A lexicon of Apte
    HEADWORDS is a list of STEMS, and only compound-internal non-final members are stems: measured
    on DCS gold, Apte covers 70.2 % of those, against 22.7 % of ordinary inflected running words
    (`bhavanti`, `vanād`) and 16.3 % of compound-FINAL members, which carry the inflection. Scoring
    or repairing over every member therefore tests running word-forms against a stem dictionary,
    where absence means nothing at all — and a repair triggered by it fires almost entirely on
    correct output.
    """
    out, cur, start = [], "", 0
    for i, (ch, lb) in enumerate(zip(text, labels)):
        cur += ch
        if lb in BREAKS:
            out.append((cur, start, lb)); cur = ""; start = i + 1
    if cur:
        out.append((cur, start, None))
    return out


def compound_nonfinal(text, labels, min_len=3):
    """Indices into `segments` that are bare stems: non-final members of a compound."""
    segs = segments(text, labels)
    return [k for k, (s, _st, br) in enumerate(segs)
            if br == "=-" and len(s.strip("'\"")) >= min_len]


def beam_decode(seg, text, lex, lex_weight, beam_size):
    """Beam over label sequences, scored by model logprob + a span-level lexicon bonus."""
    scores = seg.model.predict([seg.encode_chars(text)])[0]
    logp = np.log(np.clip(scores, 1e-12, None))
    logp -= logp.max(axis=1, keepdims=True)      # numerically safe; constant per position
    n_lab = logp.shape[1]
    # only the few best labels per position can matter; this keeps the beam cheap
    top = np.argsort(-logp, axis=1)[:, :max(3, beam_size)]

    beams = [([], 0.0)]
    for i in range(len(text)):
        nxt = []
        for labs, sc in beams:
            for j in top[i]:
                nxt.append((labs + [int(j)], sc + float(logp[i, j])))
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:beam_size]

    best, best_sc = None, -1e18
    for labs, sc in beams:
        names = [seg.labels[j] for j in labs]
        if lex_weight:
            segs = segments(text, names)
            idx = compound_nonfinal(text, names)
            if idx:
                hit = sum(segs[k][0].strip("'\"") in lex for k in idx)
                sc += lex_weight * hit / len(idx)
        if sc > best_sc:
            best, best_sc = names, sc
    return best


def score(rows, preds):
    """Split-location F and sentence perfect-match, against the gold labels."""
    tp = fp = fn = 0
    pm = 0
    for r, p in zip(rows, preds):
        g = [i for i, lb in enumerate(r["labels"]) if lb in BREAKS]
        q = [i for i, lb in enumerate(p) if lb in BREAKS]
        gs, qs = set(g), set(q)
        tp += len(gs & qs); fp += len(qs - gs); fn += len(gs - qs)
        pm += (r["labels"] == p)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return (2 * P * R / max(P + R, 1e-9)) * 100, pm / max(len(rows), 1) * 100


def repair_decode(seg, text, lex, min_len, max_edits):
    """Greedy, then LOCAL repair only where greedy produced a non-word.

    This is not the beam by another name, and the difference is the point. A beam explores the
    sequences the MODEL ranks highly, so when the model is confident and wrong the correct
    segmentation is never in the beam at all. Here the trigger is the lexicon instead: a non-final
    member that is not an attested stem is evidence something is wrong REGARDLESS of model score,
    and the edit that fixes it can be one the beam would never have reached.

    Deliberately conservative — it fires only when the current member is unattested AND the
    replacement makes every affected member attested, so it can only ever move a segmentation from
    "contains a non-word" to "all words". Short members are skipped (`min_len`): 1-2 character
    strings are attested as stems almost by accident and carry no evidence either way.
    """
    labels = list(seg.predict([text])[0])
    for _ in range(max_edits):
        ms = members(text, labels)
        if len(ms) < 2:
            break
        # character offset where each member starts
        starts, off = [], 0
        for m in ms:
            starts.append(off); off += len(m)
        target = None
        for k in compound_nonfinal(text, labels, min_len):
            if ms[k].strip("'\"") not in lex:
                target = k; break
        if target is None:
            break
        m = ms[target]; base = starts[target]
        cand = []
        # (a) split this member in two, both halves attested
        for cut in range(min_len, len(m.strip("-")) - min_len + 1):
            left, right = m.strip("-")[:cut], m.strip("-")[cut:]
            if left in lex and right in lex:
                cand.append(("split", cut))
        # (b) merge with the NEXT member (the break after it was spurious)
        if target + 1 < len(ms):
            merged = (m + ms[target + 1]).strip("-")
            if merged in lex:
                cand.append(("merge_next", 0))
        # (c) merge with the PREVIOUS member
        if target > 0:
            merged = (ms[target - 1] + m).strip("-")
            if merged in lex:
                cand.append(("merge_prev", 0))
        if not cand:
            break
        kind, cut = cand[0]
        if kind == "split":
            i = base + cut - 1
            if labels[i] not in BREAKS:
                labels[i] = "=-"                    # a new break inside a member is a COMPOUND break
        elif kind == "merge_next":
            i = base + len(m) - 1
            if labels[i] in BREAKS:
                labels[i] = "="
        else:
            i = base - 1
            if 0 <= i < len(labels) and labels[i] in BREAKS:
                labels[i] = "="
    return labels


def run(seg, rows, lex, w, beam, mode="beam", min_len=3, max_edits=3):
    if mode == "repair":
        return [repair_decode(seg, r["samhita"], lex, min_len, max_edits) for r in rows]
    return [beam_decode(seg, r["samhita"], lex, w, beam) if w else
            seg.predict([r["samhita"]])[0] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sa_presegment_dcs")
    ap.add_argument("--lexicon", default="models/apte_stems.txt")
    ap.add_argument("--dev", default="data_samhita/dcs_dev.jsonl")
    ap.add_argument("--test", default="data_samhita/dcs_test.jsonl")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--lex-weight", type=float, default=1.0)
    a = ap.parse_args()

    seg = Presegmenter.from_disk(a.model)
    lex = set(pathlib.Path(a.lexicon).read_text(encoding="utf-8").split("\n"))
    print(f"  lexicon {len(lex)} stems, model {a.model}")

    dev = load_rows(a.dev)[:a.limit]
    test = load_rows(a.test)[:a.limit]

    weight = a.lex_weight
    if a.tune:
        print(f"\n  tuning lex_weight on DEV ({len(dev)} rows) -- the reported figure is TEST")
        best = (-1, None)
        for w in (0.0, 0.5, 1.0, 2.0, 4.0):
            f, pm = score(dev, run(seg, dev, lex, w, a.beam))
            print(f"    w={w:<4} split-loc F {f:6.2f}   sentence PM {pm:6.2f}")
            if pm > best[0]:
                best = (pm, w)
        weight = best[1]
        print(f"  -> chose lex_weight={weight}")

    print(f"\n  TEST ({len(test)} rows)")
    f0, pm0 = score(test, run(seg, test, lex, 0.0, a.beam))
    print(f"    greedy (exact MAP)      split-loc F {f0:6.2f}   sentence PM {pm0:6.2f}")
    f1, pm1 = score(test, run(seg, test, lex, weight, a.beam))
    print(f"    beam + Apte (w={weight})    split-loc F {f1:6.2f}   sentence PM {pm1:6.2f}")
    print(f"    delta                              {f1-f0:+6.2f}                {pm1-pm0:+6.2f}")


if __name__ == "__main__":
    main()
