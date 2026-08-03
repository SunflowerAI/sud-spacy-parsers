#!/usr/bin/env python3
"""What the batching leak costs each character segmenter (sa / id / yue / zh).

Every model in this family encodes with `with_array(... expand_window ...)`, so thinc concatenates
the batch into ONE array and the FIRST CHARACTER of each row sees its neighbour rather than zero
padding. The effect is confined to that character (|delta| 0.81 at position 0, ~0 by position 2), so
whether it matters depends entirely on how uncertain the model is about a string's opening: measured,
it is worth 0.27 token F on zh and exactly 0.00 on id, yue and both sa CSLisers.

The DEPLOYMENT unit is one call per input string, and both runtime paths batch a string's whitespace
chunks together within that call:

    char_seg_tokenizer.CharSegTokenizer.__call__   preds = self.seg.predict(chunks)
    sa_tokenizer._cslise                           self.csliser.predict([c for c in chunks])

`make_seg_pairs.py` writes one row per CHUNK with `sent_id = "<sentence>_<chunk>"`, so grouping rows
by the sentence part reconstructs exactly that unit. For zh the sentence has no spaces and is a
single chunk, so per-sentence and per-row coincide; for id (rows are orthographic words) and for
spaced Sanskrit they do not.

    eval_seg_batching.py MODEL DATA --metric {token,samhita} [--lexicon ...] [--jieba-source K]
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BREAK = "= "


def load_seg(model, lexicon, min_lens, jieba_source):
    meta = json.loads((pathlib.Path(model) / "vocab.json").read_text(encoding="utf-8"))
    if meta.get("n_sources"):
        import sa_presegment_lex as spl
        if jieba_source is not None:
            spl.enable_jieba(jieba_source)
        mins = min_lens or [1] * len(lexicon)
        lex = [{w for w in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if len(w) >= m}
               for p, m in zip(lexicon, mins)]
        seg = spl.LexPresegmenter.from_disk(pathlib.Path(model), lex)
        seg.min_lens = mins
        return seg
    from sa_presegment import Presegmenter
    return Presegmenter.from_disk(pathlib.Path(model))


def predict_modes(seg, rows, group="row"):
    """Three prediction regimes over the same rows, differing ONLY in call grouping.

    `group="sentid"` is for `make_seg_pairs.py` data, where a row is one whitespace CHUNK and
    `sent_id` is `<sentence>_<chunk>` — there the deployment call covers a whole sentence's chunks.
    `group="row"` is for the Sanskrit pair files, where a row is already a whole input string (their
    `70340_1` suffix is a CLAUSE index, not a chunk index, so splitting on it would merge two
    separate inputs into one call).
    """
    texts = [r["samhita"] for r in rows]
    out = {"batched (whole test set in one call)": seg.predict(texts)}

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = str(r["sent_id"]).rsplit("_", 1)[0] if group == "sentid" else i
        groups[key].append(i)
    per_sent = [None] * len(rows)
    for idxs in groups.values():
        for i, lb in zip(idxs, seg.predict([texts[i] for i in idxs])):
            per_sent[i] = lb
    out["per sentence (AS DEPLOYED)"] = per_sent

    if len(groups) != len(rows):          # only informative when a sentence spans several chunks
        out["per row (fully isolated)"] = [seg.predict([t])[0] for t in texts]
    return out


def token_score(rows, labels):
    tp = fp = fn = 0
    for r, lbs in zip(rows, labels):
        pred, cur, i, ps = [], "", 0, set()
        for ch, lb in zip(r["samhita"], lbs):
            cur += ch
            if lb == BREAK:
                pred.append(cur); cur = ""
        if cur:
            pred.append(cur)
        for w in pred:
            ps.add((i, i + len(w))); i += len(w)
        gs, i = set(), 0
        for w in r["csl"].split(" "):
            gs.add((i, i + len(w))); i += len(w)
        tp += len(ps & gs); fp += len(ps - gs); fn += len(gs - ps)
    p = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
    return {"token F": 2 * p * r_ / max(p + r_, 1e-9), "token P": p, "token R": r_}


def samhita_score(rows, labels):
    from eval_samhita import score
    sc = score(rows, {r["sent_id"]: lb for r, lb in zip(rows, labels)})
    return {"split-loc F": sc["split_location"][2], "split-type F": sc["split_type"][2],
            "full-label F": sc["full_label"][2], "sentence PM": sc["sentence_pm"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--metric", choices=("token", "samhita"), default="token")
    ap.add_argument("--lexicon", nargs="*", default=[])
    ap.add_argument("--min-lens", nargs="*", type=int, default=None)
    ap.add_argument("--jieba-source", type=int, default=None)
    ap.add_argument("--group", choices=("row", "sentid"), default="row",
                    help="what one deployment call covers: a whole row (Sanskrit "
                         "pair files), or every chunk sharing a sent_id prefix "
                         "(make_seg_pairs data, where a row is one chunk)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    seg = load_seg(a.model, a.lexicon, a.min_lens, a.jieba_source)
    scorer = token_score if a.metric == "token" else samhita_score

    modes = predict_modes(seg, rows, a.group)
    keys = list(scorer(rows, next(iter(modes.values()))))
    print(f"{a.model}  x  {a.data}   ({len(rows)} rows)")
    print(f"  {'mode':38s}" + "".join(f"{k:>14s}" for k in keys))
    base = None
    for name, labels in modes.items():
        sc = scorer(rows, labels)
        if name.startswith("per sentence"):
            base = sc
        print(f"  {name:38s}" + "".join(f"{sc[k]:14.4f}" for k in keys))
    if base:
        for name, labels in modes.items():
            if name.startswith("per sentence"):
                continue
            sc = scorer(rows, labels)
            d = "".join(f"{(sc[k] - base[k]) * 100:+14.2f}" for k in keys)
            print(f"  {'  ^ minus deployment (pp)':38s}{d}   [{name.split(' ')[1]}]")


if __name__ == "__main__":
    main()
