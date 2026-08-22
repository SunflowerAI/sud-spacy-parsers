#!/usr/bin/env python3
"""Strict whole-token P/R/F for a character segmenter, on a `make_seg_pairs.py` split.

This is the metric quoted for zh throughout CLAUDE.md (pkuseg 0.8385 -> char tagger 0.8902): a
predicted token counts only if BOTH its edges match a gold token's, scored over character offsets
within each whitespace chunk. `eval_samhita.score` reports split-LOCATION F instead, which is the
right early-stopping signal but a more forgiving number — a chunk with one wrong boundary loses two
tokens here and one boundary there.

    eval_zh_seg.py MODEL_DIR data_seg_zh/test.jsonl --lexicon A.txt B.txt --min-lens 1 1 \
        [--jieba-source 1] [--jieba-userdict WORDS.txt] [--compare jieba]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BREAK = "= "


def spans(words):
    out, i = [], 0
    for w in words:
        out.append((i, i + len(w)))
        i += len(w)
    return set(out)


def token_prf(pred, gold):
    tp = fp = fn = 0
    for pw, gw in zip(pred, gold):
        ps, gs = spans(pw), spans(gw)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def words_from_labels(chunk, labels):
    out, cur = [], ""
    for ch, lb in zip(chunk, labels):
        cur += ch
        if lb == BREAK:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--lexicon", nargs="*", default=[])
    ap.add_argument("--min-lens", nargs="*", type=int, default=None)
    ap.add_argument("--jieba-source", type=int, default=None)
    ap.add_argument("--jieba-userdict", default=None)
    ap.add_argument("--compare-jieba", action="store_true",
                    help="also score jieba alone on the same rows, as a reference point")
    ap.add_argument("--batched", action="store_true",
                    help="predict all rows in ONE call. Do not use this to report a number. The "
                         "encoder is `with_array(... expand_window ...)`, so thinc concatenates the "
                         "batch and the FIRST CHARACTER of each row sees its neighbour instead of "
                         "zero padding. The effect is confined to that character (|delta| 0.81 at "
                         "position 0, ~0 by position 2), but zh's sentence-initial split is "
                         "genuinely uncertain, so it is worth 0.27 F here: 0.9229 batched vs 0.9202 "
                         "as deployed, 60 of 529 rows differing, every one a sentence-initial "
                         "split. `CharSegTokenizer.__call__` predicts one text at a time, so the "
                         "default matches deployment. On id/yue/sa the same comparison moves "
                         "NOTHING — see scripts/eval_seg_batching.py.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    chunks = [r["samhita"] for r in rows]
    gold = [r["csl"].split(" ") for r in rows]

    import sa_presegment_lex as spl
    # Read the channel's settings off the MODEL, not off the command line: a segmenter trained to
    # ask jieba about the t2s rendering has to be asked the same way here, and a flag that has to be
    # remembered is a flag that will be forgotten. --jieba-source still overrides, for probing.
    meta = json.loads((pathlib.Path(a.model) / "vocab.json").read_text(encoding="utf-8"))
    src = a.jieba_source if a.jieba_source is not None else meta.get("jieba_source")
    if src is not None:
        import zh_jieba_feature as jf
        dict_path = pathlib.Path(a.model) / jf.TRAD_DICT_FILE if meta.get("jieba_dict") else None
        if dict_path is not None and not dict_path.is_file():
            raise SystemExit(f"{a.model} was trained against the {meta['jieba_dict']} jieba "
                             f"dictionary but {jf.TRAD_DICT_FILE} is not beside its weights")
        spl.enable_jieba(src, a.jieba_userdict, t2s=meta.get("jieba_t2s", False),
                         dict_path=dict_path)
    mins = a.min_lens or [3] * len(a.lexicon)
    lex = [{w for w in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if len(w) >= ml}
           for p, ml in zip(a.lexicon, mins)]
    if lex:
        seg = spl.LexPresegmenter.from_disk(pathlib.Path(a.model), lex)
        seg.min_lens = mins
    else:
        from sa_presegment import Presegmenter
        seg = Presegmenter.from_disk(pathlib.Path(a.model))

    if a.batched:
        labels = seg.predict(chunks)
    else:
        labels = [seg.predict([c])[0] for c in chunks]
    pred = [words_from_labels(c, lb) for c, lb in zip(chunks, labels)]
    p, r, f = token_prf(pred, gold)
    mode = "batched (NOT deployment)" if a.batched else "per text (as deployed)"
    print(f"{a.model}  ({len(rows)} chunks, {sum(len(g) for g in gold)} gold tokens)  {mode}")
    print(f"  strict token   P {p:.4f}   R {r:.4f}   F {f:.4f}")

    if a.compare_jieba:
        # THE MODEL'S OWN CHANNEL, not some other jieba. This line used to cut the raw text with a
        # stock tokenizer, which on a traditional arm answered a question the model was never
        # asked: it reported F 0.7452 for a channel worth 0.7984, because `--jieba-t2s` converts
        # inside the CODE function and leaves the tokenizer alone. Reading the codes back into
        # words is regime-proof — t2s, a traditional dictionary or neither, it scores what the
        # model was actually fed.
        import zh_jieba_feature as jf
        fn, tok = spl._JIEBA["fn"], spl._JIEBA["tok"]
        if fn is None:
            tok = jf.get_tokenizer(a.jieba_userdict)
            jb = [list(tok.cut(c, HMM=True)) for c in chunks]
        else:
            jb = [words_from_labels(c, [BREAK if k in (jf.E, jf.S) else "=" for k in fn(c, tok)])
                  for c in chunks]
        p, r, f = token_prf(jb, gold)
        print(f"  jieba alone    P {p:.4f}   R {r:.4f}   F {f:.4f}")


if __name__ == "__main__":
    main()
