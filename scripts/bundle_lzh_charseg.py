#!/usr/bin/env python3
"""Swap the trained character segmenter into the released lzh arm.

WHY. The shipped lzh tokeniser is "one Han character = one token", but the Kyoto treebank is not:
2.49 % of test tokens are multi-character (君子 孔子 孟子 匈奴 契丹 五十), so the released tokeniser
splits 孔子 into 孔 + 子. Measured on the test set, strict token F:

    split every character (what ships)   P 0.9500  R 0.9751  F 0.9624   84.7 % sentences perfect
    trained segmenter                    P 0.9827  R 0.9843  F 0.9835   92.9 % sentences perfect

NO RETRAIN IS NEEDED. `gold_preproc` plus `sud.GoldTokCorpus.v1` make the parser segmenter-agnostic,
which is exactly why the Latin enclitic tokeniser could be swapped into a released arm with all seven
component weight files coming out byte-identical. The same argument applies here -- and this script
CHECKS it rather than asserting it.

⚠ ASSIGNING `nlp.tokenizer` DOES NOT UPDATE THE CONFIG. `to_disk` writes the config as it stands, so
a reloaded model rebuilds whatever the config names, `from_disk` quietly refills it, and the result
loads, runs, splits nothing and says nothing. `nlp.config["nlp"]["tokenizer"]` must be set too, and
the verification below reloads FROM DISK rather than trusting the in-memory object -- the in-memory
one is right in exactly the case where the artefact is wrong.

    bundle_lzh_charseg.py --src training_lzh_rm_morph/model-best --seg models/lzh_seg_char \\
        --out build_lzh_charseg
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_code                                    # noqa: E402,F401  (registers the tokenizers)
import spacy                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # ⚠ BOTH DEFAULTS PREVIOUSLY NAMED A SUPERSEDED ARM. lzh went traditional-only end to end, so
    # `training_lzh_rm_morph` is the both-scripts generation and `models/lzh_seg_char` is a
    # segmenter trained on both scripts. Neither is detectable from the weights. Standing hazard 2:
    # a default that names the right arm is the fix; a comment telling the next person is not.
    ap.add_argument("--src", default="training_lzh_trad_sud/model-best")
    ap.add_argument("--seg", default="models/lzh_seg_char_trad")
    ap.add_argument("--expect-corpus", default="data_seg_lzh_trad",
                    help="refuse a segmenter stamped with a different training corpus; "
                         "pass '' to skip (only when knowingly packaging another generation)")
    ap.add_argument("--lexicon", default=None,
                    help="optional word list for the lexicon channel; lzh trained without one")
    ap.add_argument("--out", default="build_lzh_charseg")
    ap.add_argument("--verify", action="store_true", help="check component weights are unchanged")
    args = ap.parse_args()

    from char_seg_tokenizer import CharSegTokenizer

    if args.expect_corpus:
        meta = json.loads((pathlib.Path(args.seg) / "vocab.json").read_text(encoding="utf-8"))
        got = meta.get("corpus")
        if got != args.expect_corpus:
            raise SystemExit(f"REFUSING: {args.seg} was trained on {got!r}, expected "
                             f"{args.expect_corpus!r}. lzh ships TRADITIONAL-ONLY and a "
                             f"both-scripts segmenter is indistinguishable by any weight check.")

    nlp = spacy.load(args.src)
    before = {n: p.model.to_bytes() for n, p in nlp.pipeline if hasattr(p, "model")}
    tok = CharSegTokenizer(nlp.vocab)
    tok.load_segmenter(args.seg, lexicon=args.lexicon)
    nlp.tokenizer = tok
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.CharSegTokenizer.v1"}
    nlp.to_disk(args.out)
    print(f"  wrote {args.out}  (pipeline {nlp.pipe_names})")

    # RELOAD FROM DISK. The whole point of the config gotcha above is that the in-memory object is
    # correct while the artefact is not, so nothing is verified until it has been round-tripped.
    rl = spacy.load(args.out)
    kind = type(rl.tokenizer).__name__
    print(f"  reloaded tokenizer: {kind}")
    if kind != "CharSegTokenizer":
        raise SystemExit(f"REFUSING: reloaded model rebuilt a {kind}, not the segmenter")

    probe = "子曰學而時習之不亦說乎"
    print(f"  {probe} -> {' '.join(t.text for t in rl(probe))}")
    if all(len(t.text) == 1 for t in rl(probe)):
        print("  ⚠ every token is one character on this probe -- may be correct here, but check a "
              "sentence containing a known multi-character token (孔子, 君子, 匈奴)")
    for s in ("孔子曰", "君子不器", "匈奴大入上郡"):
        print(f"  {s} -> {' '.join(t.text for t in rl(s))}")

    if args.verify:
        after = {n: p.model.to_bytes() for n, p in rl.pipeline if hasattr(p, "model")}
        bad = [n for n in before if before.get(n) != after.get(n)]
        print("  component weights: " +
              ("ALL BYTE-IDENTICAL" if not bad else f"CHANGED in {bad} -- investigate"))
        if bad:
            raise SystemExit(1)

    mp = pathlib.Path(args.out) / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["requirements"] = sorted(set(m.get("requirements") or []))
        mp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
