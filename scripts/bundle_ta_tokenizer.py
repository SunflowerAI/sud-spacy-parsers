#!/usr/bin/env python3
"""Swap the trained Tamil sandhi tokeniser into an arm and save it self-contained.

The ta arms train through `sud.GoldTokCorpus.v1`, which gives the parser GOLD tokens and makes it
segmenter-agnostic — so the arm that comes out of `spacy train` still carries spaCy's rule
tokeniser and cannot produce the treebank's multiword tokens at all (strict token F 0.8389).
This puts `sud.TamilSandhiTokenizer.v1` in its place, with the character segmenter trained by
`scripts/train_ta_charseg.sh` bundled beside the weights (0.9420).

⚠ **ASSIGNING `nlp.tokenizer` DOES NOT UPDATE THE CONFIG.** `nlp.config["nlp"]["tokenizer"]` has to
be set too, or the saved model names `spacy.Tokenizer.v1` in its config, rebuilds THAT on load, and
silently tokenises on whitespace — CLAUDE.md standing hazard 8, which is how a zh wheel once
shipped returning one token per input string.

⚠ **AND VERIFY THE RELOADED MODEL, NEVER THE IN-MEMORY ONE.** A `CharSegTokenizer` with no
segmenter loads cleanly, runs, splits nothing and says nothing. So this reloads from disk and
asserts on a word the treebank splits.

    bundle_ta_tokenizer.py training_ta_both_lemvec_xw build_ta_final
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import seg_code  # noqa: E402,F401  (registers every custom factory the arms name)
import ta_tokenizer  # noqa: E402,F401  (registers sud.TamilSandhiTokenizer.v1)
import spacy  # noqa: E402

#: Splits as நிலையத்துக்குக்க் + ஆன in TTB — a seam that falls INSIDE the character கா, so a
#: whitespace tokeniser cannot produce it and a probe on it cannot pass by accident.
PROBE = "நிலையத்துக்குக்கான"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--segmenter", default="models/ta_seg_char")
    args = ap.parse_args()

    nlp = spacy.load(args.in_model)
    tok = ta_tokenizer.TamilSandhiTokenizer(nlp.vocab)
    tok.load_segmenter(args.segmenter)
    nlp.tokenizer = tok
    # the half that assignment does not do
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.TamilSandhiTokenizer.v1"}
    nlp.to_disk(args.out_model)

    reloaded = spacy.load(args.out_model)
    got = [t.text for t in reloaded(PROBE)]
    if len(got) < 2:
        print(f"!! {args.out_model}: the segmenter did not survive the round trip — "
              f"{PROBE!r} came back as {got!r}", file=sys.stderr)
        return 1
    cfg = reloaded.config["nlp"]["tokenizer"].get("@tokenizers")
    if cfg != "sud.TamilSandhiTokenizer.v1":
        print(f"!! {args.out_model}: config still names {cfg!r}", file=sys.stderr)
        return 1
    doc = reloaded("சென்னை அருகே ஸ்ரீ பெரும்புதூரில் நிலம் எடுக்கப் படும்.")
    print(f"wrote {args.out_model}")
    print(f"  tokenizer   {cfg}")
    print(f"  probe       {PROBE} -> {' + '.join(got)}")
    print(f"  pipeline    {reloaded.pipe_names}")
    print(f"  parse       {[(t.text, t.dep_, t.head.text) for t in doc][:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
