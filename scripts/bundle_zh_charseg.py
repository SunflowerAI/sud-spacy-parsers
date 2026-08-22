#!/usr/bin/env python
"""Swap the treebank-trained character segmenter into the released Chinese model (post-hoc).

Same shape as `bundle_yue_pkuseg.py`, and for the same reason: the arm trains through
`sud.GoldTokCorpus.v1`, which builds every predicted doc from GOLD tokens, so training is
**segmenter-agnostic** and the raw-text tokeniser can be swapped in after the fact with no retrain.

The segmenter shipped here carries TWO input channels (`n_sources = 2`):

    0   the corpus word list, jackknifed        models/zh_lex_corpus.txt
    1   jieba's own SEGMENTATION DECISION       (BMES per character; see zh_jieba_feature.py)

Strict whole-token F on the GSD test, 5 runs each: 0.8761 for a zeroed second channel, 0.9203 with
jieba's decision in it, no overlap; raw end-to-end LAS 0.4673 -> 0.5269 over the same runs. The
shipped `models/zh_seg_jbdec` is the dev-selected run (test F 0.9210, raw LAS 0.5608 -- the top of
the range, so expect the mean from a retrain). NB raw LAS is much noisier than the token F driving
it, so do not quote it from a single training run.

**`jieba` therefore becomes a RUNTIME dependency** and is declared in `meta.json` — the lesson from
the ja wheel, which shipped requiring only `spacy` and hit an ImportError for SudachiPy on every
load. The channel is not optional: a model trained with it and loaded without it runs with one of
its inputs deleted, and nothing raises (the same silent-degradation failure as sa's `Compound`
feature on token input).

    bundle_zh_charseg.py [--src training_zh_lemma/model-best] [--seg models/zh_seg_jbdec]
                         [--out build_zh_charseg]
"""
import argparse
import json
import pathlib as _pl
import sys as _sys

# The source arm may carry custom pipes; spacy.load needs their factories registered or it fails
# with E002. seg_code is the project's single --code loader and registers the tokenizer too.
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
try:
    import seg_code  # noqa: F401
except Exception as _e:                                    # pragma: no cover
    print(f"bundle_zh_charseg: seg_code not loaded ({type(_e).__name__}: {_e})")

import spacy  # noqa: E402

SRC = "training_zh_lemma/model-best"
SEG = "models/zh_seg_jbdec"
LEX = "models/zh_lex_corpus.txt"
OUT = "build_zh_charseg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--seg", default=SEG)
    ap.add_argument("--lexicon", default=LEX)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from char_seg_tokenizer import CharSegTokenizer

    nlp = spacy.load(args.src)
    tok = CharSegTokenizer(nlp.vocab)
    # `lexicon` is the FULL training word list, not a jackknifed fold: jackknifing applies only
    # during training, to make train-time coverage match the ~87.6 % the feature meets at test.
    tok.load_segmenter(args.seg, lexicon=args.lexicon)
    nlp.tokenizer = tok
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.CharSegTokenizer.v1"}
    nlp.to_disk(args.out)

    mp = f"{args.out}/meta.json"
    m = json.load(open(mp, encoding="utf-8"))
    m["requirements"] = sorted(set(m.get("requirements") or []) | {"jieba>=0.42.1"})
    json.dump(m, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    meta = json.loads((_pl.Path(args.out) / "tokenizer" / "segmenter" /
                       "vocab.json").read_text(encoding="utf-8"))
    print(f"wrote {args.out}: n_sources={meta.get('n_sources')} "
          f"jieba_source={meta.get('jieba_source')}  requirements={m['requirements']}")
    if meta.get("jieba_source") is None:
        raise SystemExit("REFUSING: the saved segmenter has no jieba_source marker, so the wheel "
                         "would load without the channel it was trained with")
    # Same refusal for the channel's VOCABULARY: a segmenter trained against a traditional jieba
    # dictionary and shipped without it comes back on jieba's simplified one, which loads, segments
    # and is wrong only where the two disagree.
    import sys as _s
    _s.path.insert(0, str(_pl.Path(__file__).resolve().parent))
    import zh_jieba_feature as jf
    sd = _pl.Path(args.out) / "tokenizer" / "segmenter"
    if meta.get("jieba_dict") and not (sd / jf.TRAD_DICT_FILE).is_file():
        raise SystemExit(f"REFUSING: the segmenter records jieba_dict={meta['jieba_dict']!r} but "
                         f"{jf.TRAD_DICT_FILE} was not written beside its weights")


if __name__ == "__main__":
    main()
