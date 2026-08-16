#!/usr/bin/env python3
"""Compare the trained `sud_tagger` and the `sud_shared_rule` table for `Shared`, end to end.

Three arms, on identical input (gold TOKENS, everything else predicted), because that is the only
footing on which they are comparable:

  morphologizer   the status quo. `Shared` is a FEATS feature in every treebank here, so the
                  released morphologisers have been predicting it all along inside their FEATS
                  bundles -- which is why it is scored here as a baseline rather than ignored.
  sud_shared_rule the parse plus a harvested decision table (build_sud_shared_frames.py).
  sud_tagger      the trained pipe from training_<lang>_shared/ or training_<lang>_sud/.

The rule reads PREDICTED heads, relations and UPOS, so its accuracy falls with parse quality; the
trained pipe reads its own encoder. Neither can be judged on gold trees without flattering one of
them, hence end-to-end only. Scoring counts a token right when the value matches, so a `Yes` where
the gold says `No` is both a false positive and a false negative -- the same convention as
`eval_sud_subject.py`.

    scripts/eval_sud_shared.py --lang en
    scripts/eval_sud_shared.py --lang lzh --split dev
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import stamp_ja_inflection  # noqa: E402,F401
import seg_code  # noqa: E402,F401  (custom tokenisers + sud_tagger)
import sud_shared_rule  # noqa: E402,F401  (registers the rule factory)
from build_sud_shared_frames import TRAIN, gold_shared, path_for, sentences  # noqa: E402
from sud_misc import get_misc  # noqa: E402
from sud_shared_data import doc_candidates  # noqa: E402

# The released arm to source when scoring the rule / the morphologiser (see package_sud.sh).
# Must match src_model() in train_sud.sh and the arm package_sud.sh actually ships -- the layer
# reads the arm's own predictions, so scoring it against a different generation measures nothing.
LEMMA_ARM = {
    "sa": "training_sa_multitask/model-best",
    # lzh has no trained lemmatizer any more (han_lemma_lut replaces it at packaging), so the top
    # of its chain is the MORPH storey of the rule-merged punctuation arm.
    "lzh": "training_lzh_rm_morph/model-best",
    "ko": "training_ko_eojeol_lemma/model-best",
    "id": "training_id_split_lemma/model-best",
    # la ships the ORTHOGRAPHICALLY AUGMENTED chain: one copy of the macronised treebank resampled
    # into a fresh edition style every epoch, instead of the plain-plus-macron union. Different
    # parser, therefore a different coordination mask, therefore a different measurement.
    "la": "training_la_aug_lemma/model-best",
}
# --code each arm needs on top of seg_code's imports.
EXTRA_CODE = {"la": "la_macronise"}


def lemma_arm(lang):
    # EVAL_LEMMA_ARM / EVAL_SHARED_ARM: score a CANDIDATE base rather than the released one, without
    # editing the tables above. Added for the vocalisation-augmented arms, whose ship decisions have
    # to be re-measured because this layer reads the base's own predictions -- the coordination mask
    # is a fact about that parser, not about the language.
    return os.environ.get("EVAL_LEMMA_ARM") or LEMMA_ARM.get(
        lang, f"training_{lang}_lemma/model-best")


# Which trained arm to score, named EXPLICITLY rather than "whichever directory happens to exist".
# The positional version silently preferred any `training_<lang>_shared/`, so a solo pilot left on
# disk displaced the arm that ships -- en's solo arm scores test 62.23 against the shipped 62.62,
# and the eval would have quietly reported the worse one.
SHARED_ARM = {
    # lzh's Shared pipe rides the rule-merged punctuation chain (see package_sud.sh).
    "lzh": "training_lzh_rm_sud/model-best",
    # la's trained arm on the AUGMENTED base. ⚠ This is the three-feature arm, whose `model-best`
    # is picked on the MEAN of Subject/Reported/Shared -- on the union base that handicapped Shared
    # by ~5 points (30.06 against a solo-trained 35.10). So a trained-loses result here is not
    # decisive on its own; a trained-WINS result is, since the handicap runs the other way.
    "la": "training_la_aug_sud/model-best",
}


def trained_arm(lang):
    cand = os.environ.get("EVAL_SHARED_ARM") or SHARED_ARM.get(
        lang, f"training_{lang}_sud/model-best")
    return cand if pathlib.Path(cand).exists() else None



# ⚠ Any harness that builds a doc from GOLD WORDS must put back what the tokeniser would have
# supplied. `Doc(vocab, words=[...])` yields tag == 0 and MORPH unset, so an arm whose encoder
# reads tokeniser-set channels (ja: XPOS + Inflection) runs with those inputs DELETED and every
# number still prints. Measured on ja's idiom layer: the parser's `unk` F fell 0.948 -> 0.786 and
# the rule lost ~6-8 F, which looked exactly like a base regression and was not one.
# `spaces=` matters for the same reason: without it doc.text gets a space after every token, and
# re-running the tokeniser over that analyses a string that never occurs at inference -- which
# measured WORSE than supplying no channels at all.
_CHAN_CACHE = {}


def _channel_nlp(nlp):
    """The tokeniser whose channels this arm reads, or None. Behavioural, so no language table:
    ja pre-sets 3/3 tags on an ASCII probe, en/ko/zh 0/3. Cached -- building it is not free."""
    key = id(nlp)
    if key not in _CHAN_CACHE:
        _CHAN_CACHE[key] = (stamp_ja_inflection.build_tokenizer()
                            if stamp_ja_inflection.needs_channels(nlp) else None)
    return _CHAN_CACHE[key]


def _gold_doc(nlp, rows):
    doc = Doc(nlp.vocab, words=[f[1] or "_" for f in rows],
              spaces=["SpaceAfter=No" not in (f[9] if len(f) > 9 else "") for f in rows])
    chan = _channel_nlp(nlp)
    if chan is not None:
        stamp_ja_inflection.apply_channels(doc, chan)
    return doc


def score(nlp, rows_iter, source):
    """`source` is "slot" (Token._.sud_misc) or "morph" (token.morph, the morphologiser)."""
    tp = fp = fn = skipped = 0
    for rows in rows_iter:
        doc = _gold_doc(nlp, rows)
        doc = nlp(doc)
        if len(doc) != len(rows):
            skipped += 1
            continue
        for tok, f in zip(doc, rows):
            if source == "morph":
                values = tok.morph.get("Shared")
                pred = values[0] if values else None
            else:
                pred = get_misc(tok, "Shared")
            gold = gold_shared(f)
            if pred and gold and pred == gold:
                tp += 1
            else:
                fp += bool(pred)
                fn += bool(gold)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp + fn, skipped


def mask_ceiling(nlp, rows_iter):
    """How much of the gold the candidate mask can reach on a PREDICTED parse.

    This is the number that explains a language's result, and it is worth reading before either
    the rule or the trained pipe, because both are built on the mask and neither can exceed it.
    The mask is defined over the coordination -- who is a conjunct, where the conjuncts sit -- so
    its quality is the PARSER's quality on exactly that structure, not on the sentence at large.
    Sanskrit is the worked example: on gold trees the mask covers the feature well enough for the
    harvested table to reach dev F 52, but on its own predicted trees (LAS ~0.51) the mask and the
    gold barely intersect, so the trained pipe saw almost no positive example and learnt nothing.
    """
    inside = total = size = 0
    for rows in rows_iter:
        doc = nlp(_gold_doc(nlp, rows))
        if len(doc) != len(rows):
            continue
        cand = {i for i, _ in doc_candidates(doc)}
        size += len(cand)
        for i, f in enumerate(rows):
            if gold_shared(f):
                total += 1
                inside += i in cand
    return inside / total if total else 0.0, size, total


def load(path, lang):
    for name in ([EXTRA_CODE[lang]] if lang in EXTRA_CODE else []):
        __import__(name)
    return spacy.load(path)


def line(label, result):
    p, r, f, n, sk = result
    print(f"  {label:20s} gold={n:6d}  P={p:7.2%} R={r:7.2%} F={f:7.2%}"
          + (f"  [{sk} sents skipped]" if sk else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(TRAIN))
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", nargs="+", default=["mask", "morph", "rule", "trained"],
                    choices=["mask", "morph", "rule", "trained"])
    args = ap.parse_args()

    path = path_for(args.lang, args.split)
    if not pathlib.Path(path).exists():
        sys.exit(f"missing {path}")
    rows = list(sentences(path))
    print(f"{args.lang} {args.split}   (gold tokens, everything else predicted)")

    base = lemma_arm(args.lang)
    if "mask" in args.arms:
        if pathlib.Path(base).exists():
            covered, size, n = mask_ceiling(load(base, args.lang), rows)
            print(f"  {'candidate mask':20s} gold={n:6d}  covers {covered:6.2%} of it, "
                  f"over {size} candidate tokens  <- ceiling for rule and trained alike")
        else:
            print(f"  candidate mask       -- {base} missing")

    if "morph" in args.arms:
        if pathlib.Path(base).exists():
            line("morphologizer", score(load(base, args.lang), rows, "morph"))
        else:
            print(f"  morphologizer        -- {base} missing")

    if "rule" in args.arms:
        if pathlib.Path(base).exists():
            nlp = load(base, args.lang)
            if "sud_shared_rule" in nlp.pipe_names:
                nlp.remove_pipe("sud_shared_rule")
            nlp.add_pipe("sud_shared_rule", last=True, config={"lang": args.lang})
            line("sud_shared_rule", score(nlp, rows, "slot"))
        else:
            print(f"  sud_shared_rule      -- {base} missing")

    if "trained" in args.arms:
        trained = trained_arm(args.lang)
        if trained:
            line("sud_tagger (trained)", score(load(trained, args.lang), rows, "slot"))
        else:
            print(f"  sud_tagger (trained) -- no training_{args.lang}_shared or _sud arm")


if __name__ == "__main__":
    main()
