#!/usr/bin/env python3
"""Compare the trained `sud_tagger` against `sud_reported_rule`, end to end.

Both are run over gold TOKENS with everything else predicted, so the comparison is on the same
footing -- the rule reads the parser's deprels and the morphologiser's VerbForm/Mood, the trained
pipe reads only surface forms through its own encoder.

CAVEAT, and it is a large one: there is NO independent gold for `Reported`. The treebanks annotate
it nowhere, so the target here is the bootstrapped file, which is itself these rules plus an LLM
pass over the residue. The rule therefore scores high substantially BY CONSTRUCTION. What the
comparison shows is which component reproduces the intended annotation at inference, not which
annotation is correct.

    scripts/eval_sud_reported.py --lang la
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import stamp_ja_inflection  # noqa: E402,F401
import seg_code  # noqa: E402,F401
import sud_reported_rule  # noqa: E402,F401
from sud_misc import get_misc  # noqa: E402

TEST = {
    "en": "assets/en_ewt-sud-{split}.relabeled_ext.reported.conllu",
    # en_gum: the second English arm (EWT + the ten non-NonCommercial GUM genres).
    # A separate key, not a replacement -- `en` must keep pointing at the EWT-only files
    # that the released CC BY-SA en_sud_ewt wheel was built from.
    "en_gum": "assets/en_ewtgum-sud-{split}.relabeled_ext.reported.conllu",
    "ar": "assets_ar/SUD_Arabic-PADT/ar_padt-sud-{split}.relabeled_ext.reported.conllu",
    "fa": "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-{split}.relabeled_ext.reported.conllu",
    "la": "assets_la/la_ittbproiel-sud-{split}.relabeled_ext.reported.conllu",
    # ⚠ csl_mwt, not csl_rev (the superseded pausa representation). THIS FILE DOES NOT EXIST YET:
    # regenerate it with `sud_reported_gold.py sa`, which now reads the csl_mwt source. Pointed here
    # deliberately rather than left on the stale file -- a missing input is an error, and a
    # superseded one is a plausible wrong number.
    "sa": "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.relabeled_ext.csl_mwt.reported.conllu",
}
# The arm the RULE is attached to. It must be the arm the rule actually ships on, because the rule
# reads that arm's predicted deprels and VerbForm/Mood -- attach it to a different generation and
# what is measured is the mismatch, not the rule.
# ⚠ sa was `training_sa_lemma3_noannot/model-best` until 2026-08-23: an arm of the SUPERSEDED
# csl_rev chain, scored against current-representation gold. The released sa arm is
# `training_sa_mp2_sub_s1/model-best` (`SA_BASE` in package_sud.sh), which is where
# `add_sud_reported_rule.py` puts the pipe at packaging time.
LEMMA_ARM = {"sa": "training_sa_mp2_sub_s1/model-best"}


def sentences(path):
    block = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if block:
                yield block
            block = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        block.append(f)
    if block:
        yield block



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


def score(nlp, rows_iter):
    tp = fp = fn = skipped = 0
    for rows in rows_iter:
        doc = _gold_doc(nlp, rows)
        doc = nlp(doc)
        if len(doc) != len(rows):
            skipped += 1
            continue
        for tok, f in zip(doc, rows):
            pred = get_misc(tok, "Reported") == "Yes"
            gold = "Reported=Yes" in f[9]
            tp += pred and gold
            fp += pred and not gold
            fn += gold and not pred
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp + fn, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(TEST))
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm", default=None,
                    help="override the arm the RULE is attached to (default: the released arm)")
    ap.add_argument("--gold", default=None,
                    help="override the gold path template, e.g. a superseded representation, "
                         "for an explicitly like-for-like comparison")
    ap.add_argument("--trained", default=None,
                    help="override the trained arm dir (e.g. training_ar_sudrep/model-best)")
    args = ap.parse_args()

    path = (args.gold or TEST[args.lang]).format(split=args.split)
    if not pathlib.Path(path).exists():
        sys.exit(f"missing {path}")
    rows = list(sentences(path))
    print(f"{args.lang} {args.split}   (gold tokens, everything else predicted)")

    trained = args.trained or f"training_{args.lang}_sud/model-best"
    if pathlib.Path(trained).exists():
        nlp = spacy.load(trained)
        if "sud_reported" in nlp.pipe_names:
            P, R, F, n, sk = score(nlp, rows)
            print(f"  sud_tagger (trained)  gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}")
        else:
            print(f"  sud_tagger (trained)  -- no sud_reported pipe in {trained}")
    else:
        print(f"  sud_tagger (trained)  -- {trained} missing")

    lemma = args.arm or LEMMA_ARM.get(args.lang, f"training_{args.lang}_lemma/model-best")
    if pathlib.Path(lemma).exists():
        nlp = spacy.load(lemma)
        if "sud_reported_rule" in nlp.pipe_names:
            nlp.remove_pipe("sud_reported_rule")
        nlp.add_pipe("sud_reported_rule", last=True, config={"lang": args.lang})
        P, R, F, n, sk = score(nlp, rows)
        print(f"  sud_reported_rule     gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}")
    else:
        print(f"  sud_reported_rule     -- {lemma} missing")


if __name__ == "__main__":
    main()
