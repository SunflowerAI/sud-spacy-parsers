#!/usr/bin/env python3
"""Score the `sud_idiom` rule against the treebank's gold `Idiom=Yes` / `InIdiom=Yes`.

Two modes, and the difference between them is the point:

  --gold-trees (default)  Build each Doc from the treebank's own heads/deprels/FEATS and run only
                          the component. This scores the RULE in isolation: it should be P=R=100%
                          (la InIdiom 99.86%, one token). Use it as a regression check -- any drop
                          means the rule or its inputs changed.

  --model PATH            Run the real pipeline over the sentence text, then the component, and
                          score against gold. This is what users actually get, and it is LOWER,
                          because the rule inherits the morphologiser's `ExtPos` errors and the
                          parser's `unk` errors. Reported alongside the gold-tree number, the gap
                          is the honest measure of what the layer contributes.

    scripts/eval_sud_idiom.py --lang ar
    scripts/eval_sud_idiom.py --lang ar --split test --model training_ar_lemma/model-best
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import sud_idiom  # noqa: E402
from sud_misc import get_misc  # noqa: E402

# The seven arms whose treebanks annotate idioms. zh/yue/ko/id carry none, so the component is
# not added to them at packaging time and there is nothing here to score.
FILES = {
    "en":  "assets/en_ewt-sud-{split}.relabeled_ext.conllu",
    # en_gum: the second English arm (EWT + the ten non-NonCommercial GUM genres).
    # A separate key, not a replacement -- `en` must keep pointing at the EWT-only files
    # that the released CC BY-SA en_sud_ewt wheel was built from.
    "en_gum": "assets/en_ewtgum-sud-{split}.relabeled_ext.conllu",
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-{split}.relabeled_ext.conllu",
    "ja":  "assets_ja/SUD_Japanese-GSD/ja_gsd-sud-{split}.relabeled_ext.conllu",
    "fa":  "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-{split}.relabeled_ext.conllu",
    "ar":  "assets_ar/SUD_Arabic-PADT/ar_padt-sud-{split}.relabeled_ext.conllu",
    "la":  "assets_la/la_ittbproiel-sud-{split}.relabeled_ext.conllu",
    # csl_mwt, not csl_rev: the DCS/MWT generation. `corpus_sa_csl_rev` is the SUPERSEDED pausa-normalised representation (CLAUDE.md lists `rebuild_sa_csl_rev.sh` under "Superseded but kept"); its FORMs and tokenisation differ from the arm that ships, and it is UNRELABELLED (`udep` 7.89 % of tokens against 0.00 %). See the BUILD PROVENANCE table in docs/sanskrit.md.
    "sa":  "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.relabeled_ext.csl_mwt.conllu",
}
KEYS = ("Idiom", "InIdiom")


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
        if "-" in f[0] or "." in f[0]:  # MWT range / empty node
            continue
        block.append(f)
    if block:
        yield block


def col_dict(col):
    d = {}
    if col == "_":
        return d
    for item in col.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
    return d


def gold_doc(vocab, rows):
    """A Doc carrying the treebank's own analysis -- what the rule would see if the model were perfect."""
    ids = {f[0]: i for i, f in enumerate(rows)}
    return Doc(
        vocab,
        words=[f[1] or "_" for f in rows],
        spaces=["SpaceAfter=No" not in (f[9] if len(f) > 9 else "") for f in rows],
        heads=[ids.get(f[6], i) for i, f in enumerate(rows)],
        deps=[f[7] for f in rows],
        morphs=["" if f[5] == "_" else f[5] for f in rows],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(FILES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None,
                    help="score end-to-end through this model instead of on gold trees")
    args = ap.parse_args()

    path = FILES[args.lang].format(split=args.split)
    if not pathlib.Path(path).exists():
        sys.exit(f"missing {path}")

    if args.model:
        # Registers the custom tokenisers (ar/yue/lzh/sa) and clause_parser, exactly as
        # `spacy train --code scripts/seg_code.py` does; without it ar/lzh/sa fail with E893.
        import seg_code  # noqa: F401
        import stamp_ja_inflection
        nlp = spacy.load(args.model)
        if "sud_idiom" not in nlp.pipe_names:
            nlp.add_pipe("sud_idiom", last=True)
        vocab = nlp.vocab
        _chan_nlp = (stamp_ja_inflection.build_tokenizer()
                     if stamp_ja_inflection.needs_channels(nlp) else None)
        if _chan_nlp is not None:
            print("  (restoring tokeniser-supplied channels: this arm's encoder reads them)")
    else:
        nlp = None
        _chan_nlp = None
        vocab = spacy.blank("xx").vocab
    component = sud_idiom.SudIdiom(None, "sud_idiom")

    counts = {k: [0, 0, 0] for k in KEYS}   # tp, fp, fn
    skipped = 0
    for rows in sentences(path):
        if nlp is None:
            doc = gold_doc(vocab, rows)
            component(doc)
            pairs = list(zip(doc, rows))
        else:
            # Feed gold tokens so the comparison is token-aligned: this measures the ExtPos/unk
            # predictions, not the tokeniser. (A tokeniser-level evaluation is a different question.)
            # SpaceAfter matters, and not only cosmetically: `Doc(words=...)` puts a space after
            # EVERY token, so doc.text becomes fully-spaced Japanese -- a string that never occurs
            # at inference. Re-running the tokeniser over that to recover its channels yields an
            # analysis of the wrong text, which measured WORSE than having no channels at all.
            doc = Doc(vocab, words=[f[1] or "_" for f in rows],
                      spaces=["SpaceAfter=No" not in f[9] for f in rows])
            # ⚠ A bare Doc has tag == 0 and MORPH unset. For an arm whose encoder reads
            # tokeniser-supplied channels (ja: XPOS + Inflection) that DELETES its inputs and the
            # arm runs out of regime while every number still prints -- measured at -16 F on ja's
            # `unk`, which this rule conjoins. So put the channels back exactly as the tokeniser
            # would at inference. No-op for every arm whose tokeniser supplies nothing.
            if _chan_nlp is not None:
                stamp_ja_inflection.apply_channels(doc, _chan_nlp)
            doc = nlp(doc)
            if len(doc) != len(rows):
                skipped += 1
                continue
            pairs = list(zip(doc, rows))

        for tok, f in pairs:
            gold = col_dict(f[9])
            for key in KEYS:
                p = get_misc(tok, key) == "Yes"
                g = gold.get(key) == "Yes"
                c = counts[key]
                c[0] += p and g
                c[1] += p and not g
                c[2] += g and not p

    mode = f"model={args.model}" if args.model else "gold trees"
    print(f"{args.lang} {args.split}  ({mode})")
    for key in KEYS:
        tp, fp, fn = counts[key]
        P = tp / (tp + fp) if tp + fp else 1.0
        R = tp / (tp + fn) if tp + fn else 1.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print(f"  {key:8} gold={tp + fn:6}  P={P:7.2%}  R={R:7.2%}  F={F:7.2%}")
    if skipped:
        print(f"  ({skipped} sentences skipped: pipeline changed the token count)")


if __name__ == "__main__":
    main()
