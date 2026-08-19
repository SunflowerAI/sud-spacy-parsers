#!/usr/bin/env python3
"""Stamp SudachiPy's ``Inflection`` onto a ja ``.spacy`` corpus, as ``SudInfl`` in FEATS.

WHY THIS EXISTS. ``spacy.ja.JapaneseTokenizer`` writes two things into MORPH that the treebank does
not have: ``Reading`` (katakana) and ``Inflection`` (UniDic conjugation class + inflectional form,
e.g. ``五段-カ行;未然形-一般``). ``Inflection`` is a candidate INPUT feature for the conditioned XPOS
tagger, on the same footing as sa's ``Compound``: the tokeniser supplies it at inference for free,
so conditioning on it is not leakage.

But it is absent while TRAINING. Training reads through ``sud.GoldTokCorpus.v1``, whose predicted
doc is built from the reference's gold words -- the tokeniser never runs -- so a tokeniser-set
feature would be constant-empty in training and appear from nowhere at inference. This script
closes that gap the same way ``sa_tokenizer`` does for ``Compound``: put the value in the corpus,
and let ``sud.InflCorpus.v1`` copy it onto the predicted doc.

WHERE THE VALUE COMES FROM, AND WHY NOT THE TREEBANK. UD Japanese GSD *does* carry readings, in
``MISC=UnidicInfo=...``. They are the wrong source twice over: ``spacy convert --converter conllu``
discards every MISC key but ``SpaceAfter=No`` and the NER pattern, and -- measured -- the best
matching UnidicInfo field agrees with SudachiPy on only 84.7 % of tokens, with the disagreement
concentrated on を / は / し, the highest-frequency function words. An input feature has to be what
the model will MEET at inference, not what the annotators wrote (CLAUDE.md standing hazard 10). So
the value is taken from spaCy's own ja tokeniser, run over the corpus text, and never reconstructed
by hand -- ``inf`` is ``";".join(part_of_speech()[4:])`` and reimplementing it is exactly the kind
of assumed input regime this project keeps paying for.

ALIGNMENT. The tokeniser's segmentation is not the treebank's (measured: 82.3 % positional token
identity on raw text). Each gold token takes the ``Inflection`` of the tokeniser token covering its
LAST character: Japanese inflection sits at the right edge of the word, so a gold token the
tokeniser split carries the ending of its final piece. Gold tokens with no covering value are left
UNSET, not empty -- ``sud.MultiHashEmbedFeats.v1`` maps an absent feature to its own ``Inflection=``
row, and an unset MORPH and an empty one are different inputs (CLAUDE.md).

The value is handed over RAW. No repair of tokeniser/treebank disagreement, no normalisation:
pre-correcting a noisy source destroyed the signal every time this repo tried it
(NEGATIVE-RESULTS.md, "Hand noisy sources over raw").

Usage:
    stamp_ja_inflection.py IN.spacy OUT.spacy [IN2.spacy OUT2.spacy ...]
"""
import sys

import spacy
from spacy.tokens import DocBin

# `=` and `|` would corrupt the FEATS string, `,` would silently become a multi-valued feature and
# change what MultiHashEmbedFeats hashes. spaCy sanitises Reading for exactly this and does NOT
# sanitise Inflection, so the guard belongs here -- and it REFUSES rather than mangling, because a
# component that silently loses an input is the failure mode this project has paid for most.
_FORBIDDEN = set("=|,")

USAGE = "usage: stamp_ja_inflection.py [--tag] IN.spacy OUT.spacy [IN2 OUT2 ...]"


def build_tokenizer():
    """split_mode A, matching every released ja arm's ``[nlp.tokenizer]``.

    Asserted against the arm by ``train_ja_infl.sh`` rather than read off it: this has to run
    before any arm exists, and a silent split_mode drift would be a training/inference skew of
    precisely the kind the script is written to prevent.
    """
    return spacy.blank("ja", config={"nlp": {"tokenizer": {
        "@tokenizers": "spacy.ja.JapaneseTokenizer", "split_mode": "A"}}})


def stamp_docs(docs, tokenizer_nlp, with_tag: bool = False):
    """Stamp in place; return (tokens, stamped, unaligned, distinct values, tags stamped).

    With ``with_tag``, also transports the tokeniser's XPOS as ``SudTag``. That is the exact
    COMPLEMENT of Inflection out of one UniDic analysis (``"-".join(pos[:4])`` against
    ``";".join(pos[4:])``), so the two channels together are the whole analysis.

    ⚠ It must come from the TOKENISER, never from the reference's own TAG. The reference carries
    GOLD XPOS, which is the tagger's target -- copying that onto the predicted doc would be
    leakage, and it would also be a lie about inference, where the tokeniser's tag agrees with
    gold only 76.7 % of the time.
    """
    n_tok = n_val = n_unaligned = n_tag = 0
    values = set()
    for doc in docs:
        sdoc = tokenizer_nlp.make_doc(doc.text)
        cover = {}                                  # char offset -> (Inflection, tag)
        for t in sdoc:
            vals = t.morph.get("Inflection")
            v = ",".join(vals) if vals else None
            for i in range(t.idx, t.idx + len(t.text)):
                cover[i] = (v, t.tag_)
        for tok in doc:
            n_tok += 1
            if not tok.text:
                continue
            got = cover.get(tok.idx + len(tok.text) - 1)    # right edge: the inflected ending
            if got is None:
                n_unaligned += 1
                continue
            v, tag = got
            d = tok.morph.to_dict()
            for key, val in (("SudInfl", v), ("SudTag", tag if with_tag else None)):
                if not val:
                    continue
                bad = _FORBIDDEN & set(val)
                if bad:
                    raise SystemExit(f"refusing to stamp: {key} value {val!r} contains "
                                     f"{sorted(bad)}, which FEATS cannot carry losslessly")
                d[key] = val
            if v:
                values.add(v)
                n_val += 1
            if with_tag and tag:
                n_tag += 1
            tok.set_morph(d)
    return n_tok, n_val, n_unaligned, len(values), n_tag


def main():
    with_tag = "--tag" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--tag"]
    if not args or len(args) % 2:
        sys.exit(USAGE)
    nlp = build_tokenizer()
    for src, dst in zip(args[0::2], args[1::2]):
        db_in = DocBin().from_disk(src)
        docs = list(db_in.get_docs(nlp.vocab))
        n_tok, n_val, n_unal, n_vals, n_tag = stamp_docs(docs, nlp, with_tag)
        # same attrs as the source bin, so nothing the corpus carried (SENT_START above all, which
        # is what makes these multi-sentence docs teach segmentation) is dropped on the way out.
        db_out = DocBin(attrs=db_in.attrs, store_user_data=True)
        for d in docs:
            db_out.add(d)
        db_out.to_disk(dst)
        print(f"{src} -> {dst}")
        print(f"  docs {len(docs)}  tokens {n_tok}")
        print(f"  SudInfl stamped on {n_val} ({n_val / max(n_tok, 1):.2%}), "
              f"{n_vals} distinct values")
        print(f"  no covering tokeniser value on {n_unal} ({n_unal / max(n_tok, 1):.2%})"
              f" -- left UNSET")
        if with_tag:
            print(f"  SudTag stamped on {n_tag} ({n_tag / max(n_tok, 1):.2%})")


if __name__ == "__main__":
    main()


def apply_channels(doc, tokenizer_nlp):
    """Set the LIVE tokeniser channels (``Inflection`` in MORPH, XPOS on ``tag``) on ``doc``.

    The corpus path transports these as ``SudInfl``/``SudTag`` because a ``.spacy`` file has to
    survive a round trip; an evaluator holding a Doc does not, and needs the values under the names
    the model actually reads.

    WHY THIS IS NEEDED AT ALL. Any harness that builds a doc as ``Doc(vocab, words=[...])`` -- which
    is what gold-token evaluation means -- produces a doc with ``tag == 0`` and MORPH unset. For an
    arm whose SHARED encoder reads those two channels, that deletes both inputs, and the arm runs
    out of its own regime while every number still prints. Measured on ja: the parser's ``unk`` F
    falls 0.948 -> 0.786 on a bare Doc, which drags the Idiom rule (a conjunction of ExtPos and
    ``unk``) down ~6 F and InIdiom (pure ``unk`` chains) ~8 F, with nothing raising.
    """
    sdoc = tokenizer_nlp.make_doc(doc.text)
    cover = {}
    for t in sdoc:
        vals = t.morph.get("Inflection")
        v = ",".join(vals) if vals else None
        for i in range(t.idx, t.idx + len(t.text)):
            cover[i] = (v, t.tag_)
    for tok in doc:
        got = cover.get(tok.idx + len(tok.text) - 1)   # right edge, as in stamp_docs
        if got is None:
            continue
        v, tag = got
        if v:
            tok.set_morph({"Inflection": v})           # only where there IS a value: unset != empty
        if tag:
            tok.tag_ = tag
    return doc


def needs_channels(nlp) -> bool:
    """Does this arm's own tokeniser supply anything a bare Doc would lack?

    Behavioural, so it needs no per-language table: ja pre-sets 3/3 tags on an ASCII probe,
    en/ko/zh 0/3. Same probe the packaging guard uses.
    """
    probe = nlp.make_doc("Test 1.")
    return any(t.tag != 0 or t.morph.get("Inflection") for t in probe)
