#!/usr/bin/env python3
"""Score `ar_vocalise` against PADT's gold ``Vform``.

Reports the two measures the Arabic diacritisation literature uses, so the result can be read
beside published systems, plus the split that matters most for THIS design:

    WER   whole-token exact match complement -- the honest headline
    DER   diacritic error rate, per diacritic-bearing position
    -ce   both again ignoring the FINAL character, i.e. blind to the case ending (iʿrāb)

The `-ce` pair is the diagnostic. The case ending is a syntactic fact, not a lexical one, so it is
the part a table cannot settle and the part this pipeline's parser and morphologiser exist to
supply; the gap between the two columns is precisely what the morphology is buying, and where the
remaining errors live.

Everything is compared through `ar_vocalise.canon`, so a difference of writing convention is not
scored as an error -- see the module docstring for the four conventions involved.

    python scripts/eval_ar_vocalise.py                       # predicted morphology (deployment)
    python scripts/eval_ar_vocalise.py --gold-morph          # the table's own ceiling
    python scripts/eval_ar_vocalise.py --no-camel            # table only, no GPL fall-through
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar_tokenizer            # noqa: F401,E402  registers ar.CamelAtbTokenizer.v1
import sud_feats_embed         # noqa: F401,E402
import sud_tagger              # noqa: F401,E402
from ar_vocalise import DIAC, ArVocalise, canon, strip_diac   # noqa: E402

DEFAULT_TEST = "assets_ar/SUD_Arabic-PADT/ar_padt-sud-test.conllu"
DEFAULT_MODEL = "training_ar_sud/model-best"
DEFAULT_LUT = Path(__file__).resolve().parent / "ar_vocalise_lut.json.gz"


def read_sents(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                sents.append(cur)
            cur = []
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            continue
        misc = dict(kv.split("=", 1) for kv in f[9].split("|") if "=" in kv)
        cur.append((f[1], f[3], f[5], misc.get("Vform")))
    if cur:
        sents.append(cur)
    return sents


def diac_seq(s):
    """Per-consonant diacritic string: for each non-diacritic character, the run of diacritics
    that follows it. Aligning on the SKELETON rather than on raw character index is what makes DER
    meaningful -- a single inserted vowel would otherwise shift every later position."""
    out, cur = [], None
    for ch in s:
        if ch in DIAC:
            if cur is not None:
                out[cur] += ch
        else:
            out.append("")
            cur = len(out) - 1
    return out


def score(pairs):
    """pairs: (predicted, gold). Returns WER/DER, whole and case-ending-blind."""
    n = werr = 0
    n_ce = werr_ce = 0
    dn = derr = dn_ce = derr_ce = 0
    for pred, gold in pairs:
        p, g = canon(pred), canon(gold)
        n += 1
        werr += (p != g)
        ps, gs = diac_seq(p), diac_seq(g)
        if len(ps) == len(gs):                     # skeletons agree -> DER is well defined
            for i, (a, b) in enumerate(zip(ps, gs)):
                dn += 1
                derr += (a != b)
                if i < len(gs) - 1:
                    dn_ce += 1
                    derr_ce += (a != b)
            n_ce += 1
            werr_ce += (ps[:-1] != gs[:-1])
        else:
            dn += len(gs)
            derr += len(gs)
    return dict(n=n, WER=werr / n, DER=derr / max(dn, 1),
                WER_ce=werr_ce / max(n_ce, 1), DER_ce=derr_ce / max(dn_ce, 1),
                aligned=n_ce / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lut", default=str(DEFAULT_LUT))
    ap.add_argument("--gold-morph", action="store_true",
                    help="use the treebank's UPOS/FEATS instead of the model's")
    ap.add_argument("--no-camel", action="store_true", help="table only")
    a = ap.parse_args()

    comp = ArVocalise(lut=a.lut, camel=not a.no_camel)
    sents = read_sents(a.test)
    pairs, levels = [], collections.Counter()

    if a.gold_morph:
        for s in sents:
            for form, upos, feats, v in s:
                if not v:
                    continue
                pred, lvl = comp.lookup(form, upos, feats)
                levels[lvl] += 1
                pairs.append((pred, v))
    else:
        import spacy
        from spacy.tokens import Doc
        nlp = spacy.load(a.model)
        for s in sents:
            # gold WORDS, predicted annotation: the tokenisation is not under test here, and
            # re-tokenising would misalign the comparison with the gold Vform column.
            doc = Doc(nlp.vocab, words=[t[0] for t in s])
            for _, pipe in nlp.pipeline:
                doc = pipe(doc)
            for (form, _, _, v), tok in zip(s, doc):
                if not v:
                    continue
                pred, lvl = comp.lookup(tok.text, tok.pos_, str(tok.morph) or "_")
                levels[lvl] += 1
                pairs.append((pred, v))

    r = score(pairs)
    tag = "GOLD morphology" if a.gold_morph else "PREDICTED morphology"
    cam = "table only" if a.no_camel else "table + CAMeL fall-through"
    print(f"{tag}, {cam} -- {r['n']} tokens")
    print(f"  WER    {r['WER']:.4f}   (exact {1 - r['WER']:.2%})")
    print(f"  DER    {r['DER']:.4f}")
    print(f"  WER-ce {r['WER_ce']:.4f}   (case ending ignored)")
    print(f"  DER-ce {r['DER_ce']:.4f}")
    print(f"  skeleton-aligned {r['aligned']:.2%}")
    print("  rungs:", dict(levels))


if __name__ == "__main__":
    main()
