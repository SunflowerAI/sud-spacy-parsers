#!/usr/bin/env python3
"""Score ``la_macronise`` against the Alatius macroniser's output on a held-out split.

Two conditions, because the gap between them IS the contribution of the morphologiser:

  --gold        key the lookup on the treebank's gold UPOS/FEATS (an upper bound)
  (default)     run the released pipeline over gold TOKENS and key on its PREDICTIONS,
                which is what a caller actually gets

Reported per token (the whole vowel-length pattern must be exactly right) and per vowel (each
vowel independently), broken down by which backoff level fired.

The reference is the Alatius macroniser's output, NOT gold vowel length -- Alatius is itself
~98-99 % accurate on vowels, so these are agreement figures and its errors are counted as ours
being correct. See scripts/la_macronise.py.

    eval_la_macronise.py training_la_lemma/model-best \
        assets_la/la_ittbproiel-sud-test.conllu \
        assets_la/la_ittbproiel-sud-test.macron.conllu
"""
import argparse
import importlib.util
import unicodedata
from collections import Counter

import spacy
from spacy.tokens import Doc

VOWELS = set("aeiouyAEIOUY")


def load_code(path):
    spec = importlib.util.spec_from_file_location(path.split("/")[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strip_macron(s):
    n = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in n if c != "̄"))


def sentences(plain, macron):
    """Yield (forms, macronised forms, upos, feats) per sentence block."""

    def blocks(path):
        cur = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    yield cur
                cur = []
            elif "\t" in line and line.split("\t", 1)[0].isdigit():
                cur.append(line.split("\t"))
        if cur:
            yield cur

    for pb, mb in zip(blocks(plain), blocks(macron)):
        yield ([r[1] for r in pb], [r[1] for r in mb],
               [r[3] for r in pb], [r[5] for r in pb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("plain")
    ap.add_argument("macron")
    ap.add_argument("--lut", default="scripts/la_macron_lut.json.gz")
    ap.add_argument("--gold", action="store_true",
                    help="use the treebank's gold UPOS/FEATS instead of the model's predictions")
    ap.add_argument("--no-paradigm", action="store_true",
                    help="disable the paradigm override (see _PARADIGM in la_macronise.py)")
    ap.add_argument("--passes", type=int, default=1,
                    help="re-run the pipeline over its OWN macronised output this many times. "
                         "Macrons disambiguate Case, so pass 2 tags better than pass 1 -- the "
                         "component is not idempotent, by design.")
    args = ap.parse_args()

    mod = load_code("scripts/la_macronise.py")
    nlp = spacy.load(args.model)
    if "la_macronise" in nlp.pipe_names:
        nlp.remove_pipe("la_macronise")
    nlp.add_pipe("la_macronise", config={"lut": args.lut}, last=True)
    comp = nlp.get_pipe("la_macronise")
    comp.paradigm = not args.no_paradigm

    fired, corr = Counter(), Counter()
    tok_ok = tok_n = v_ok = v_n = 0
    for forms, macd, upos, feats in sentences(args.plain, args.macron):
        if args.gold:
            pairs = zip(forms, macd, upos, feats)
        else:
            words = list(forms)
            for _ in range(args.passes):
                doc = nlp(Doc(nlp.vocab, words=words))
                # feed the pipeline its own macronised tokens on the next pass
                words = [t._.macron for t in doc]
            pairs = zip(forms, macd, [t.pos_ for t in doc],
                        [str(t.morph) or "_" for t in doc])
        for form, gold, u, x in pairs:
            if not any(c.isalpha() for c in form) or strip_macron(gold) != form:
                continue
            out, lvl = comp.resolve(form, u, x)
            pred = sum(1 << i for i, c in enumerate(out)
                       if mod.strip_macron(c) != c)
            tok_n += 1
            ok = (out == gold)
            tok_ok += ok
            fired[lvl or "bare"] += 1
            corr[lvl or "bare"] += ok
            for i, c in enumerate(form):
                if c in VOWELS:
                    v_n += 1
                    v_ok += (((pred >> i) & 1) == (strip_macron(gold[i]) != gold[i]))

    cond = "GOLD morphology" if args.gold else "PREDICTED morphology"
    print(f"{cond}  ({tok_n} alphabetic tokens)")
    for lvl in ("L1", "L2", "L3", "S4", "S3", "bare"):
        if fired[lvl]:
            print(f"  {lvl:5s} fired {fired[lvl]:6d} ({100*fired[lvl]/tok_n:5.2f}%)"
                  f"  token-exact {100*corr[lvl]/fired[lvl]:5.2f}%")
    print(f"  => whole-token {100*tok_ok/tok_n:.2f}%   per-vowel {100*v_ok/v_n:.2f}%")


if __name__ == "__main__":
    main()
