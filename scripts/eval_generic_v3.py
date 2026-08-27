#!/usr/bin/env python3
"""Score a v3 arm on the held-out languages, under a NAMED fill regime.

The v3 channel trains on aligned LEMMA vectors and deploys on ENGLISH GLOSS vectors. Those are the
same space, but they are not the same input, so an arm must be scored under the regime it will
actually meet. Three regimes are meaningful and they are not interchangeable:

    lemma   the training fill. On a held-out language with no rows this is all-OOV, so it measures
            the arm WITHOUT a lexical channel -- the honest floor, and the thing a naive run would
            report by accident while calling it the headline.
    gloss   the deployment fill. Token._.gloss is set from a Wiktionary bag and resolved against
            the ENGLISH rows, which exist for every language by construction.
    auto    gloss where there is one, lemma otherwise -- a language that has both.

⚠ TWO GLOSS KEYS, AND ONLY ONE OF THEM IS THE DEPLOYMENT NUMBER.

    --gloss-key lemma   an UPPER BOUND. The v2 contract declares UPOS and tb_lang as user inputs
                        and says nothing about lemmas, so keying the dictionary by the gold lemma
                        column quietly adds an annotation layer the contract does not have.
    --gloss-key form    what a deployer actually has. Wiktionary's inflected-form entries inherit
                        their lemma's bag (kaikki_anchors --follow-forms), which is what makes this
                        route viable at all.

Report both. Never quote one for the other.

⚠ THREE TEST LANGUAGES CANNOT BE SCORED ON THE GLOSS FILL BY ANY ROUTE -- Bororo, Komi-Zyrian and
Xavante have neither a Wiktionary extract nor a `Gloss=` column. A macro "over the test languages"
that silently means seventeen of twenty is the kind of number this repo has had to retract before.
The JSON records `scored` and `unscorable` explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                                    # noqa: E402
from spacy.tokens import Doc                                    # noqa: E402
from spacy.training import Example                              # noqa: E402

import generic_code_v3                                          # noqa: E402,F401
from generic_corpus import annotate                             # noqa: E402
from eval_generic_v2 import load_docs, score, one_sentence_per_doc   # noqa: E402
from sud_generic_embed_v3 import set_gloss_debias, set_vectors_fill   # noqa: E402


def load_gloss_dict(path, top=3):
    """{headword: "w1 w2 w3"} from a kaikki bag, most frequent English content words first.

    Three words rather than one: a bag averaged over a couple of senses is a better centroid than
    whichever sense Wiktionary happens to list first, and the layer already averages the pieces it
    can resolve. Beyond about three the tail is usage notes rather than meaning.
    """
    raw = json.load(open(path, encoding="utf-8"))
    return {k: " ".join(list(v)[:top]) for k, v in raw.items() if v}


def run(nlp, refs, lang, gloss=None, gloss_key="lemma", copy_lemma=True):
    """Predict over gold tokens with gold UPOS/FEATS -- the arm's DECLARED inputs -- plus glosses.

    Mirrors eval_generic_v2.run(); the only addition is Token._.gloss. The morph OBJECT is copied
    rather than its string, which is what preserves the unset/empty distinction (CLAUDE.md: sa,
    6.8 LAS).
    """
    examples, n_tok, n_gloss = [], 0, 0
    for ref in refs:
        ref._.tb_lang = lang
        pred = Doc(nlp.vocab, words=[t.text for t in ref],
                   spaces=[bool(t.whitespace_) for t in ref])
        for p, r in zip(pred, ref):
            p.pos = r.pos
            p.set_morph(r.morph)
            # ⚠ THE LEMMA IS COPIED ONLY WHERE THE FILL REGIME USES IT, and that is a claim about
            # what the caller has, not a convenience. `generic_corpus` copies UPOS, FEATS AND LEMMA
            # onto the predicted doc during training, so the `lemma` fill must too or the model
            # meets a regime it never trained on. But the DEPLOYMENT story is UPOS + a gloss: a user
            # of an unseen language has no lemmatiser, so handing the gloss fill a gold lemma column
            # would quietly score an input the contract does not offer.
            if copy_lemma:
                p.lemma = r.lemma
            n_tok += 1
            if gloss is not None:
                key = (r.lemma_ if gloss_key == "lemma" else r.text)
                if key and key != "_":
                    g = gloss.get(key) or gloss.get(key.lower())
                    if g:
                        p._.gloss = g
                        n_gloss += 1
        annotate(pred, lang)
        for _, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, ref))
    return examples, (n_gloss / n_tok if n_tok else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--gloss-dir", default="assets_vec/dict")
    ap.add_argument("--gloss-key", default="form", choices=("lemma", "form"))
    ap.add_argument("--fill", default="gloss", choices=("lemma", "gloss", "auto"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--pool", default="test", choices=("test", "train"))
    ap.add_argument("--lang", nargs="*", default=None)
    ap.add_argument("--gloss-debias", default=None,
                    help="assets_vec/gloss_shift_v3.npy -- subtract the measured source->English "
                         "displacement from GLOSS rows only. Estimated on a training language; "
                         "whether it generalises is what the held-out number answers.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))["languages"]
    langs = a.lang or sorted(l for l, v in man.items() if v["pool"] == a.pool)

    nlp = spacy.load(a.model)
    has_channel = True
    try:
        n = set_vectors_fill(nlp, a.fill)
        print(f"fill regime set to {a.fill!r} on {n} node(s)")
    except ValueError as e:
        # g3_base and g3_vec_ctl have no channel; that is a legitimate arm, not a mistake, but the
        # fill flag then means nothing and saying so beats printing it in the header.
        has_channel = False
        print(f"no lexical channel in this arm; --fill {a.fill!r} is inert ({str(e)[:60]}...)")
    if has_channel and a.gloss_debias:
        set_gloss_debias(nlp, a.gloss_debias)
        print(f"gloss de-bias applied from {a.gloss_debias}")

    rows, unscorable = [], []
    for lang in langs:
        refs = load_docs(a.corpus, lang, a.split, nlp.vocab)
        if not refs:
            continue
        refs = one_sentence_per_doc(refs, lang)
        gd = None
        if has_channel and a.fill in ("gloss", "auto"):
            p = os.path.join(a.gloss_dir, f"{lang}-en.json")
            if os.path.exists(p):
                gd = load_gloss_dict(p)
            else:
                unscorable.append(lang)
        # `gloss` alone is the pure deployment regime: UPOS, FEATS and a gloss, no lemma.
        ex, fill_rate = run(nlp, refs, lang, gloss=gd, gloss_key=a.gloss_key,
                            copy_lemma=(a.fill != "gloss"))
        if gd is not None and fill_rate == 0.0:
            # The failure the layer cannot see: a whole language whose channel was never filled.
            # Scoring it would report the no-channel number under the gloss fill's name.
            sys.exit(f"{lang}: --fill {a.fill} with a dictionary loaded, and NOT ONE token got a "
                     f"gloss. Wrong --gloss-key, or a dictionary keyed differently from this "
                     f"treebank's {a.gloss_key} column.")
        s = score(ex)
        rows.append(dict(lang=lang, tokens=sum(len(e.reference) for e in ex),
                         sents=len(ex), uas=s["uas"], las=s["las"], sents_f=s["sents_f"],
                         gloss_fill=round(fill_rate, 4),
                         gloss_source=("wiktionary" if gd else None),
                         genus=man[lang]["genus"], family=man[lang]["family"]))
        print(f"  {lang:5s} n={rows[-1]['tokens']:6d}  UAS {s['uas']*100:5.2f}  "
              f"LAS {s['las']*100:5.2f}  gloss fill {fill_rate:5.1%}")

    scored = [r for r in rows if r["gloss_source"]] if a.fill != "lemma" else rows
    macro = sum(r["las"] for r in scored) / len(scored) if scored else 0.0
    print(f"\nmacro LAS over {len(scored)} scored languages: {macro*100:.2f}")
    if unscorable:
        print(f"NOT scorable on the gloss fill (no Wiktionary extract): {unscorable}")

    out = dict(model=a.model, fill=a.fill, gloss_key=a.gloss_key, gloss_debias=a.gloss_debias, split=a.split, pool=a.pool,
               macro_las=macro, n_scored=len(scored), unscorable=unscorable, languages=rows)
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
