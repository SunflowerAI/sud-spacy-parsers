#!/usr/bin/env python3
"""Score the generic parser per language -- and, on request, the monolingual arms THROUGH THE SAME
HARNESS, which is the only way the comparison means anything.

WHY THIS EXISTS RATHER THAN `spacy evaluate`. Two reasons, and both are silent if ignored.

**1. The generic arm's inputs have to be put on the doc.** `spacy evaluate` builds the predicted doc
by running the model's tokenizer over raw text, and this model has neither a tokenizer worth running
nor a morphologiser: UPOS, FEATS and LEMMA are its declared INPUTS and must be supplied, exactly as
`sud.GenericCorpus.v1` supplies them during training. Evaluated the stock way it would read POS 0
and an unset MORPH on every token -- a model that loads, runs, and scores like a different model.
`Doc._.tb_lang` likewise: without it the vector channel raises (by design) rather than quietly
reading nothing.

**2. THE PUBLISHED MONOLINGUAL LAS FIGURES ARE NOT COMPARABLE TO THIS ARM'S, and subtracting them
would be a straightforward error.** The released arms predict up to 120 deprels including the
`@`-subtypes; the generic arm predicts the 27 that survive stripping. A coarser label set is
strictly easier -- every `mod@relcl` vs `mod` confusion the monolingual arm can make simply does not
exist here -- so `metrics_release_*.json` sits on a different target and cannot be differenced
against these numbers. `--monolingual` runs the released arm over the SAME normalised gold and
strips `@` from its predictions before scoring, so both models are asked the same question. This is
the repo's own meta-lesson: never compare numbers from two different harnesses; the lzh "7 LAS"
claim and the "+2.51 zh raw LAS" claim were both individually-correct numbers in an invalid
comparison.

⚠ WHAT THE COMPARISON IS AND IS NOT, per language. The generic arm trains on a TYPOLOGICALLY
BALANCED SAMPLE, so for a large treebank it has seen a fraction of what the monolingual arm has
(la 15 k of 587 k training tokens) and is expected to lose. For a small one it has seen ALL of it
(ta, te and yue cap out below their group's budget) and the difference is then exactly the question
worth asking: does the other twelve languages' syntax help a treebank that has 5 097 training tokens
of its own? Read the two halves of the table differently -- the manifest's `train_subsampled` flag
says which case each language is.

Punctuation is excluded from LAS/UAS, matching `spacy.parser_scorer.v1` and the training logs.

    .venv/bin/python scripts/eval_generic.py training_generic/model-best
    .venv/bin/python scripts/eval_generic.py training_generic/model-best --monolingual ta te yue
    .venv/bin/python scripts/eval_generic.py training_generic/model-best --lang ja --json m.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                                                    # noqa: E402
from spacy.scorer import Scorer                                                 # noqa: E402
from spacy.tokens import Doc, DocBin                                            # noqa: E402
from spacy.training import Example                                              # noqa: E402

import generic_code                        # noqa: E402,F401  (registers everything the arm needs)
from prep_generic import GROUPS, LANGS      # noqa: E402

MONO_ARMS = {
    # Taken from `pkg()`'s per-language `base=` defaults in scripts/package_sud.sh, which is the
    # authority on which arm each wheel actually ships -- NOT from directory names. This repo has
    # paid four times for a default that named a plausible-looking arm one generation sideways
    # (CLAUDE.md hazard 2), and a comparison against the wrong arm would be wrong in a way no
    # number here would reveal.
    "en":  "training_en_sud_xpos/model-best",
    "zh":  "training_zh_trad_lemma/model-best",
    "yue": "training_yue_sud_xpos",                 # already a model dir, no model-best
    "lzh": "training_lzh_seg_sud_xw",               # the SENTENCE-SEGMENTING arm
    "fa":  "training_fa_vocal_sud_xpos/model-best",
    "ar":  "training_ar_vocal_sud_idiom/model-best",
    "la":  "training_la_lemvec_sud/model-best",
    "id":  "training_id_sud_xpos/model-best",
    "ko":  "training_ko_eojeol_lemma/model-best",
    "ja":  "training_ja_lemma/model-best",          # the `*` fallback in package_sud.sh
    "ta":  "training_ta_both_lemvec/model-best",    # docs/dravidian.md
    "te":  "training_te_lemma/model-best",          # docs/dravidian.md
    "sa":  "training_sa_mp2_sub_s1/model-best",
}


#: FEATS a released arm's TOKENISER supplies at inference, which a gold-token harness bypasses.
#: sa's parser READS `Compound=Yes` as a model input -- `sa_tokenizer` reads it off the CSL join
#: marker at P/R 0.9998 (`docs/sanskrit.md`, +1.30 LAS) -- so an evaluation that hands the arm gold
#: tokens without it denies the arm an input it always has in production. `eval_sa_compound.py`
#: exists for exactly this and its `--reader norm` is what `metrics_sa_*_Vedic.json` was measured
#: with; copying the feature across here is the same move.
TOKENISER_FEATS = {"sa": ("Compound",)}


def stamp_tokeniser_feats(pred, ref, lang):
    keys = TOKENISER_FEATS.get(lang)
    if not keys:
        return pred
    for p_tok, r_tok in zip(pred, ref):
        vals = [f"{k}={v}" for k in keys for v in r_tok.morph.get(k)]
        if vals:
            p_tok.set_morph("|".join(vals))
    return pred


def gold_docs(corpus_dir, lang, split, vocab):
    p = os.path.join(corpus_dir, f"{lang}-{split}.spacy")
    if not os.path.exists(p):
        return None
    return list(DocBin().from_disk(p).get_docs(vocab))


def strip_at(dep: str) -> str:
    return dep.split("@", 1)[0]


def make_input(vocab, reference, lang, with_tagging=True):
    """The predicted doc as the arm's input regime defines it: gold tokens, gold UPOS/FEATS/LEMMA.

    `with_tagging=False` is for a monolingual arm, which predicts its own morphology and must be
    given the raw tokens only -- handing it gold FEATS would be a different (and much easier)
    experiment than the one it was trained for.
    """
    doc = Doc(vocab,
              words=[t.text for t in reference],
              spaces=[bool(t.whitespace_) for t in reference])
    doc._.tb_lang = lang
    if with_tagging:
        for tok, ref in zip(doc, reference):
            tok.pos = ref.pos
            tok.set_morph(ref.morph)
            tok.lemma = ref.lemma
    return doc


def coarsen(doc):
    """Rewrite every predicted deprel to its `@`-stripped form, in place."""
    for tok in doc:
        d = strip_at(tok.dep_)
        if d != tok.dep_:
            tok.dep_ = d
    return doc


def score(examples):
    deps = Scorer.score_deps(examples, "dep", ignore_labels=("p", "punct"))
    sents = Scorer.score_spans(examples, "sents")
    return {"uas": deps["dep_uas"], "las": deps["dep_las"],
            "per_type": deps.get("dep_las_per_type") or {},
            "sents_f": sents.get("sents_f")}


def one_sentence_per_doc(refs, lang):
    """Split multi-sentence reference docs into one doc per sentence -- GOLD segmentation.

    Why it is worth a separate condition. The arm is trained on ten-sentence docs and has to find
    sentence boundaries itself, so its LAS confounds attachment with segmentation. On a language it
    has never seen, that confound dominates: the zero-shot ar and lzh arms score SENTS_F 0.23 and
    0.31, and a parser that cannot tell where a sentence ends cannot attach anything correctly
    either. Handing it gold boundaries separates "it does not know this language's syntax" from
    "it does not know where this language's sentences stop", which are different problems with
    different fixes. This is also the `gold_preproc` condition every other metric in this project
    is reported under, so these numbers are the ones comparable to `metrics_release_*.json`.
    """
    out = []
    for ref in refs:
        for sent in ref.sents:
            d = sent.as_doc(copy_user_data=False)
            d._.tb_lang = lang
            out.append(d)
    return out


def run_generic(model, corpus_dir, langs, split, gold_sents=False):
    nlp = spacy.load(model)
    out = {}
    for lang in langs:
        refs = gold_docs(corpus_dir, lang, split, nlp.vocab)
        if not refs:
            continue
        if gold_sents:
            refs = one_sentence_per_doc(refs, lang)
        examples = []
        for ref in refs:
            ref._.tb_lang = lang
            pred = make_input(nlp.vocab, ref, lang, with_tagging=True)
            for name, proc in nlp.pipeline:
                pred = proc(pred)
            examples.append(Example(pred, ref))
        out[lang] = score(examples)
        out[lang]["tokens"] = sum(len(d) for d in refs)
    return out


_SEG_CODE_LOADED = [False]


def load_released_registrations():
    """Import `seg_code` -- the module that registers every released arm's factories and languages.

    Deliberately NOT imported by `generic_code.py`: the generic arm has no tokenizer of its own and
    needs none of it, and pulling in thirteen languages' optional dependencies to train a model that
    reads no strings would only add ways for the run to fail. It IS needed the moment a released arm
    is loaded -- `yue` is a custom language, `sud_tagger` a custom factory, `sud.LemmaVecEmbed.v1` a
    custom architecture -- and each of those is an E002/E048/E893 rather than a wrong number.
    """
    if not _SEG_CODE_LOADED[0]:
        import seg_code                                                       # noqa: F401
        _SEG_CODE_LOADED[0] = True


def tag_with(arm_path, corpus_dir, lang, split, vocab_donor, gold_sents=False):
    """Run a released arm's OWN morphologiser/lemmatiser over the gold tokens.

    This is what makes the generic-vs-monolingual comparison honest. The generic arm's declared
    inputs are UPOS/FEATS/LEMMA, and scoring it on GOLD ones against a monolingual arm that has to
    predict its own is not a comparison -- it hands one side the answer to a subproblem the other
    side must solve. Tamil's FEATS sit on 88 % of its tokens, so on that language the gold-input
    condition could account for the entire gap on its own.

    Returns one predicted-tagging doc per reference doc, token-aligned.
    """
    load_released_registrations()
    nlp = spacy.load(arm_path)
    keep = [(n, p) for n, p in nlp.pipeline
            if n in ("tok2vec", "morphologizer", "lemmatizer", "tagger")]
    refs = gold_docs(corpus_dir, lang, split, vocab_donor)
    if gold_sents:
        refs = one_sentence_per_doc(refs, lang)
    out = []
    for ref in refs:
        d = Doc(nlp.vocab, words=[t.text for t in ref],
                spaces=[bool(t.whitespace_) for t in ref])
        stamp_tokeniser_feats(d, ref, lang)
        for _, proc in keep:
            d = proc(d)
        out.append(d)
    return out


def run_generic_predicted(model, corpus_dir, lang, split, tagged, gold_sents=False):
    """The generic arm, but reading a released morphologiser's PREDICTED UPOS/FEATS/LEMMA."""
    nlp = spacy.load(model)
    refs = gold_docs(corpus_dir, lang, split, nlp.vocab)
    if gold_sents:
        refs = one_sentence_per_doc(refs, lang)
    examples = []
    for ref, src in zip(refs, tagged):
        ref._.tb_lang = lang
        pred = Doc(nlp.vocab, words=[t.text for t in ref],
                   spaces=[bool(t.whitespace_) for t in ref])
        pred._.tb_lang = lang
        for tok, s_tok in zip(pred, src):
            if s_tok.pos_:
                tok.pos_ = s_tok.pos_
            tok.set_morph(str(s_tok.morph) or None)
            # identity fallback, as everywhere else: sa's vectors are keyed by lemma and a blank
            # one would be a silently all-OOV language
            tok.lemma_ = s_tok.lemma_ or tok.text
        for _, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, ref))
    return score(examples)


def run_monolingual(arm_path, corpus_dir, lang, split, vocab_donor, gold_sents=False):
    """The released arm over the same gold, its predictions coarsened to the 27-label inventory.

    Its own tokenizer is bypassed (gold tokens, as `--gold-preproc` does for every non-en arm in
    this project), and its own morphologiser runs -- so this is the arm as it actually ships,
    scored on the generic arm's target.
    """
    load_released_registrations()
    nlp = spacy.load(arm_path)
    refs = gold_docs(corpus_dir, lang, split, vocab_donor)
    if not refs:
        return None
    if gold_sents:
        refs = one_sentence_per_doc(refs, lang)
    examples = []
    for ref in refs:
        pred = Doc(nlp.vocab,
                   words=[t.text for t in ref],
                   spaces=[bool(t.whitespace_) for t in ref])
        stamp_tokeniser_feats(pred, ref, lang)
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        coarsen(pred)
        # The reference must live in the SAME vocab as the predicted doc for Example to align them.
        ref_here = Doc(nlp.vocab).from_bytes(ref.to_bytes())
        coarsen(ref_here)
        examples.append(Example(pred, ref_here))
    return score(examples)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="training_generic*/model-best")
    ap.add_argument("--corpus", default="corpus_generic")
    ap.add_argument("--manifest", default="assets_generic/manifest.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lang", nargs="*", default=None, help="default: every language present")
    ap.add_argument("--monolingual", nargs="*", default=[],
                    help="also score these languages' released arms through this harness; "
                         "`all` for every one that resolves")
    ap.add_argument("--gold-sents", action="store_true",
                    help="one gold sentence per doc, so segmentation cannot confound attachment "
                         "-- the `gold_preproc` condition the rest of this project reports under")
    ap.add_argument("--per-label", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    langs = a.lang or [l for l in LANGS
                       if os.path.exists(os.path.join(a.corpus, f"{l}-{a.split}.spacy"))]
    manifest = {}
    if os.path.exists(a.manifest):
        manifest = json.load(open(a.manifest, encoding="utf-8")).get("languages", {})

    print(f"generic arm: {a.model}   split: {a.split}   punctuation excluded from UAS/LAS"
          + ("   GOLD sentence boundaries" if a.gold_sents else "   model-predicted sentences")
          + "\n")
    res = run_generic(a.model, a.corpus, langs, a.split, gold_sents=a.gold_sents)

    group_of = {l: g for g, ms in GROUPS.items() for l in ms}
    print(f"{'lang':5} {'group':14} {'train seen':>10} {'all?':>5} {'test tok':>9} "
          f"{'UAS':>7} {'LAS':>7} {'SENTS_F':>8}")
    for lang in langs:
        m = manifest.get(lang, {})
        seen = m.get("train_sampled", 0)
        allof = "no" if m.get("train_subsampled") else "yes"
        if m.get("held_out"):
            seen, allof = 0, "0-shot"
        r = res[lang]
        print(f"{lang:5} {group_of.get(lang,''):14} {seen:>10} {allof:>5} {r['tokens']:>9} "
              f"{100*r['uas']:>7.2f} {100*r['las']:>7.2f} "
              f"{100*(r['sents_f'] or 0):>8.2f}")

    tot = sum(res[l]["tokens"] for l in langs)
    macro_las = sum(res[l]["las"] for l in langs) / len(langs)
    micro_las = sum(res[l]["las"] * res[l]["tokens"] for l in langs) / tot
    print(f"\nmacro-average LAS {100*macro_las:.2f}   micro-average LAS {100*micro_las:.2f}")
    print("macro is the one to read: micro is dominated by la and lzh, whose test sets are "
          "4-8x the others'.")

    out = {"model": a.model, "split": a.split, "gold_sents": a.gold_sents, "generic": res,
           "macro_las": macro_las, "micro_las": micro_las}

    mono_langs = list(langs) if a.monolingual == ["all"] else list(a.monolingual)
    if mono_langs and not a.gold_sents:
        sys.exit(
            "--monolingual requires --gold-sents, and this is not a style preference.\n"
            "The generic arm is trained on ten-sentence docs and learns to segment; the released\n"
            "monolingual arms are trained and reported under `gold_preproc`, i.e. one sentence per\n"
            "doc, and most of them never learned to START a sentence. Handing them running text\n"
            "makes them segment something they were never trained to segment, and the LAS that\n"
            "results is not theirs: the sa arm reads 37.23 that way against 49.73 under\n"
            "`spacy evaluate --gold-preproc` on the same file. Re-run with --gold-sents.")
    if mono_langs:
        print(f"\n=== the released monolingual arms, SAME harness, predictions coarsened to the "
              f"27-label inventory ===")
        print(f"{'lang':5} {'arm':34} {'mono':>7} {'gen/pred':>9} {'delta':>7} {'gen/gold':>9}")
        out["monolingual"] = {}
        for lang in mono_langs:
            arm = MONO_ARMS.get(lang, f"training_{lang}/model-best")
            if not os.path.exists(arm):
                print(f"{lang:5} {arm:34} {'absent':>7}")
                continue
            try:
                nlp_g = spacy.load(a.model)
                r = run_monolingual(arm, a.corpus, lang, a.split, nlp_g.vocab,
                                    gold_sents=a.gold_sents)
                tagged = tag_with(arm, a.corpus, lang, a.split, nlp_g.vocab,
                                  gold_sents=a.gold_sents)
                rp = run_generic_predicted(a.model, a.corpus, lang, a.split, tagged,
                                           gold_sents=a.gold_sents)
            except Exception as e:                       # a missing optional dep, e.g. mecab-ko
                print(f"{lang:5} {arm:34} {type(e).__name__}: {str(e)[:34]}")
                continue
            if r is None:
                continue
            g = res[lang]["las"]
            out["monolingual"][lang] = {
                "arm": arm,
                "mono_las": r["las"], "mono_uas": r["uas"],
                "generic_predicted_las": rp["las"], "generic_predicted_uas": rp["uas"],
                "generic_gold_las": g,
            }
            print(f"{lang:5} {arm:34} {100*r['las']:>7.2f} {100*rp['las']:>9.2f} "
                  f"{100*(rp['las'] - r['las']):>+7.2f} {100*g:>9.2f}")
        print("""
`mono`      the released arm end to end: its own tokens' tags, its own parse, coarsened to 27 labels.
`gen/pred`  the generic arm reading THAT ARM'S predicted UPOS/FEATS/LEMMA. The honest comparison:
            both sides now solve the same problem from the same information.
`delta`     gen/pred minus mono. This is the number to quote.
`gen/gold`  the generic arm on GOLD tags -- its defined input regime, and the right column for
            comparing generic arms to each other, but NOT to a monolingual arm, which would be
            handed the answer to a subproblem the other side has to solve.

Positive `delta` on a language whose `all?` column reads `yes` is the low-resource uplift claim: the
generic arm saw all of that treebank's training data, so nothing about data volume explains it. On a
language reading `no` the generic arm saw a fraction of the treebank and a negative delta is the
price of balance, not a finding.""")

    if a.per_label:
        print("\n=== per-label LAS F, generic arm ===")
        labels = sorted({k for l in langs for k in res[l]["per_type"]})
        print(f"{'label':14} " + " ".join(f"{l:>6}" for l in langs))
        for lab in labels:
            row = []
            for l in langs:
                v = res[l]["per_type"].get(lab)
                row.append(f"{100*v['f']:>6.1f}" if v else f"{'-':>6}")
            print(f"{lab:14} " + " ".join(row))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
