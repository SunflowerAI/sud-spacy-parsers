#!/usr/bin/env python3
"""Score a v2 arm on the held-out languages. Zero-shot is the headline; everything else is context.

Forked from `eval_generic.py`. What went, and why: `MONO_ARMS`, `tag_with`, `run_monolingual` and
`load_released_registrations` are all gone, because no released wheel exists for Basque, Wolof or
K'iche' and reaching for one would be the cross-harness comparison this repo has recorded twice
(the lzh "7 LAS" and "+2.51 zh raw LAS" claims were both artefacts of scoring two arms through
different harnesses).

⚠ **GOLD SENTENCE BOUNDARIES ARE THE DEFAULT HERE, INVERTING v1.** Zero-shot SENTS_F was 0.23-0.31
in v1, so without gold boundaries the number measures segmentation rather than attachment. Both
columns are printed and neither is ever quoted for the other -- `gold_preproc` hiding exactly this
is a standing hazard in CLAUDE.md.

⚠ **THE HEADLINE IS MACRO OVER TEST LANGUAGES.** Micro is printed and deprecated: the test set runs
from Xavante at 2 239 tokens to Basque at 20 164, so micro is a weighted average of whichever
treebanks happened to be large.

⚠ **A COLLAPSED-LABEL LAS IS REPORTED BESIDE THE PLAIN ONE.** This repo's thirteen treebanks carry an
LLM pass that splits `udep` into `comp:obl` and `mod`; the other seventy and every test treebank do
not. That puts ~17 % of training tokens on a different policy for the single most confusable label
group, so the figure with those three merged is the one immune to it.

⚠ **THE HEADLINE IS REFUSED UNTIL THE BASELINES EXIST.** 56.5 % once read as a result in this repo
until a 58.5 % majority baseline was put beside it.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from spacy.training import Example  # noqa: E402
from spacy.scorer import Scorer  # noqa: E402

import generic_code_v2  # noqa: E402,F401  (registers the layer and the reader)
from generic_corpus import annotate  # noqa: E402  (the ONE definition of the input regime)

#: The `udep` relabel's three-way confusion. Merged, the label-policy split cannot move the number.
POLICY_GROUP = ("udep", "comp:obl", "mod")
COLLAPSED = "MOD-OBL-UDEP"


def load_docs(corpus_dir, lang, split, vocab):
    p = os.path.join(corpus_dir, f"{lang}-{split}.spacy")
    if not os.path.exists(p):
        return []
    return list(DocBin().from_disk(p).get_docs(vocab))


def score(examples, collapse=False):
    if collapse:
        examples = [Example(collapse_doc(e.predicted), collapse_doc(e.reference))
                    for e in examples]
    deps = Scorer.score_deps(examples, "dep", ignore_labels=("p", "punct"))
    sents = Scorer.score_spans(examples, "sents")
    return {"uas": deps["dep_uas"], "las": deps["dep_las"],
            "per_type": {str(k): v for k, v in (deps.get("dep_las_per_type") or {}).items()},
            "sents_f": sents.get("sents_f")}


def collapse_doc(doc):
    """Re-label the `udep`/`comp:obl`/`mod` triple to one symbol, keeping the tree.

    Built through the constructor: writing `is_sent_start` after the heads raises E043, and writing
    the heads alone leaves SENT_START unset so `doc.sents` raises E030 in the scorer.
    """
    return Doc(doc.vocab,
               words=[t.text for t in doc],
               spaces=[bool(t.whitespace_) for t in doc],
               heads=[t.head.i for t in doc],
               deps=[COLLAPSED if t.dep_ in POLICY_GROUP else (t.dep_ or "dep") for t in doc],
               sent_starts=[bool(t.is_sent_start) for t in doc])


def one_sentence_per_doc(refs, lang):
    """One Doc per gold sentence, re-stamping `tb_lang` (as_doc drops user data)."""
    out = []
    for ref in refs:
        for sent in ref.sents:
            d = sent.as_doc(copy_user_data=False)
            d._.tb_lang = lang
            out.append(d)
    return out


def upos_dist(refs):
    """The tag distribution of the documents being scored.

    Errors are drawn from this rather than uniformly, so a simulated tagger confuses NOUN with
    PROPN rather than with PUNCT. Taken from the refs themselves: an earlier version globbed the
    `.spacy` corpus directory for `.conllu` files, found none, and produced an empty distribution.
    """
    c = collections.Counter(t.pos_ for d in refs for t in d if t.pos_)
    tags = sorted(c)
    tot = max(sum(c.values()), 1)
    return tags, [c[t] / tot for t in tags]


def run(nlp, refs, lang, no_feats=False, upos_noise=0.0, seed=0):
    """Predict over gold tokens with gold UPOS/FEATS -- the arm's DECLARED inputs."""
    import random as _r
    rng = _r.Random(seed)
    tags, probs = upos_dist(refs) if upos_noise else ([], [])
    examples = []
    for ref in refs:
        ref._.tb_lang = lang
        pred = Doc(nlp.vocab, words=[t.text for t in ref],
                   spaces=[bool(t.whitespace_) for t in ref])
        for p, r in zip(pred, ref):
            p.pos = r.pos
            if upos_noise and rng.random() < upos_noise:
                # Draw a WRONG tag from the corpus distribution: a uniform draw would mostly
                # produce tags no real tagger would ever emit here, and would overstate the damage.
                for _ in range(8):
                    t = rng.choices(tags, probs)[0]
                    if t != r.pos_:
                        p.pos_ = t
                        break
            # Copy the morph OBJECT, not its string: that preserves the unset/empty distinction
            # exactly as the reference has it (CLAUDE.md; sa, 6.8 LAS).
            if not no_feats:
                p.set_morph(r.morph)
        annotate(pred, lang)
        for _, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, ref))
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--typology", default="assets_typ/typology_v2.json")
    ap.add_argument("--baseline", default="metrics/generic_v2/baseline.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lang", nargs="*", default=None)
    ap.add_argument("--held-in", action="store_true",
                    help="also score the training languages' dev, to quantify the zero-shot gap")
    ap.add_argument("--no-gold-sents", action="store_true",
                    help="let the arm find its own sentence boundaries. Reported, never quoted "
                         "for the gold-boundary figure")
    ap.add_argument("--typology-override", default=None,
                    help="swap the profiles baked into the trained layer for those in this file. "
                         "Used ONLY to reproduce v1's ORACLE condition -- test profiles taken from "
                         "the test treebank's own trees -- so that 'typology does not help' can be "
                         "separated from 'our external profiles are too noisy to help'. Any number "
                         "produced this way is an upper bound and must be labelled as one.")
    ap.add_argument("--upos-noise", type=float, default=0.0,
                    help="corrupt this fraction of UPOS tags, simulating a tagger of that error "
                         "rate. Replacements are drawn from the corpus tag distribution.")
    ap.add_argument("--no-feats", action="store_true",
                    help="withhold FEATS from the model, simulating a language with no "
                         "morphological analyser. UPOS is still supplied.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    typ = json.loads(pathlib.Path(a.typology).read_text(encoding="utf-8"))["languages"]
    base = None
    if os.path.exists(a.baseline):
        base = json.loads(pathlib.Path(a.baseline).read_text(encoding="utf-8"))

    nlp = spacy.load(a.model)
    if a.typology_override:
        blob = json.loads(pathlib.Path(a.typology_override).read_text(encoding="utf-8"))
        n = 0
        for _, proc in nlp.pipeline:
            mdl = getattr(proc, "model", None)
            if mdl is None:
                continue
            for node in mdl.walk():
                if node.name != "extract_typology":
                    continue
                vecs = dict(node.attrs["ty_vecs"])
                dim = int(node.attrs["ty_dim"])
                for lg, rec in blob["languages"].items():
                    bits = [float(x) for x in rec["bits"]]
                    if dim == 12:
                        bits += [float(bits[i] or bits[i + 1]) for i in (0, 2, 4, 6)]
                    vecs[lg] = bits
                node.attrs["ty_vecs"] = vecs
                node.attrs["ty_langs"] = sorted(vecs)
                n += 1
        if not n:
            sys.exit("--typology-override: this model has no typology channel to override")
        print(f"⚠ ORACLE CONDITION: profiles overridden from {a.typology_override} "
              f"({n} node(s)). Every number below is an UPPER BOUND.")
    test = a.lang or sorted(k for k, v in man.items() if v["pool"] == "test")

    out = {"model": a.model, "split": a.split, "languages": {}}
    rows = []
    print(f"{'lang':6s} {'family':16s} {'cell':20s} {'tok':>6s} {'UAS':>6s} {'LAS':>6s} "
          f"{'LASraw':>7s} {'SENTS':>6s} {'coll':>6s} {'FEATS':>6s}")
    for lang in test:
        refs = load_docs(a.corpus, lang, a.split, nlp.vocab)
        if not refs:
            continue
        gold_sents = one_sentence_per_doc(refs, lang)
        ex_gs = run(nlp, gold_sents, lang, a.no_feats, a.upos_noise)
        s_gs = score(ex_gs)
        s_coll = score(ex_gs, collapse=True)
        # The same model over the multi-sentence docs, finding its own boundaries.
        s_raw = score(run(nlp, refs, lang, a.no_feats, a.upos_noise))
        v = man[lang]
        rec = {"tokens": sum(len(d) for d in gold_sents),
               "uas": s_gs["uas"], "las": s_gs["las"], "las_collapsed": s_coll["las"],
               "las_predicted_sents": s_raw["las"], "sents_f": s_raw["sents_f"],
               "per_type": s_gs["per_type"],
               "bits": typ[lang]["bits"], "sources": typ[lang]["sources"],
               "cell": v["cell"], "family": v["family"], "genus": v["genus"],
               "feats_fill": v["feats_fill"], "corpus_source": v["source"]}
        out["languages"][lang] = rec
        rows.append((lang, rec))
        print(f"{lang:6s} {v['family'][:15]:16s} {v['cell']:20s} {rec['tokens']:6d} "
              f"{100 * rec['uas']:6.2f} {100 * rec['las']:6.2f} "
              f"{100 * rec['las_predicted_sents']:7.2f} {100 * (rec['sents_f'] or 0):6.2f} "
              f"{100 * rec['las_collapsed']:6.2f} {v['feats_fill']:6.2f}")

    if not rows:
        sys.exit("no test corpora found")
    macro = sum(r["las"] for _, r in rows) / len(rows)
    macro_coll = sum(r["las_collapsed"] for _, r in rows) / len(rows)
    tot = sum(r["tokens"] for _, r in rows)
    micro = sum(r["las"] * r["tokens"] for _, r in rows) / tot
    out["macro_las"], out["macro_las_collapsed"], out["micro_las"] = macro, macro_coll, micro

    # By corpus provenance: a UD conversion and a natively-annotated corpus are annotated to
    # different depths (ja GSD carries `comp:obl` on 18 tokens against `udep` on 21 530).
    by_src = collections.defaultdict(list)
    for _, r in rows:
        by_src[r["corpus_source"]].append(r["las"])

    print(f"\nMACRO LAS over {len(rows)} test languages : {100 * macro:6.2f}")
    print(f"  collapsed (udep/comp:obl/mod merged) : {100 * macro_coll:6.2f}   "
          f"<- immune to the label-policy split")
    print(f"  micro (deprecated, size-weighted)    : {100 * micro:6.2f}")
    for src, v in sorted(by_src.items()):
        print(f"  {src:20s} {len(v):3d} langs : {100 * sum(v) / len(v):6.2f}")

    if a.held_in:
        hi = []
        for lang in sorted(k for k, v in man.items() if v["pool"] == "train"):
            refs = load_docs(a.corpus, lang, "dev", nlp.vocab)
            if not refs:
                continue
            s = score(run(nlp, one_sentence_per_doc(refs, lang), lang))
            hi.append(s["las"])
        if hi:
            out["held_in_macro_las"] = sum(hi) / len(hi)
            print(f"\nheld-in macro LAS (train languages' dev, {len(hi)} langs): "
                  f"{100 * out['held_in_macro_las']:6.2f}")
            print(f"  zero-shot gap: {100 * (out['held_in_macro_las'] - macro):6.2f}")

    if base is None:
        print("\n⚠ NO BASELINE FILE. The headline is withheld: a zero-shot LAS is not "
              "interpretable without the trivial baselines beside it, and this repo has already "
              "reported 56.5 % as a result against a 58.5 % constant.\n"
              "  Run: .venv/bin/python scripts/baseline_generic.py")
    else:
        bar = base["bar"]
        print(f"\nbaselines (macro LAS): " + "  ".join(
            f"{k} {100 * v:.2f}" for k, v in sorted(base["macro_las"].items())))
        delta = 100 * (macro - bar["macro_las"])
        print(f"the bar is {bar['baseline']} at {100 * bar['macro_las']:.2f}; this arm is "
              f"{delta:+.2f} against it")
        out["baseline_bar"] = bar
        out["delta_vs_bar"] = macro - bar["macro_las"]

    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
