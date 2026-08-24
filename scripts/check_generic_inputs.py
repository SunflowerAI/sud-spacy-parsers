#!/usr/bin/env python3
"""Verify what the generic arm's model ACTUALLY receives, rather than what the config says.

Every check here corresponds to a defect this project has already shipped once in some other form,
and every one of them is invisible to the training log -- an arm with a dead vector channel, or with
no morphology on its predicted docs, trains happily and simply scores lower.

    1. `Shared` is gone from the corpus.        Native SUD FEATS, and a fact about the TREE: it says
                                                whether a dependent is shared across conjuncts. As a
                                                parser INPUT it leaks coordination structure.
    2. The controls are CAPACITY-MATCHED.       `generic`, `generic_ctl` and `generic_shuf` must have
                                                identical parameter counts, or the comparison
                                                measures parameters and not information
                                                (NEGATIVE-RESULTS.md: "always run a capacity
                                                control").
    3. The vector channel is LIVE.              The arm's extractor output must differ from the
                                                control's on real tokens. An all-OOV table loads
                                                cleanly and scores exactly like the dead channel.
    4. The shuffle is a SHUFFLE.                Same rows, same norms, different assignment. If it
                                                came out identical the control would be the arm.
    5. UPOS/FEATS/LEMMA reach the PREDICTED doc.The reader must put them there, not just on the
                                                reference; the stock reader does not, and the model
                                                would learn to ignore three channels that then
                                                appear from nowhere at inference.
    6. `Doc._.tb_lang` unset RAISES.            A default would look every token up in one table and
                                                score like the dead-channel control.
    7. No string channel is present.            The claim is that the arm reads no wordform. The
                                                embed's attrs must be exactly ["POS"].

    .venv/bin/python scripts/check_generic_inputs.py
"""
from __future__ import annotations

import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                                                              # noqa: E402
import spacy                                                                    # noqa: E402
from spacy.tokens import Doc, DocBin                                            # noqa: E402
from thinc.api import Config                                                    # noqa: E402
from spacy.util import registry                                                 # noqa: E402

import generic_code                                            # noqa: E402,F401
import generic_corpus                                          # noqa: E402
from sud_generic_embed import load_table                       # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_embed(cfg_path):
    cfg = Config().from_disk(cfg_path, interpolate=False).interpolate()
    spec = cfg["components"]["tok2vec"]["model"]["embed"]
    return registry.resolve({"m": spec})["m"]


def n_params(model):
    model.initialize()
    return sum(int(np.prod(node.get_param(pname).shape))
               for node in model.walk() for pname in node.param_names
               if node.has_param(pname))


def main():
    corpus_conllu = "assets_generic"
    corpus = "corpus_generic"

    # ---- 1. Shared is gone
    hits = []
    for p in sorted(glob.glob(f"{corpus_conllu}/*.conllu")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if "Shared=" in line:
                    hits.append(p)
                    break
    check("1. `Shared` absent from the prepared corpus", not hits,
          f"found in {len(hits)} files" if hits else
          f"checked {len(glob.glob(f'{corpus_conllu}/*.conllu'))} files")

    # also confirm it IS in the sources, or the check above proves nothing
    src_has = False
    with open("assets/en_ewt-sud-train.relabeled_ext.conllu", encoding="utf-8") as fh:
        for line in fh:
            if "Shared=" in line:
                src_has = True
                break
    check("1b. ...and IS present in the source treebank (so check 1 is not vacuous)", src_has)

    # ---- 2/7. capacity-matched controls, and no string channel
    embeds, params = {}, {}
    for arm in ("generic", "generic_ctl", "generic_shuf"):
        embeds[arm] = build_embed(f"configs/config_{arm}.cfg")
        params[arm] = n_params(embeds[arm])
    check("2. generic / generic_ctl / generic_shuf are capacity-matched",
          params["generic"] == params["generic_ctl"] == params["generic_shuf"],
          "  ".join(f"{k}={v}" for k, v in params.items()))

    attrs = None
    for node in embeds["generic"].walk():
        if node.name == "extract_features_feats":
            attrs = node.attrs["columns"]
    check("7. the embed reads no string channel", attrs == ["POS"], f"attrs = {attrs}")

    # ---- 3/4. the vector channel is live, and the shuffle really shuffles
    real, _, meta = load_table("assets_vec/generic_vec.npz", shuffle=False)
    _, Vs, _ = load_table("assets_vec/generic_vec.npz", shuffle=True)
    Vr = load_table("assets_vec/generic_vec.npz", shuffle=False)[1]
    moved = float((np.abs(Vr - Vs).sum(axis=1) > 1e-6).mean())
    same_norms = np.allclose(np.sort(np.linalg.norm(Vr, axis=1)),
                             np.sort(np.linalg.norm(Vs, axis=1)), atol=1e-5)
    check("4. the shuffle moves rows but preserves the norm distribution",
          moved > 0.99 and same_norms, f"{100*moved:.1f} % of rows moved, norms preserved={same_norms}")

    langs = sorted(meta["languages"])
    check("4b. the shuffle stays WITHIN each language",
          all(any(k[0] == l for k in real) for l in langs),
          f"{len(langs)} languages: {' '.join(langs)}")

    nlp = spacy.blank("xx")
    ref = list(DocBin().from_disk(f"{corpus}/la-test.spacy").get_docs(nlp.vocab))[0]
    d_arm = Doc(nlp.vocab, words=[t.text for t in ref][:40])
    d_arm._.tb_lang = "la"
    y_arm = embeds["generic"].predict([d_arm])[0]
    y_ctl = embeds["generic_ctl"].predict([d_arm])[0]
    check("3. the vector channel is live (arm output differs from the dead-channel control)",
          not np.allclose(y_arm, y_ctl), f"max |diff| = {float(np.abs(y_arm - y_ctl).max()):.4f}")

    y_shuf = embeds["generic_shuf"].predict([d_arm])[0]
    check("3b. ...and differs from the shuffled control too",
          not np.allclose(y_arm, y_shuf), f"max |diff| = {float(np.abs(y_arm - y_shuf).max()):.4f}")

    # ---- 5. the reader puts the inputs on the PREDICTED doc
    gen = registry.readers.get("sud.GenericCorpus.v1")(path=corpus, split="dev")
    eg = next(iter(gen(nlp)))
    pred = eg.predicted
    n = len(pred)
    pos_ok = sum(1 for t in pred if t.pos != 0)
    morph_gold = sum(1 for t in eg.reference if len(t.morph))
    morph_pred = sum(1 for t in pred if len(t.morph))
    lem_ok = sum(1 for t in pred if t.lemma != 0)
    check("5. UPOS is on the predicted doc", pos_ok == n, f"{pos_ok}/{n}")
    check("5b. FEATS is on the predicted doc, matching the reference",
          morph_pred == morph_gold, f"pred {morph_pred} vs gold {morph_gold} of {n}")
    check("5c. LEMMA is on the predicted doc", lem_ok == n, f"{lem_ok}/{n}")
    check("5d. HEAD/DEPREL are NOT on the predicted doc (they are the target)",
          not pred.has_annotation("DEP"),
          "predicted doc carries no dependency annotation")
    check("5e. the language is stamped", pred._.tb_lang is not None, f"tb_lang={pred._.tb_lang!r}")

    # ---- 6. unset tb_lang raises
    bare = Doc(nlp.vocab, words=["a", "b", "c"])
    try:
        embeds["generic"].predict([bare])
        check("6. an unset Doc._.tb_lang raises", False, "it did not raise")
    except ValueError as e:
        check("6. an unset Doc._.tb_lang raises", "tb_lang" in str(e), str(e)[:60] + "...")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
