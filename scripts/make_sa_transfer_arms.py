#!/usr/bin/env python3
"""Generate the three parser arms that test transferring the joint morph+lemma encoder.

The question: the parser has real syntax on 20 164 Vedic sentences, while the morphologiser and
lemmatiser now train on 244 k (Vedic + DCS). DCS carries NO dependency annotation, so the parser can
never see that data directly — but it can inherit a REPRESENTATION learned from it. Sanskrit is
morphologically rich with free word order, so case marking is the primary cue for grammatical
relations, and an encoder good at predicting Case/Number/Gender should carry most of what a parser
needs. Precedent: yue's encoder initialised from the Mandarin `zh_both_tok2vec.bin` gained
+1.15 baseline LAS, and that was CROSS-lingual.

Three arms, all at the joint encoder's width/depth so nothing is confounded with capacity:

    w64          from scratch                  — the capacity control
    w64_init     init_tok2vec = joint encoder  — transfer as a starting point (the yue recipe)
    w64_frozen   joint encoder sourced+FROZEN  — transfer as a fixed representation

`w64_frozen` is the strict form of the idea and would shrink the wheel most (one encoder shared by
everything); `w64_init` is likelier to win, since a 64-wide encoder trained for a different
objective is a plausible bottleneck for parsing. They answer different questions — whether
morph/lemma features are SUFFICIENT for parsing, versus whether they are a better STARTING POINT.

Trained on Vedic only: it is the only Sanskrit data here with gold syntax.

    make_sa_transfer_arms.py --joint training_sa_mwt_joint/model-best
"""
import argparse
import copy
import pathlib

from thinc.api import Config

W, DEPTH = 64, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="training_sa_mwt_joint/model-best")
    ap.add_argument("--base", default="configs/config_sa_mwt.cfg")
    ap.add_argument("--bin", default="sa_joint_tok2vec.bin")
    a = ap.parse_args()

    joint = Config().from_disk(pathlib.Path(a.joint) / "config.cfg", interpolate=False)
    aux = copy.deepcopy(joint["components"]["aux_tok2vec"]["model"])
    base = Config().from_disk(a.base, interpolate=False)

    # --- common: the base arm rebuilt at the joint encoder's exact architecture ---------------
    common = copy.deepcopy(base)
    common["components"]["tok2vec"]["model"] = copy.deepcopy(aux)
    for comp in ("tagger", "parser"):
        common["components"][comp]["model"]["tok2vec"] = {
            "@architectures": "spacy.Tok2VecListener.v1", "width": W, "upstream": "*"}

    # --- w64: from scratch (control) ----------------------------------------------------------
    common.to_disk("configs/config_sa_mwt_w64.cfg")

    # --- w64_init: same, but the encoder starts from the joint one ----------------------------
    init = copy.deepcopy(common)
    init.setdefault("paths", {})["init_tok2vec"] = a.bin
    init["initialize"]["init_tok2vec"] = "${paths.init_tok2vec}"
    # spaCy needs a [pretraining] block naming the component whose weights the bin fills
    # the WHOLE [pretraining] block, copied from a config known to work — spaCy also requires
    # optimizer / batcher / objective here, and omitting them fails at config validation, not at
    # run time, so a partial block looks fine until the arm is launched.
    yue = Config().from_disk("configs/config_yue.cfg", interpolate=False)
    init["pretraining"] = dict(yue["pretraining"])
    init["pretraining"]["component"] = "tok2vec"
    init["pretraining"]["layer"] = ""
    init.to_disk("configs/config_sa_mwt_w64_init.cfg")

    # --- w64_frozen: the joint encoder sourced and frozen; only tagger+parser learn ------------
    frz = copy.deepcopy(common)
    frz["components"]["tok2vec"] = {"source": a.joint, "component": "aux_tok2vec"}
    frz["training"]["frozen_components"] = ["tok2vec"]
    # A frozen tok2vec that LISTENERS depend on must also be annotating, else spaCy never runs it
    # and the listeners get nothing (E203).
    frz["training"]["annotating_components"] = ["tok2vec"]
    frz.to_disk("configs/config_sa_mwt_w64_frozen.cfg")

    print(f"  joint encoder: {aux['embed']['@architectures']} width {aux['embed']['width']}, "
          f"encode depth {aux['encode']['depth']}")
    for n in ("w64", "w64_init", "w64_frozen"):
        print(f"  wrote configs/config_sa_mwt_{n}.cfg")


if __name__ == "__main__":
    main()
