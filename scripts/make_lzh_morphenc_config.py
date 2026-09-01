#!/usr/bin/env python3
"""Write the lzh PARSER configs that read the MORPHOLOGISER'S TRAINED ENCODER as a side channel.

The parser's `tok2vec` becomes `concatenate(Tok2VecListener(96), FrozenPipeTok2Vec(64))` = 160,
where the second is the UPOS-supervised encoder lifted out of a trained morphologiser arm and
frozen. The shared encoder underneath stays frozen and byte-identical, exactly as in the
`parservec` arm, so the two are directly comparable and the only thing that changes is WHAT the
side channel carries: a raw SikuBERT row there, an already-extracted category representation here.

⚠ **THE CONTROL IS THE SAME ARCHITECTURE WITH A DONOR TRAINED ON THE SHUFFLED TABLE.** That holds
the parameter count, the depth, the width, the UPOS supervision and the whole training recipe
fixed, and varies ONLY how good the donor is at category on rare and unseen forms (73.98 % vs
66.92 % UPOS on treebank-unseen forms). A "no side channel" baseline would confound the transfer
with the extra 499 456 parameters and is not used.

⚠ **EACH DONOR IS PAIRED WITH THE VECTOR TABLE IT WAS TRAINED AGAINST.** `spacy.StaticVectors.v2`
reads `doc.vocab.vectors` at forward time, so a donor fitted on `vectors_lzh_siku96` running in a
host that loaded the shuffled table would be silently out of distribution.

⚠ **THE DONOR IS FIXED AT ONE SEED AND THE PARSER SEED IS WHAT VARIES.** That measures parser-seed
spread, not donor-seed spread, and is the right first cut — but it means a donor that happened to be
lucky is not averaged away. Say so when reporting.

Usage:
    make_lzh_morphenc_config.py --variant vectors --seed 0 --out configs/config_lzh_morphenc_s0.cfg
    make_lzh_morphenc_config.py --variant control --seed 0 --out configs/config_lzh_morphenc_ctl_s0.cfg
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh_seg.cfg")
    ap.add_argument("--source", default="training_lzh_seg/model-best")
    ap.add_argument("--variant", choices=("vectors", "control"), default="vectors")
    ap.add_argument("--donor", default="training_lzh_sikuvec_s0/model-best")
    ap.add_argument("--control-donor", default="training_lzh_sikuvec_ctl_s0/model-best")
    ap.add_argument("--vectors", default="vectors_lzh_siku96")
    ap.add_argument("--control-vectors", default="vectors_lzh_siku96_shuf")
    ap.add_argument("--component", default="morphologizer")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Config().from_disk(a.base, interpolate=False)   # interpolate=False or E913
    cfg["system"]["seed"] = a.seed
    for comp in ("tok2vec", "tagger"):
        cfg["components"][comp] = {"source": a.source}
    cfg["training"]["frozen_components"] = ["tok2vec", "tagger"]
    cfg["training"]["annotating_components"] = ["tok2vec"]

    parser = cfg["components"]["parser"]
    inner = parser["model"]["tok2vec"]
    if inner.get("@architectures") != "spacy.Tok2VecListener.v1":
        raise SystemExit(f"{a.base}: the parser reads {inner.get('@architectures')}, not a listener")
    inner["upstream"] = "tok2vec"
    inner["width"] = a.width
    donor = a.donor if a.variant == "vectors" else a.control_donor
    parser["model"]["tok2vec"] = {
        "@architectures": "sud.Tok2VecPlusFeats.v1",
        "tok2vec": inner,
        "feats_embed": {"@architectures": "sud.FrozenPipeTok2Vec.v1",
                        "path": donor, "component": a.component},
    }

    table = a.vectors if a.variant == "vectors" else a.control_vectors
    cfg["paths"]["vectors"] = table
    cfg["initialize"]["vectors"] = table

    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.variant}: donor={donor}, table={table}, seed={a.seed})")


if __name__ == "__main__":
    main()
