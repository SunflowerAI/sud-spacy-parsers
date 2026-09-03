#!/usr/bin/env python3
"""Write the lzh PARSER configs that read PCA'd SikuBERT vectors ABOVE the frozen encoder.

THE DESIGN, and every part of it is there to keep the comparison single-variable.

  * `tok2vec` and `tagger` are SOURCED FROM THE BASE ARM AND FROZEN, and `tok2vec` is listed in
    `annotating_components` so the parser's listener is actually fed during training (a listener on
    a frozen encoder that is not annotating receives nothing — the wiring `config_lzh_xposwarm.cfg`
    already uses).
  * The parser is NOT sourced. Its input width changes from 96 to 96 + `--dim`, so its weights
    cannot be carried over; it is retrained from scratch on the frozen encoder. That costs LAS
    against the released co-trained parser and it does not matter, because the CONTROL carries the
    identical handicap.
  * The control is the SHUFFLED table, never "no vectors": `StaticVectors` adds a projection and
    widens the parser's lower layer, so an arm-versus-baseline comparison would confound the
    information with the parameters.

⚠ THIS ARM CANNOT BE READ OFF ITS HEADLINE. A static vector informs only a decision the FORM does
not already settle; unseen forms are 1.15 % of Kyoto test tokens and forms seen twice or fewer
2.19 %, so even +15 LAS on the unseen slice is +0.17 aggregate against a ~0.5 seed spread. Score it
with `scripts/eval_lex_slices.py`. The headline is expected to be flat and that is not the result.

Usage:
    make_lzh_parservec_config.py --variant vectors --seed 0 --out configs/config_lzh_parservec_s0.cfg
    make_lzh_parservec_config.py --variant control --seed 0 --out configs/config_lzh_parservec_ctl_s0.cfg
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh_seg.cfg")
    ap.add_argument("--source", default="training_lzh_seg/model-best")
    ap.add_argument("--variant", choices=("vectors", "control"), default="vectors")
    ap.add_argument("--vectors", default="vectors_lzh_siku96")
    ap.add_argument("--control-vectors", default="vectors_lzh_siku96_shuf")
    ap.add_argument("--dim", type=int, default=96, help="width the vector is projected to")
    ap.add_argument("--width", type=int, default=96, help="the shared encoder's width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # interpolate=False or ${paths.train} resolves to null and the CLI overrides break (E913).
    cfg = Config().from_disk(a.base, interpolate=False)
    cfg["system"]["seed"] = a.seed

    # source + freeze everything but the parser; tok2vec must ANNOTATE or the listener is starved
    for comp in ("tok2vec", "tagger"):
        cfg["components"][comp] = {"source": a.source}
    cfg["training"]["frozen_components"] = ["tok2vec", "tagger"]
    cfg["training"]["annotating_components"] = ["tok2vec"]

    parser = cfg["components"]["parser"]
    inner = parser["model"]["tok2vec"]
    if inner.get("@architectures") != "spacy.Tok2VecListener.v1":
        raise SystemExit(f"{a.base}: the parser reads {inner.get('@architectures')}, not a listener")
    # the listener must name the component explicitly: with "*" and a frozen upstream the wiring is
    # resolved by position and a sourced tok2vec is easy to miss.
    inner["upstream"] = "tok2vec"
    inner["width"] = a.width
    parser["model"]["tok2vec"] = {
        "@architectures": "sud.Tok2VecPlusFeats.v1",
        "tok2vec": inner,
        "feats_embed": {"@architectures": "sud.StaticVecChannel.v1", "width": a.dim},
    }

    path = a.vectors if a.variant == "vectors" else a.control_vectors
    cfg["paths"]["vectors"] = path
    cfg["initialize"]["vectors"] = path

    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.variant}: vectors={path}, listener {a.width} + channel {a.dim} "
          f"= {a.width + a.dim}, seed={a.seed})")


if __name__ == "__main__":
    main()
