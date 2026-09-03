#!/usr/bin/env python3
"""Write the lzh morphologiser configs that read PCA'd SikuBERT vectors — and the matched CONTROL.

THE QUESTION. The morphologiser's own encoder is `spacy.HashEmbedCNN.v2`, width 64, depth 3,
**2 000 hash rows for 9 029 training types**, over NORM/PREFIX/SUFFIX/SHAPE and no vectors at all.
For a single Han character those four attrs are the same character, so the channel is NORM alone,
and an unfamiliar glyph lands on a colliding row with a ~50 % PROPN prior waiting for it. Setting
`pretrained_vectors = true` makes `MultiHashEmbed` CONCATENATE a `StaticVectors` projection with
those hash channels before the Maxout — literally "tok2vec ⊕ PCA'd SikuBERT".

⚠ THE CONTROL IS A SHUFFLE, NOT AN ABSENCE. `include_static_vectors` adds a projection and widens
the Maxout, so arm-versus-baseline confounds the information with the parameters.
`build_lzh_sikubert_vectors.py --shuffle` writes the SAME rows with the type-to-row correspondence
destroyed: identical shapes, identical norms, identical parameter count, zero information. Anything
the shuffle also achieves was never the vectors. This is the mistake the kanripo-vector arm did NOT
make and is the only reason its +0.04 mean is readable.

⚠ AND READ NO SINGLE SEED. That entry's headline row was +0.46 on seed 0 and +0.04 over three.
`--seed` writes one config per seed; run all three before believing anything.

Usage:
    make_lzh_sikuvec_config.py --variant vectors --seed 0 --out configs/config_lzh_sikuvec_s0.cfg
    make_lzh_sikuvec_config.py --variant control --seed 0 --out configs/config_lzh_sikuvec_ctl_s0.cfg
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh_seg_morph.cfg")
    ap.add_argument("--variant", choices=("vectors", "control", "baseline"), default="vectors")
    ap.add_argument("--vectors", default="vectors_lzh_siku96")
    ap.add_argument("--control-vectors", default="vectors_lzh_siku96_shuf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # interpolate=False or ${paths.train} resolves to null and the CLI overrides break (E913).
    cfg = Config().from_disk(a.base, interpolate=False)
    cfg["system"]["seed"] = a.seed

    enc = cfg["components"]["morphologizer"]["model"]["tok2vec"]
    if "pretrained_vectors" not in enc:
        raise SystemExit(f"{a.base}: the morphologizer encoder is {enc.get('@architectures')}, "
                         f"which has no pretrained_vectors field")
    if a.variant == "baseline":
        enc["pretrained_vectors"] = None
        path = None
    else:
        enc["pretrained_vectors"] = True
        path = a.vectors if a.variant == "vectors" else a.control_vectors
    # `paths.vectors` is what the interpolated `${paths.vectors}` in [initialize] resolves to, so
    # setting only one of the two leaves the other pointing at null and the vectors never load.
    cfg["paths"]["vectors"] = path
    cfg["initialize"]["vectors"] = path

    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.variant}: vectors={path}, seed={a.seed})")


if __name__ == "__main__":
    main()
