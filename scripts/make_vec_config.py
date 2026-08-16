#!/usr/bin/env python3
"""Write the lzh static-vector configs: the kanripo arm and its SHUFFLED control.

THE QUESTION, and why it differs from the two channels that failed before it. The XPOS channel
carried nothing the parser did not already have (XPOS is linearly decodable from its own input, at
the tagger's own 92.59 %), and the per-form lexicon carried exactly 0.0000 bits by an identity. The
kanripo vectors are the first channel with information the parser cannot otherwise reach: 42 M
tokens of Classical Chinese against the treebank's 460 k, with a row for every type at every
frequency, unseen forms included.

⚠ MEASURED OUTCOME (2026-08-16): +0.04 LAS mean over three seeds (+0.46 / -0.13 / -0.20, sd 0.29)
against the shuffled control. Seed 0 alone said +0.46 and it was noise. The vectors ARE informative
in the aggregate (UPOS 63.30 % from a held-out character's vector, against a 44.37 % majority) but
decay to the graphic backoff exactly where the parser needs them: at kanripo frequency 1-5 the probe
gives 57.79 %, the RADICAL's 57.00 to within noise, and treebank-unseen forms have a median kanripo
frequency of 4. Informative where the parser copes, near-empty where it does not.

THE CONTROL IS A SHUFFLE, NOT AN ABSENCE. `include_static_vectors = true` adds a `StaticVectors`
projection and widens the Maxout, so an arm-versus-baseline comparison confounds the information
with the parameters. `vectors_lzh_shuffled` holds the SAME 17 166 rows with the type-to-row
correspondence destroyed: identical shapes, identical norm distribution, identical parameter count,
zero information. Anything the shuffle also achieves was never the vectors.

⚠ THE VECTORS MUST BE THE LEAK-FREE SET. kanripo IS the source of the Kyoto treebank, so the stock
corpus contains every test sentence verbatim and 160 of 279 treebank-unseen types have their ONLY
occurrence inside held-out text. `make_leakfree_lzh_corpus.py` removes dev/test before training, and
`build_lzh_vectors.py --extra-types` keeps a row for every treebank type anyway -- prune by
DIMENSION, never by VOCABULARY, which is the mistake that left `vectors_lzh_apt96` with 0 % coverage
of unseen forms and no punctuation at all.

Usage:

    .venv/bin/python scripts/make_vec_config.py --variant vectors --out configs/config_lzh_vec.cfg
    .venv/bin/python scripts/make_vec_config.py --variant control --out configs/config_lzh_vec_ctl.cfg
"""
import argparse

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=("vectors", "control"), default="vectors")
    ap.add_argument("--vectors", default="vectors_lzh_leakfree")
    ap.add_argument("--control-vectors", default="vectors_lzh_shuffled")
    a = ap.parse_args()

    # interpolate=False or ${paths.train} resolves to null and the CLI overrides break (E913).
    cfg = Config().from_disk(a.base, interpolate=False)

    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    if "include_static_vectors" not in embed:
        raise SystemExit(f"{a.base} embed has no include_static_vectors field: {sorted(embed)}")
    embed["include_static_vectors"] = True

    path = a.vectors if a.variant == "vectors" else a.control_vectors
    cfg["initialize"]["vectors"] = path
    # `paths.vectors` is what the interpolated `${paths.vectors}` in [initialize] resolves to, so
    # setting only one of the two leaves the other pointing at null and the vectors never load.
    cfg["paths"]["vectors"] = path

    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.variant}: vectors={path}, include_static_vectors=True)")


if __name__ == "__main__":
    main()
