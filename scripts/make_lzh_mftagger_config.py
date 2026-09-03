#!/usr/bin/env python3
"""Write the lzh `multifield_tagger` config: per-XPOS-field heads over listener + UPOS + SikuBERT.

The tagger's input is `sud.Tok2VecPlusFeats.v1(listener, side)` exactly as the released one is,
but `side` now carries BOTH channels rather than one:

    sud.Tok2VecPlusFeats.v1
      tok2vec     spacy.Tok2VecListener.v1   on the frozen shared encoder  (the arm's own context)
      feats_embed sud.Tok2VecPlusFeats.v1                                   (nested, to get three)
        tok2vec     sud.MultiHashEmbedFeats.v1   POS + FEATS   — the UPOS the user may have edited
        feats_embed sud.StaticVecChannel.v1      PCA'd SikuBERT — lexical knowledge the treebank lacks

⚠ Both side channels enter ABOVE the encoder, not in the embed. That is the one thing
NEGATIVE-RESULTS.md found decisive for this tagger: identical information in the embed cost
-0.3 to -0.6 because a depth-4 `MaxoutWindowEncoder` then convolves a noisy predicted feature over
a ±4-token window, while above the encoder it helps. Representation was worth <= 0.10; position
was worth ~0.7.

⚠ The base is SOURCED AND FROZEN, so this is the freeze recipe and the parser cannot move. Point
`--source` at whichever width arm won.
"""
import argparse
import pathlib

from thinc.api import Config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh_seg_xposwarm.cfg")
    ap.add_argument("--source", default="training_lzh_seg_morph/model-best")
    ap.add_argument("--vectors", default="vectors_lzh_siku96")
    ap.add_argument("--tables", default="models/lzh_xpos_tables.json")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--vec-dim", type=int, default=96)
    ap.add_argument("--no-vectors", action="store_true", help="UPOS/FEATS channel only")
    ap.add_argument("--field-weight", type=float, default=0.0,
                    help="0.0 = rank by the joint head alone (+ the UPOS mask); >0 blends the fields")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Config().from_disk(a.base, interpolate=False)      # interpolate=False or E913
    cfg["system"]["seed"] = a.seed

    pipeline = list(cfg["nlp"]["pipeline"])
    if "tagger" in pipeline:
        pipeline[pipeline.index("tagger")] = "multifield_tagger"
        cfg["components"].pop("tagger", None)
    cfg["nlp"]["pipeline"] = pipeline
    for name in list(cfg["components"]):
        comp = cfg["components"][name]
        if isinstance(comp, dict) and "source" in comp:
            comp["source"] = a.source
    # ⚠ DROP THE WARM-START HOOK. `config_lzh_seg_xposwarm.cfg` installs an `after_init` callback
    # that copies the RELEASED tagger's weights into the new one, which is right when the label set
    # is preserved and impossible here: this component's parameters are 146 per-FIELD values, not
    # 121 joint tags, so there is no position-for-position correspondence to warm-start from.
    # `warm_start_tagger.py` would fail loudly on the missing `tagger` anyway — which is the
    # behaviour CLAUDE.md standing hazard 7 asks for, and the reason it is cleared rather than
    # worked around.
    for key in ("after_init", "before_init"):
        if key in cfg["initialize"]:
            cfg["initialize"][key] = None
    frozen = [c for c in pipeline if c != "multifield_tagger"]
    cfg["training"]["frozen_components"] = frozen
    # the listener needs its upstream to ANNOTATE, and the morphologiser must run so `POS` is set
    cfg["training"]["annotating_components"] = [c for c in ("tok2vec", "morphologizer")
                                                if c in pipeline]

    feats = {"@architectures": "sud.MultiHashEmbedFeats.v1", "width": 32, "attrs": ["POS"],
             "rows": [100], "feats": ["Case", "NameType", "Degree", "VerbForm", "PronType",
                                      "Person"],
             "feat_rows": [16, 32, 32, 16, 16, 16], "include_static_vectors": False}
    side = feats if a.no_vectors else {
        "@architectures": "sud.Tok2VecPlusFeats.v1",
        "tok2vec": feats,
        "feats_embed": {"@architectures": "sud.StaticVecChannel.v1", "width": a.vec_dim}}

    import json as _json
    tabs = _json.loads(pathlib.Path(a.tables).read_text(encoding="utf-8"))
    sizes = [len({t.split(",")[i] for t in tabs["attested"]}) for i in range(4)]
    print(f"  field sizes from {a.tables}: {sizes}")
    cfg["components"]["multifield_tagger"] = {
        "factory": "multifield_tagger", "sep": ",", "n_fields": 4,
        "tables": a.tables, "project": True, "upos_mask": False, "overwrite": True,
        "joint": True, "field_weight": a.field_weight,
        "model": {"@architectures": "sud.MultiFieldTagger.v2", "field_sizes": sizes,
                  "n_joint": len(tabs["attested"]),
                  "tok2vec": {"@architectures": "sud.Tok2VecPlusFeats.v1",
                              "tok2vec": {"@architectures": "spacy.Tok2VecListener.v1",
                                          "width": a.width, "upstream": "tok2vec"},
                              "feats_embed": side}}}

    table = None if a.no_vectors else a.vectors
    cfg["paths"]["vectors"] = table
    cfg["initialize"]["vectors"] = table
    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  (source={a.source}, vectors={table}, tables={a.tables}, seed={a.seed})")


if __name__ == "__main__":
    main()
