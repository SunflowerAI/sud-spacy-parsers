#!/usr/bin/env python3
"""Widen the CONTEXT WINDOW of an lzh arm's encoder, holding everything else fixed.

`MaxoutWindowEncoder` sees `window_size` tokens each side per layer, and stacks `depth` of them, so
the receptive field is `+-window_size * depth` tokens. The shipped arms are all at window 1:

    arm             encoder                     width  depth  window   receptive field
    base tok2vec    MaxoutWindowEncoder.v2         96      4       1   +-4  tokens
    morphologiser   HashEmbedCNN.v2 (own)          64      3       1   +-3  tokens
    tagger          listener on the base           96      4       1   +-4  tokens

⚠ **THE BASE IS A BASE RECIPE.** `tok2vec` is co-trained with the parser and the tagger listens to
it, so widening it retrains all three and every layer above has to be rebuilt on the result
(CLAUDE.md: `seg` is a BASE recipe, not a stackable layer). The morphologiser has its OWN encoder
and can be widened independently.

⚠ **WIDENING IS NOT FREE AND THE COST IS NOT ONLY COMPUTE.** `MaxoutWindowEncoder` concatenates
`2*window_size+1` vectors before each Maxout, so parameters in the encoder grow ROUGHLY LINEARLY in
the window: window 2 is ~1.7x the encoder's weights, window 3 ~2.3x. A width sweep that does not
say which of the two moved the metric is uninterpretable, which is why `--depth` is exposed: a
DEPTH-matched control at the original window is the arm that separates "more context" from "more
parameters". Receptive field is `window * depth` either way.

Usage:
    make_lzh_width_config.py --base configs/config_lzh_seg.cfg --window 2 \
        --out configs/config_lzh_seg_w2.cfg
    make_lzh_width_config.py --base configs/config_lzh_seg_morph.cfg --window 2 \
        --component morphologizer --source training_lzh_seg_w2/model-best \
        --out configs/config_lzh_seg_w2_morph.cfg
"""
import argparse

from thinc.api import Config


def encoder_of(cfg, component):
    """The encode block whose window this call edits, for either encoder shape."""
    model = cfg["components"][component]["model"]
    if "encode" in model:                       # spacy.Tok2Vec.v2 -> .encode
        return model["encode"]
    t = model.get("tok2vec", {})
    if "encode" in t:
        return t["encode"]
    if "window_size" in t:                      # spacy.HashEmbedCNN.v2 is flat
        return t
    raise SystemExit(f"{component}: no window_size found under {sorted(model)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--component", default="tok2vec")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="the depth-matched CONTROL: same receptive field, window left alone")
    ap.add_argument("--source", default=None,
                    help="for a layer config, the new base arm its components are sourced from")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Config().from_disk(a.base, interpolate=False)   # interpolate=False or E913
    enc = encoder_of(cfg, a.component)
    old_w, old_d = enc.get("window_size"), enc.get("depth")
    if a.window is not None:
        enc["window_size"] = a.window
    if a.depth is not None:
        enc["depth"] = a.depth
    if a.seed is not None:
        cfg["system"]["seed"] = a.seed
    if a.source:
        for name, comp in cfg["components"].items():
            if isinstance(comp, dict) and "source" in comp:
                comp["source"] = a.source
    new_w, new_d = enc["window_size"], enc["depth"]
    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.component}: window {old_w}->{new_w}, depth {old_d}->{new_d}, "
          f"receptive field ±{old_w*old_d} -> ±{new_w*new_d} tokens"
          + (f", source={a.source}" if a.source else "") + ")")


if __name__ == "__main__":
    main()
