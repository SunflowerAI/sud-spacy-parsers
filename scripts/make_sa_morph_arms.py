#!/usr/bin/env python3
"""Generate the sa morphologiser affix-ablation configs from `configs/config_sa_morph.cfg`.

Every arm differs from the baseline in the `[components.morphologizer.model.tok2vec]` subtree ONLY,
and the generator asserts that — a two-variable arm is unreadable, and this ablation exists to
settle a question the previous (confounded) PREFIX/SUFFIX experiment could not answer.

The affix arms swap `spacy.MultiHashEmbed.v2` for `sud.MultiHashEmbedAffix.v1`, which is exactly
equivalent when no affix is configured (verified by `scripts/check_affix_embed.py`), and add one
hash-embedded table per affix length computed from `token.text[-k:]` at forward time. The
lexeme-level `SUFFIX` stays at spaCy's default 3 throughout — see scripts/sud_affix_embed.py for
why that matters.

`w96` is the CAPACITY CONTROL and is the most important arm after the baseline: it adds parameters
without adding the feature. If it moves `morph_acc` as much as an affix arm, the affix feature is
doing nothing and the gain is just capacity.

Loads/saves with interpolation OFF so `${paths.train}` survives (CLAUDE.md gotcha).

    .venv/bin/python scripts/make_sa_morph_arms.py [--out-dir configs] [--list]
"""
import argparse
import copy
import pathlib

from thinc.api import Config

BASE = "configs/config_sa_morph.cfg"
EMBED = ("components", "morphologizer", "model", "tok2vec", "embed")
ENCODE = ("components", "morphologizer", "model", "tok2vec", "encode")
AFFIX_ARCH = "sud.MultiHashEmbedAffix.v1"

# name -> (suffixes, suffix_rows, prefixes, prefix_rows, width or None)
# Row counts are sized against the distinct-value counts measured on sa train
# (k=4 7 188, k=5 15 594, k=6 23 434), so the tables run at ~1x load; `sfx5_r8000` halves that to
# test row sensitivity. Table cost in the wheel is rows * width * 4 bytes.
ARMS = {
    "sfx4_r8000":  ([4], [8000],  [], [],   None),
    "sfx5_r16000": ([5], [16000], [], [],   None),
    "sfx6_r24000": ([6], [24000], [], [],   None),
    "sfx5_r8000":  ([5], [8000],  [], [],   None),
    "pfx2_r1000":  ([],  [],      [2], [1000], None),
    "pfx3_r2500":  ([],  [],      [3], [2500], None),
    "w96":         ([],  [],      [], [],   96),      # capacity control: no feature, more params
}


def get(cfg, path):
    node = cfg
    for k in path:
        node = node[k]
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--list", action="store_true", help="just print the arm names")
    args = ap.parse_args()

    if args.list:
        print("base " + " ".join(ARMS))
        return

    base = Config().from_disk(args.base, interpolate=False)
    base_embed = get(base, EMBED)
    out_dir = pathlib.Path(args.out_dir)

    for name, (sfx, sfx_rows, pfx, pfx_rows, width) in ARMS.items():
        cfg = copy.deepcopy(base)
        embed = get(cfg, EMBED)
        if width is not None:
            # capacity control: widen the dedicated encoder, keep the stock embed architecture
            embed["width"] = width
            get(cfg, ENCODE)["width"] = width
        else:
            embed["@architectures"] = AFFIX_ARCH
            embed["suffixes"] = sfx
            embed["suffix_rows"] = sfx_rows
            embed["prefixes"] = pfx
            embed["prefix_rows"] = pfx_rows

        # single-variable assertion: nothing outside the morphologiser's tok2vec may move
        a, b = copy.deepcopy(cfg), copy.deepcopy(base)
        get(a, EMBED).clear(); get(b, EMBED).clear()
        get(a, ENCODE).clear(); get(b, ENCODE).clear()
        assert a == b, f"{name}: arm differs from the baseline outside morphologizer.model.tok2vec"
        assert get(cfg, EMBED)["attrs"] == base_embed["attrs"], f"{name}: attrs must not change"
        assert get(cfg, EMBED)["rows"] == base_embed["rows"], f"{name}: base rows must not change"

        out = out_dir / f"config_sa_morph_{name}.cfg"
        cfg.to_disk(out)
        cost = sum(r for r in sfx_rows + pfx_rows) * get(cfg, EMBED)["width"] * 4
        note = f"+{cost / 1e6:.1f} MB table" if cost else "capacity control"
        print(f"  wrote {out}  ({note})")


if __name__ == "__main__":
    main()
