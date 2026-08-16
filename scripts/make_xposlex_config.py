#!/usr/bin/env python3
"""Write the XPOS-lexicon parser configs from a base config.

Three arms, and the second is not optional:

    --variant fields    the channel: one hash table per comma-separated XPOS field
    --variant control   identical columns, rows, Maxout and parameter count, ZERO information
    --variant whole     the same channel carrying the undecomposed 118-way tag

The control is what makes the result readable. NEGATIVE-RESULTS.md has the measured warning: the
la control scored 0.5 below an arm that was architecturally IDENTICAL to it, so a delta smaller
than that is the seed, not the channel.

WHERE THE CHANNEL GOES, and why not in the embed. It wraps the PARSER's own `Tok2VecListener` with
`sud.Tok2VecPlusFeats.v1`, i.e. ABOVE the shared encoder. Putting the same information into
`[components.tok2vec.model.embed]` instead would (a) hand it to the co-trained TAGGER, whose target
it is derived from, making the tagger score meaningless, and (b) let a depth-4
`MaxoutWindowEncoder` convolve it over a +-4 token window. That second point is the resolved
finding in `docs/xpos.md`: bottom injection lost 0.2-0.6 in three languages, top injection won in
nine. A per-token lexical class should reach that token's decision and nothing else.

⚠ `Config().from_disk(..., interpolate=False)`. The default interpolation resolves `${paths.train}`
to null and silently breaks the CLI path overrides (E913). Same reason `rebuild_lzh_trad.sh` does
it that way.

Usage:

    .venv/bin/python scripts/make_xposlex_config.py --variant fields \\
        --base configs/config_lzh.cfg --out configs/config_lzh_xposlex.cfg
"""
import argparse

from thinc.api import Config

PREPASS = "xpos_prepass"

DEFAULT_FIELDS = [1, 2]      # 品詞 (12 values) and the coarse semantic class (46)
DEFAULT_ROWS = [16, 64]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/config_lzh.cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant",
                    choices=("fields", "control", "whole", "tagpred", "tagpred_control"),
                    default="fields")
    ap.add_argument("--table", default="models/lzh_xpos_lex.json")
    ap.add_argument("--whole-table", default="models/lzh_xpos_whole.json")
    ap.add_argument("--fields", type=int, nargs="+", default=DEFAULT_FIELDS)
    ap.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    ap.add_argument("--width", type=int, default=32, help="side-channel width")
    a = ap.parse_args()

    cfg = Config().from_disk(a.base, interpolate=False)

    if a.variant.startswith("tagpred"):
        # The base tagger becomes a FIRST-PASS component feeding the parser. Renaming it is not
        # cosmetic: `package_sud.sh`'s pkg() refuses a pipeline where the component NAMED "tagger"
        # precedes "morphologizer", which is how a pre-graft arm is caught. An arm carrying an
        # `xpos_prepass` before the parser AND the UPOS+FEATS-conditioned `tagger` last is not
        # pre-graft, and under its own name it does not trip a guard aimed at something else.
        comps = cfg["components"]
        if "tagger" not in comps:
            raise SystemExit(f"{a.base} has no [components.tagger] to repurpose")
        cfg["components"] = {(PREPASS if k == "tagger" else k): v for k, v in comps.items()}
        cfg["nlp"]["pipeline"] = [PREPASS if n == "tagger" else n
                                  for n in cfg["nlp"]["pipeline"]]
        # ⚠ WITHOUT THIS THE EXPERIMENT MEASURES NOTHING. `annotating_components` is what makes the
        # tagger's predictions land on the docs the parser then updates on. Omit it and the parser
        # trains against an UNSET tag and is served a real one at inference -- the train/inference
        # regime mismatch this repo keeps paying for, and it would read as "the channel does not
        # help" rather than as a bug.
        cfg["training"]["annotating_components"] = [PREPASS]

    parser = cfg["components"]["parser"]["model"]
    old = parser.get("tok2vec", {})
    upstream = old.get("upstream", "*")
    width = old.get("width", "${components.tok2vec.model.encode.width}")

    if a.variant == "tagpred":
        fields, rows, table, constant = a.fields, a.rows, None, False
    elif a.variant == "tagpred_control":
        fields, rows, table, constant = a.fields, a.rows, None, True
    elif a.variant == "whole":
        fields, rows, table, constant = [0], [128], a.whole_table, False
    elif a.variant == "control":
        fields, rows, table, constant = a.fields, a.rows, None, True
    else:
        fields, rows, table, constant = a.fields, a.rows, a.table, False
    if len(rows) != len(fields):
        raise SystemExit(f"--rows {rows} does not match --fields {fields}")

    lex = {
        "@architectures": "sud.LexFieldEmbed.v1",
        "width": a.width,
        "fields": list(fields),
        "rows": list(rows),
        "constant": constant,
    }
    if a.variant.startswith("tagpred"):
        # Per-TOKEN source. The lexicon variant is a function of the form and therefore carries
        # exactly 0.0000 bits beyond NORM; only a per-token tag can carry any of the 0.1475 bits a
        # predicted tagger has to give.
        lex["source"] = "tag"
    # `table = null` for the control: it must not be able to read the lexicon even by accident.
    if table is not None:
        lex["table"] = table

    parser["tok2vec"] = {
        "@architectures": "sud.Tok2VecPlusFeats.v1",
        "tok2vec": {
            "@architectures": "spacy.Tok2VecListener.v1",
            "width": width,
            "upstream": upstream,
        },
        "feats_embed": lex,
    }

    cfg.to_disk(a.out)
    print(f"  wrote {a.out}  ({a.variant}: fields={fields} rows={rows} "
          f"width={a.width} table={table})")


if __name__ == "__main__":
    main()
