#!/usr/bin/env python3
"""Derive a `[pretraining]`-capable copy of a base config so `init_tok2vec` can load a donor blob.

spaCy resolves the load target through `get_tok2vec_ref(nlp, config["pretraining"])`, so a config
whose `[pretraining]` is EMPTY -- which is what every base config here ships -- cannot accept a
donor. This copies the block (and the `corpora.pretrain` it names) from a config already proven to
work in this repo, `config_yue.cfg`, changing nothing else.

`init_tok2vec` is deliberately LEFT as `${paths.init_tok2vec}`: one config then serves BOTH arms of
the experiment, the graft passing `--paths.init_tok2vec <blob>` and the control passing nothing. A
control that shares the code path is the only kind worth reading (cf. the constant-channel control
that came out bit-identical from two different configs).

    .venv/bin/python scripts/make_graft_config.py configs/config_zh_seg.cfg \
        --out configs/config_zh_graft.cfg
"""
import argparse
from thinc.api import Config

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--donor-config", default="configs/config_yue.cfg")
ap.add_argument("--out", required=True)
a = ap.parse_args()

# interpolate=False is mandatory: the default resolves ${paths.train} to null and silently breaks
# the CLI path overrides (this is what caused E913 elsewhere in the repo).
base = Config().from_disk(a.base, interpolate=False)
donor = Config().from_disk(a.donor_config, interpolate=False)

if base.get("pretraining"):
    raise SystemExit(f"{a.base} already has a non-empty [pretraining]; refusing to clobber it")

base["pretraining"] = donor["pretraining"]
base["corpora"]["pretrain"] = donor["corpora"]["pretrain"]
base["paths"].setdefault("raw_text", None)

assert base["initialize"]["init_tok2vec"] == "${paths.init_tok2vec}", \
    "base config does not route init_tok2vec through [paths]; the CLI override would not reach it"
assert base["paths"]["init_tok2vec"] is None, "base config pins init_tok2vec; control arm would not be clean"

base.to_disk(a.out)
print(f"wrote {a.out}")
print(f"  pretraining.component={base['pretraining']['component']!r} layer={base['pretraining']['layer']!r}")
print(f"  init_tok2vec stays {base['initialize']['init_tok2vec']} (CLI-controlled)")
