#!/usr/bin/env python3
"""Continuation config for la's beam-search parser retrain (configs/config_la_beam.cfg).

WHY. `spacy train` has no `--resume`: hitting `max_steps` ends the run regardless of whether SCORE
is still improving. Watching `train_la_beam.log` live, dev LAS climbed from ~72-73 in the low teens
of thousands of steps to a sustained ~74.0-74.15 by step 16600-17000, out of a 20000-step budget --
if it reaches max_steps WITHOUT exhausting patience first (i.e. still finding new bests, not
plateaued), that is a run stopped by an arbitrary step cap, not a converged one, and worth
continuing rather than accepting as final.

WHAT IT DOES. Sources `tok2vec`/`tagger`/`parser` from the FIRST run's own `model-last` (or
`model-best`, if that scores higher and you pass it explicitly) instead of random-initialising them,
while leaving `morphologizer`/`lemmatizer` exactly as the base config already has them (sourced from
`training_la_aug_lemma/model-best`, frozen, annotating -- untouched, since they were never being
retrained in the first place). This is the SAME "source a component instead of building it fresh"
mechanism `make_la_lemvec_config.py`/`make_sud_config.py` already use, just applied to components
that were TRAINED (not frozen) in the run being continued -- they stay TRAINABLE here too, since
`frozen_components` is a separate list this script does not add them to. A sourced component brings
its own full definition (weights AND the config it was built with), so beam_parser's own
factory/beam_width/beam_update_prob settings travel automatically; only the base config's
`training.max_steps` is overridden, giving the continuation its own fresh step/patience budget
rather than treating the first run's step count as if it still applied.

    make_la_beam_continue_config.py --out configs/config_la_beam_cont.cfg
    make_la_beam_continue_config.py --out configs/config_la_beam_cont.cfg --source training_la_beam/model-best
"""
from __future__ import annotations

import argparse

from thinc.api import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/config_la_beam.cfg")
    ap.add_argument("--source", default="training_la_beam/model-last",
                    help="the checkpoint to continue FROM -- model-last (wherever training was "
                         "when it stopped) by default, or model-best to continue from the single "
                         "highest-SCORE checkpoint seen instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="a FRESH budget for the continuation, not added to the first run's own "
                         "step count -- the first run's step count has no meaning to a newly "
                         "sourced component's own patience tracking")
    args = ap.parse_args()

    cfg = Config().from_disk(args.base, interpolate=False)
    for name in ("tok2vec", "tagger", "parser"):
        cfg["components"][name] = {"source": args.source}
        # A sourced component brings its own labels; an initialize block pointed at the ORIGINAL
        # labels file is now stale (harmless here, since the label SET is unchanged between the two
        # runs, but leaving a stale path around is how a future rename goes unnoticed -- the same
        # reasoning make_la_lemvec_config.py's own step 1 already documents).
        cfg["initialize"].get("components", {}).pop(name, None)
    cfg["training"]["max_steps"] = args.max_steps

    cfg.to_disk(args.out)
    print(f"wrote {args.out} (continuing tok2vec/tagger/parser from {args.source}, "
          f"fresh max_steps={args.max_steps}; morphologizer/lemmatizer untouched)")


if __name__ == "__main__":
    main()
