#!/usr/bin/env python3
"""Configs for the v3 arms: the released v2 arm, plus a lexical channel and its two controls.

    g3_base       v2's published `g2_base`, reproduced       the comparator
    g3_vec        + the aligned lemma-vector channel        THE HEADLINE
    g3_vec_ctl    + the same Linear, a zero vector          what the PARAMETERS buy
    g3_vec_shuf   + the vectors, permuted within a doc      whether the RIGHT vector matters
    g3_vec_aug    + training-time REAL shift displacement    whether the gloss route is recoverable
    g3_vec_aug2   the same, calibrated to the real geometry   how much of it the mis-calibration cost

`g3_vec_aug` exists because seed 0 said the channel is worth +4.51 macro LAS on held-out languages
that have aligned rows, and that the English-gloss substitution delivers -0.86 of it. The gap is an
input-regime mismatch, not a geometry failure: a gloss sits at cos +0.46 from its source lemma,
which is far above a shuffled control and far below the same distribution. So the arm trains on the
lemma vector DISPLACED by a real measured source->English shift, and meets at training time what it
will be handed at inference. No gloss enters training.

⚠ THE BASELINE IS `g2_base`, NOT THE RELEASED WHEEL, AND THE REASON IS THE WORD "ZERO-SHOT". The
released wheel carries a trainable per-language embedding, and that channel REFUSES an unseen
language until one of its spare rows is fitted on the target language's own gold trees. Building on
it would make every held-out number depend on 10-200 annotated sentences of the language being
scored -- a defensible deployment story, and not a zero-shot measurement. `g2_base` is v2's own
published baseline (UPOS + FEATS, no typology, no language embedding), it scores all twenty held-out
languages with no adaptation at all, and `metrics/generic_v2/metrics_g2_base_s*.json` already holds
three seeds of it at macro LAS 54.24. So v3's delta is read against a published number measured by
the same harness, which is the comparison this repo keeps having to redo.

⚠ AND IT IS BUILT BY CALLING v2's OWN `build()`, NOT BY COPYING IT. The fastest way to fake a gain
is a baseline that differs in some second thing nobody wrote down; this repo has had two such
confounds, one faking +11 LAS and one +12.5.

⚠ `g3_vec_ctl` AND `g3_base` ARE BOTH NEEDED AND ARE NOT THE SAME ARM. The first has no vector block
at all; the second has every parameter of the channel and none of its information. A delta is
quoted against `g3_vec_ctl`; `g3_base` says what the whole block costs in capacity. v1 drew exactly
this distinction for its own vector channel and NEGATIVE-RESULTS.md records why.

⚠ ONLY 32 OF THE 80 TRAINING LANGUAGES HAVE VECTOR ROWS -- about 42 % of training tokens once
per-language coverage is applied. The other 48 reach the model with the OOV dimension set on every
token, which is a state the channel can represent and the model can condition on. That is realistic
rather than unfortunate: a deployed arm meets both. But it does mean the channel is absent for most
of the data, so a null result here is NOT evidence that lexical information does not help -- it
would be evidence that it does not help AT THIS FILL RATE, which is a different claim and must be
written as one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from thinc.api import Config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_generic_config_v2 import build as build_v2, feats_inventory   # noqa: E402

#: The augmentation STRENGTH sweep. k scales distance-from-1 on the empirical cosine distribution,
#: so the shape survives and only the centre moves: k=1.00 reproduces the real lemma->gloss geometry
#: (cos 0.461), k=0.25 is a gentle nudge (0.865). k=0.55 lands on cos 0.704 -- the same STRENGTH as
#: g3_vec_aug's accidental additive perturbation but with realistic SPREAD (sd 0.101 against 0.072),
#: which is what separates the two variables those two arms confounded.
AUG_K = {"g3_aug_k25": 0.25, "g3_aug_k40": 0.40, "g3_aug_k55": 0.55, "g3_aug_k75": 0.75}

ARMS = (("g3_base", "g3_vec", "g3_vec_ctl", "g3_vec_shuf", "g3_vec_aug", "g3_vec_aug2")
        + tuple(AUG_K))


def build(arm, *, vectors, vectors_fill, shift_aug=None, shift_prob=0.0,
          shift_cos=None, **kw):
    # v2's OWN published baseline: UPOS + FEATS, no typology, NO language embedding.
    cfg = build_v2("g2_base", **kw)
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    if arm == "g3_base":
        return cfg                                  # untouched: this IS the released arm
    embed["@architectures"] = "sud.GenericEmbed.v3"
    # ⚠ NOT `paths.vectors`. That key is spaCy's own: `[initialize] vectors` interpolates it and
    # tries to load a spaCy Vectors object, so pointing it at this table fails with E884 -- and
    # would be wrong even if it loaded, because this is a lookup table the LAYER reads, not a
    # vectors table the vocab attaches. `paths.typology` is a custom key for the same reason.
    cfg["paths"]["vec_table"] = vectors
    embed.update({"vectors": "${paths.vec_table}", "vectors_fill": vectors_fill,
                  "vectors_constant": False, "vectors_shuffle": False})
    if arm == "g3_vec":
        pass
    elif arm == "g3_vec_ctl":
        embed["vectors_constant"] = True
    elif arm == "g3_vec_shuf":
        embed["vectors_shuffle"] = True
    elif arm == "g3_vec_aug":
        embed.update({"vectors_shift_aug": shift_aug, "vectors_shift_prob": shift_prob})
    elif arm in AUG_K:
        embed.update({"vectors_shift_aug": shift_aug, "vectors_shift_prob": shift_prob,
                      "vectors_shift_cos": shift_cos, "vectors_shift_strength": AUG_K[arm]})
    elif arm == "g3_vec_aug2":
        # Same idea, correctly calibrated. g3_vec_aug ADDS a sampled displacement, which
        # under-perturbs: cos 0.704 (sd 0.072) against a real gloss's 0.460 (sd 0.185), because a
        # shift is correlated with its own source and lands smaller against an unrelated vector.
        # This draws the TARGET COSINE from the empirical distribution and takes only the direction
        # from a real shift, so both moments match the deployment geometry by construction.
        embed.update({"vectors_shift_aug": shift_aug, "vectors_shift_prob": shift_prob,
                      "vectors_shift_cos": shift_cos})
    else:
        raise ValueError(f"unknown arm {arm}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-conllu", default="assets_generic_v2")
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--typology", default="assets_typ/typology_v2.json")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--vectors", default="assets_vec/generic_vec_v3.npz")
    ap.add_argument("--shift-aug", default="assets_vec/gloss_shift_sample_v3.npy")
    ap.add_argument("--shift-cos", default="assets_vec/gloss_cos_v3.npy")
    ap.add_argument("--shift-prob", type=float, default=0.5,
                    help="fraction of in-table tokens displaced per step, for g3_vec_aug")
    ap.add_argument("--vectors-fill", default="lemma", choices=("lemma", "gloss", "auto"),
                    help="TRAINING fill. `lemma` is the arm as designed; the gloss regime is a "
                         "property of INFERENCE and is set on the loaded model, not here.")
    ap.add_argument("--arm", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-langs", type=int, default=5)
    ap.add_argument("--min-token-pct", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--hidden-width", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--eval-frequency", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4000)
    ap.add_argument("--spare", type=int, default=32)
    ap.add_argument("--lang-dim", type=int, default=0)
    ap.add_argument("--outdir", default="configs")
    a = ap.parse_args()

    if not pathlib.Path(a.vectors).exists():
        sys.exit(f"no vector table at {a.vectors} -- build it with build_generic_vectors_v3.py")

    feats, feat_rows, _report, dropped = feats_inventory(a.corpus_conllu, a.min_langs, a.min_token_pct)
    man = json.load(open(a.manifest, encoding="utf-8"))["languages"]
    langs = sorted(l for l, v in man.items() if v["pool"] == "train")
    print(f"{len(feats)} FEATS categories ({len(dropped)} dropped), {len(langs)} training languages")

    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for arm in a.arm:
        cfg = build(arm, vectors=a.vectors, vectors_fill=a.vectors_fill,
                    shift_aug=a.shift_aug, shift_prob=a.shift_prob, shift_cos=a.shift_cos,
                    corpus=a.corpus, typology=a.typology, feats=feats, feat_rows=feat_rows,
                    langs=langs, seed=a.seed, width=a.width, depth=a.depth,
                    hidden_width=a.hidden_width, max_steps=a.max_steps,
                    eval_frequency=a.eval_frequency, patience=a.patience,
                    spare=a.spare, lang_dim=a.lang_dim)
        p = out / f"config_{arm}{a.suffix}.cfg"
        Config(cfg).to_disk(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
