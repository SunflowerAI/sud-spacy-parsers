#!/usr/bin/env python
"""Derive a TinyBERT-tok2vec config from configs/config.cfg (the en base arm).

Only the encoder and the optimiser change: tagger and parser keep their ordinary
`spacy.Tok2VecListener.v1`, so the comparison against the shipped base arm is like for like.
Transformer fine-tuning wants a warmed-up, small learning rate and smaller padded batches.

    make_tinybert_config.py OUT.cfg [--no-mix] [--lr 1e-4] [--steps 20000]
"""
import argparse

from thinc.api import Config

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", default="configs/config.cfg")
ap.add_argument("--model-path", default="models/tinybert_general_4l_312d")
ap.add_argument("--no-mix", action="store_true", help="last layer only (the control)")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--warmup", type=int, default=250)
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--freeze", action="store_true", help="keep every BERT weight fixed")
ap.add_argument("--hash", action="store_true", help="concatenate the base arm's hash-embed CNN (sees case)")
args = ap.parse_args()

# interpolate=False: the default resolves ${paths.train} to null and breaks CLI path overrides (E913)
cfg = Config().from_disk(args.base, interpolate=False)
width = args.width or 312

cfg["components"]["tok2vec"]["model"] = {
    "@architectures": "sud.TinyBertTok2Vec.v1",
    "model_path": args.model_path,
    "width": args.width,
    "mix": not args.no_mix,
    "dropout": 0.1,
    "max_len": 256,
    "freeze": args.freeze,
}
if args.hash:
    old = cfg["components"]["tok2vec"]["model"]  # the TinyBERT block just written
    base = Config().from_disk(args.base, interpolate=False)["components"]["tok2vec"]["model"]
    embed, encode = dict(base["embed"]), dict(base["encode"])
    embed["width"] = encode["width"] = 96  # literal: the base config interpolates it, we read raw
    old["@architectures"] = "sud.TinyBertPlusHash.v1"
    old["embed"], old["encode"] = embed, encode
    width += 96
for comp in ("tagger", "parser"):
    cfg["components"][comp]["model"]["tok2vec"]["width"] = width

cfg["training"]["max_steps"] = args.steps
cfg["training"]["patience"] = 5000
cfg["training"]["optimizer"] = {
    "@optimizers": "Adam.v1",
    "beta1": 0.9,
    "beta2": 0.999,
    "L2_is_weight_decay": True,
    "L2": 0.01,
    "grad_clip": 1.0,
    "use_averages": False,
    "eps": 1e-8,
    "learn_rate": {
        "@schedules": "warmup_linear.v1",
        "warmup_steps": args.warmup,
        "total_steps": args.steps,
        "initial_rate": args.lr,
    },
}
cfg["training"]["batcher"] = {
    "@batchers": "spacy.batch_by_padded.v1",
    "discard_oversize": True,
    "get_length": None,
    "size": 2000,
    "buffer": 256,
}
cfg.to_disk(args.out)
print(f"{args.out}: mix={not args.no_mix} lr={args.lr} steps={args.steps}")
