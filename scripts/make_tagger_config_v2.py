#!/usr/bin/env python3
"""Config for the generic TAGGING arm: UPOS, FEATS and lemmas from wordform + a language embedding.

One shared encoder feeding a `morphologizer`. UPOS is an INPUT here, not an output: it is the one
column that cannot be transferred (a multilingual tagger over 80 languages reaches ~35 % on held-out
languages, no better than a single English tagger), so the realistic configuration is one where the
annotator supplies it and the model predicts FEATS from it. That is worth +13.9 points of held-out
FEATS over the predict-nothing baseline, against +4.2 without UPOS.

⚠ **THERE IS NO LEMMATISER, AND THAT IS A MEASURED DECISION.** An edit-tree `trainable_lemmatizer`
trained across all 80 treebanks is INERT on a held-out language: across six languages and two
architectures its largest deviation from simply copying the wordform was +0.31 points. No tree in
the learned inventory maps an unseen language's wordform to its lemma, so it falls back to `orth`
every time, and UPOS does not help -- the problem is not choosing among candidates, it is that the
right candidate is absent. Shipping it would report 48-73 % "lemma accuracy" for a component that
returns its input.

⚠ NOT the freeze recipe. Everywhere else in this repo a layer above the base arm gets its OWN small
encoder and the base is frozen, because co-training was measured and dominated. Here the two heads
are trained together on ONE encoder deliberately: they read the same wordform, they are predicting
correlated things (a lemma is the form minus its inflection, which is what FEATS names), and the
language embedding has to be shared or each head would fit its own and the adaptation story would
need two fittings instead of one.
"""
import argparse
import json
import pathlib

from thinc.api import Config


def build(corpus, langs, seed, width, depth, max_steps, eval_frequency, patience, spare,
          lang_dim):
    return {
        "paths": {"corpus": corpus, "vectors": None, "init_tok2vec": None},
        "system": {"gpu_allocator": None, "seed": seed},
        "nlp": {"lang": "xx", "pipeline": ["tok2vec", "morphologizer"],
                "batch_size": 1000, "disabled": [], "before_creation": None,
                "after_creation": None, "after_pipeline_creation": None,
                "tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"},
                "vectors": {"@vectors": "spacy.Vectors.v1"}},
        "corpora": {
            "train": {"@readers": "sud.GenericTagCorpus.v1", "path": "${paths.corpus}",
                      "split": "train", "seed": seed},
            "dev": {"@readers": "sud.GenericTagCorpus.v1", "path": "${paths.corpus}",
                    "split": "dev", "seed": seed}},
        "training": {
            "dev_corpus": "corpora.dev", "train_corpus": "corpora.train",
            "seed": "${system.seed}", "gpu_allocator": "${system.gpu_allocator}",
            "dropout": 0.1, "accumulate_gradient": 1, "patience": patience,
            "max_epochs": 0, "max_steps": max_steps, "eval_frequency": eval_frequency,
            "frozen_components": [], "annotating_components": [],
            "before_to_disk": None, "before_update": None,
            "optimizer": {"@optimizers": "Adam.v1", "beta1": 0.9, "beta2": 0.999,
                          "L2_is_weight_decay": True, "L2": 0.01, "grad_clip": 1.0,
                          "use_averages": False, "eps": 1e-08, "learn_rate": 0.001},
            "batcher": {"@batchers": "spacy.batch_by_words.v1", "discard_oversize": False,
                        "tolerance": 0.2, "get_length": None,
                        "size": {"@schedules": "compounding.v1", "start": 100, "stop": 1000,
                                 "compound": 1.001, "t": 0.0}},
            "logger": {"@loggers": "spacy.ConsoleLogger.v1", "progress_bar": False},
            "score_weights": {"pos_acc": 0.3, "morph_acc": 0.7,
                              "morph_per_feat": None, "tag_acc": None}},
        "initialize": {"vectors": "${paths.vectors}", "init_tok2vec": "${paths.init_tok2vec}",
                       "vocab_data": None, "lookups": None, "before_init": None,
                       "after_init": None, "tokenizer": {}, "components": {}},
        "pretraining": {},
        "components": {
            "tok2vec": {"factory": "tok2vec",
                        "model": {"@architectures": "spacy.Tok2Vec.v2",
                                  "embed": {"@architectures": "sud.GenericTagEmbed.v1",
                                            "width": "${components.tok2vec.model.encode.width}",
                                            "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                                            "rows": [8000, 3000, 3000, 3000],
                                            "lang_embed": True,
                                            "lang_slots": {lg: i for i, lg in enumerate(langs)},
                                            "lang_embed_rows": len(langs) + spare,
                                            "lang_embed_dim": lang_dim},
                                  "encode": {"@architectures": "spacy.MaxoutWindowEncoder.v2",
                                             "width": width, "depth": depth,
                                             "window_size": 1, "maxout_pieces": 3}}},
            "morphologizer": {
                "factory": "morphologizer", "overwrite": True, "extend": False,
                "label_smoothing": 0.05,
                "model": {"@architectures": "spacy.Tagger.v2", "nO": None, "normalize": False,
                          "tok2vec": {"@architectures": "spacy.Tok2VecListener.v1",
                                      "width": "${components.tok2vec.model.encode.width}",
                                      "upstream": "*"}},
                "scorer": {"@scorers": "spacy.morphologizer_scorer.v1"}},
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--out", default="configs/config_g2_tagger.cfg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-frequency", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4000)
    ap.add_argument("--spare", type=int, default=32)
    ap.add_argument("--lang-dim", type=int, default=0)
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    langs = sorted(k for k, v in man.items() if v["pool"] == "train")
    cfg = build(a.corpus, langs, a.seed, a.width, a.depth, a.max_steps, a.eval_frequency,
                a.patience, a.spare, a.lang_dim)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Config(cfg).to_disk(a.out)
    print(f"wrote {a.out}  ({len(langs)} languages + {a.spare} spare rows)")


if __name__ == "__main__":
    main()
