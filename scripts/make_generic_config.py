#!/usr/bin/env python3
"""Write `configs/config_generic*.cfg` — the generic arm and the four controls it is read against.

THE ARMS. A single number for the generic parser would mean nothing on its own, because adding a
128-d block adds parameters as well as information, and because "it works" and "it works
cross-lingually" are different claims. Five configs, differing in exactly one thing each:

    generic           UPOS + decomposed FEATS + the aligned vector.        the arm
    generic_ctl       `constant = true`: identical Linear, identical       what the PARAMETERS buy
                      parameter count, every token handed the zero
                      vector and the OOV flag.
    generic_shuf      `shuffle = true`: the same rows with the key-to-row  what the ALIGNMENT buys
                      correspondence destroyed within each language.
    generic_nofeats   `feats = []`: UPOS + vector, no morphology.          what MORPHOLOGY buys
    generic_langid    + a thirteen-row language embedding.                 what knowing the
                                                                           LANGUAGE buys

`generic_ctl` and `generic_shuf` are not redundant. `_ctl` asks whether the channel carries anything
at all; `_shuf` asks whether what it carries is the CROSS-LINGUAL alignment rather than merely some
per-language lexical signal, and it is the harder test -- a shuffled table still gives each language
a distinct, consistent, arbitrary code per wordform, which a monolingual parser could exploit and a
cross-lingual one could not. If `_shuf` matches the arm, the alignment did nothing and the model is
thirteen parsers in a trench coat.

`generic_langid` is the arm's own honesty check. The headline claim is that the model has no
parameter that varies with the language, so it cannot have memorised "Latin is verb-final" as a fact
about Latin rather than as a fact about Latin's UPOS/FEATS/vector distribution. Measuring what the
shortcut would have been worth is the only way to say it was not taken.

THE FEATS CHANNEL LIST IS DERIVED, NOT HARDCODED. Which morphological categories are worth a table
is a property of the corpus (`build_feats_inventory.py` makes the same argument per-treebank), and
across thirteen treebanks it is not guessable at all: the union is large, the intersection is nearly
empty, and four of the thirteen populate FEATS on under 5 % of their tokens. Rows are the next power
of two at or above 4x the distinct values, minimum 8 -- a value set is tiny, so the tables are
nearly free and under-provisioning is the only way to mask a real gain.

  ⚠ `--min-langs` DEFAULTS TO 1, i.e. every category present anywhere gets a channel, including the
  ones only Latin has. That is deliberate. A per-language category costs one small table and cannot
  confuse another language (a token that does not declare it hashes to `InflClass=` and lands on the
  same row as every other token that does not), whereas dropping it throws away real morphology for
  the language that has it -- and the low-resource languages are the ones with the most idiosyncratic
  FEATS inventories, so a majority threshold would tax exactly the languages this arm exists for.

    .venv/bin/python scripts/make_generic_config.py                    # all five
    .venv/bin/python scripts/make_generic_config.py --arm generic --suffix _lolo_ja
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

from thinc.api import Config

ARMS = ("generic", "generic_ctl", "generic_shuf", "generic_nofeats", "generic_langid",
        "generic_typ", "generic_typ_ctl")


def feats_inventory(corpus_dir, min_langs=1):
    """(feature list, row counts) over the prepared train CoNLL-U, plus a per-feature report."""
    vals = collections.defaultdict(set)
    langs = collections.defaultdict(set)
    tokens = collections.Counter()
    total = 0
    for p in sorted(glob.glob(os.path.join(corpus_dir, "*-train.conllu"))):
        lang = os.path.basename(p).split("-")[0]
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 6 or "-" in f[0] or "." in f[0]:
                    continue
                total += 1
                if f[5] == "_":
                    continue
                for kv in f[5].split("|"):
                    k, _, v = kv.partition("=")
                    vals[k].add(v)
                    langs[k].add(lang)
                    tokens[k] += 1
    keep = sorted(k for k in vals if len(langs[k]) >= min_langs)
    rows = []
    for k in keep:
        n = 4 * len(vals[k])
        r = 8
        while r < n:
            r *= 2
        rows.append(r)
    report = [{"feat": k, "values": len(vals[k]), "langs": len(langs[k]),
               "token_pct": round(100 * tokens[k] / total, 2), "rows": r}
              for k, r in zip(keep, rows)]
    return keep, rows, report


def build(arm, corpus, table, fingerprint, feats, feat_rows, seed, width, depth,
          hidden_width, max_steps, eval_frequency, patience, typology=None):
    cfg = {
        "paths": {"corpus": corpus, "table": table, "vectors": None, "init_tok2vec": None},
        "system": {"gpu_allocator": None, "seed": seed},
        "nlp": {
            # `xx` is spaCy's MultiLanguage. The arm has NO tokenizer of its own and never runs
            # one: the reader hands it gold tokens, and at inference the caller supplies them.
            # `spacy.Tokenizer.v1` is here only because the config schema requires a tokenizer,
            # and reaching it at all would mean the input regime was wrong.
            "lang": "xx",
            "pipeline": ["tok2vec", "parser"],
            "batch_size": 1000,
            "disabled": [], "before_creation": None, "after_creation": None,
            "after_pipeline_creation": None,
            "tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"},
            "vectors": {"@vectors": "spacy.Vectors.v1"},
        },
        "corpora": {
            "train": {"@readers": "sud.GenericCorpus.v1", "path": "${paths.corpus}",
                      "split": "train", "seed": seed},
            "dev": {"@readers": "sud.GenericCorpus.v1", "path": "${paths.corpus}",
                    "split": "dev", "seed": seed},
        },
        "training": {
            "dev_corpus": "corpora.dev", "train_corpus": "corpora.train",
            "seed": "${system.seed}", "gpu_allocator": "${system.gpu_allocator}",
            "dropout": 0.1, "accumulate_gradient": 1,
            "patience": patience, "max_epochs": 0, "max_steps": max_steps,
            "eval_frequency": eval_frequency,
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
            "score_weights": {"dep_uas": 0.3, "dep_las": 0.6, "sents_f": 0.1,
                              "dep_las_per_type": None, "sents_p": None, "sents_r": None},
        },
        "initialize": {"vectors": "${paths.vectors}", "init_tok2vec": "${paths.init_tok2vec}",
                       "vocab_data": None, "lookups": None, "before_init": None,
                       "after_init": None, "tokenizer": {}, "components": {}},
        "pretraining": {},
        "components": {
            "tok2vec": {
                "factory": "tok2vec",
                "model": {
                    "@architectures": "spacy.Tok2Vec.v2",
                    "embed": {
                        "@architectures": "sud.GenericEmbed.v1",
                        "width": "${components.tok2vec.model.encode.width}",
                        "upos_rows": 64,
                        "feats": feats, "feat_rows": feat_rows,
                        "table": "${paths.table}",
                        "fingerprint": fingerprint,
                        "vector_dim": None, "constant": False, "shuffle": False,
                        "lang_id": False,
                    },
                    "encode": {
                        "@architectures": "spacy.MaxoutWindowEncoder.v2",
                        "width": width, "depth": depth, "window_size": 1, "maxout_pieces": 3,
                    },
                },
            },
            "parser": {
                "factory": "parser", "moves": None, "update_with_oracle_cut_size": 100,
                "learn_tokens": False,
                # 1, never the default 30. At this corpus size a frequency floor DELETES labels --
                # 7 of ta TTB's 19 deprels, 19 of the combined Dravidian arm's 33 -- silently, with
                # their recall pinned to zero (`docs/dravidian.md`). The generic arm has 27 labels
                # and a long tail (`conj` at 17 tokens), so the floor would be doing exactly the
                # damage this parser exists to avoid.
                "min_action_freq": 1,
                "model": {"@architectures": "spacy.TransitionBasedParser.v2",
                          "state_type": "parser", "extra_state_tokens": False,
                          "hidden_width": hidden_width, "maxout_pieces": 3, "use_upper": True,
                          "nO": None,
                          "tok2vec": {"@architectures": "spacy.Tok2VecListener.v1",
                                      "width": "${components.tok2vec.model.encode.width}",
                                      "upstream": "*"}},
                "scorer": {"@scorers": "spacy.parser_scorer.v1"},
            },
        },
    }
    embed = cfg["components"]["tok2vec"]["model"]["embed"]
    if arm == "generic_ctl":
        # The capacity control constructs WITHOUT the table -- that is the point of it -- so the
        # width has to be stated rather than read off a file that may not be there.
        embed.update({"table": None, "fingerprint": None, "constant": True, "vector_dim": 128})
    elif arm == "generic_shuf":
        embed["shuffle"] = True
    elif arm == "generic_nofeats":
        embed.update({"feats": [], "feat_rows": []})
    elif arm == "generic_langid":
        embed["lang_id"] = True
    elif arm == "generic_typ":
        # Graded word-order profile per language. ORACLE in a LOLO arm -- the held-out language's
        # profile is derived from its own gold train, so this is an UPPER BOUND on typological
        # conditioning, not a deployable setting (scripts/build_typology.py).
        embed["typology"] = typology
    elif arm == "generic_typ_ctl":
        embed["typology"] = typology
        embed["typology_shuffle"] = True
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-conllu", default="assets_generic",
                    help="prepared CoNLL-U, read to derive the FEATS channel list")
    ap.add_argument("--corpus", default="corpus_generic",
                    help="the .spacy corpus directory the config points at")
    ap.add_argument("--table", default="assets_vec/generic_vec.npz")
    ap.add_argument("--typology", default="assets_vec/typology.json")
    ap.add_argument("--arm", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--suffix", default="", help="appended to each config name, e.g. _lolo_ja")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-langs", type=int, default=1,
                    help="keep a FEATS channel only if this many languages use it (see the "
                         "warning in the module docstring before raising it)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--hidden-width", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-frequency", type=int, default=400)
    ap.add_argument("--patience", type=int, default=3200)
    ap.add_argument("--outdir", default="configs")
    a = ap.parse_args()

    import numpy as np
    if not os.path.exists(a.table):
        sys.exit(f"missing {a.table} -- run scripts/build_generic_vectors.py first")
    fingerprint = json.loads(str(np.load(a.table, allow_pickle=False)["meta"]))["fingerprint"]

    feats, rows, report = feats_inventory(a.corpus_conllu, a.min_langs)
    print(f"{len(feats)} FEATS channels derived from {a.corpus_conllu}/*-train.conllu")
    print(f"{'feat':22} {'values':>7} {'langs':>6} {'tok%':>7} {'rows':>5}")
    for r in report:
        print(f"{r['feat']:22} {r['values']:>7} {r['langs']:>6} {r['token_pct']:>7.2f} {r['rows']:>5}")
    print(f"\ntable {a.table}  fingerprint {fingerprint}")

    os.makedirs(a.outdir, exist_ok=True)
    for arm in a.arm:
        cfg = build(arm, a.corpus, a.table, fingerprint, feats, rows, a.seed, a.width, a.depth,
                    a.hidden_width, a.max_steps, a.eval_frequency, a.patience, a.typology)
        out = os.path.join(a.outdir, f"config_{arm}{a.suffix}.cfg")
        # interpolate=False: the default resolves ${paths.train} to null and silently breaks CLI
        # path overrides (CLAUDE.md; this caused E913).
        Config(cfg).to_disk(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
