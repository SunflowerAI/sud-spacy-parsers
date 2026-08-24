#!/usr/bin/env python3
"""Generate the v2 arm configs. Forked from `make_generic_config.py`; the FEATS derivation is v1's.

Arms, and what each one isolates:

    g2_base        UPOS + FEATS                        the honest no-channel baseline
    g2_typ         + the 8-bit typology channel        THE HEADLINE
    g2_typ_ctl     the channel, carrying zeros         what the PARAMETERS buy
    g2_typ_der     the channel, wrong languages        what the RIGHT profile buys -- THE GATE
    g2_nofeats     UPOS + typology, no FEATS           the floor for a FEATS-less test language
    g2_langid      + a language embedding              what knowing the language buys (held-in only)
    g2_typ12       typology with 4 `measured` flags    what the `00 = unknown` overload costs
    g2_feats_all   --min-langs 1                       what the FEATS channel cut cost

⚠ **`g2_base` AND `g2_typ_ctl` ARE BOTH NEEDED AND ARE NOT THE SAME ARM.** The first has no typology
block at all; the second has the block with nothing in it. The delta that gets QUOTED is against
`g2_typ_ctl`, because adding a block adds `width * nP * width` Maxout weights and a gain from the
parameters is not a gain from typology (NEGATIVE-RESULTS.md, "always run a capacity control").

⚠ **`--min-langs` DEFAULTS TO 5 HERE, AGAINST v1's 1, AND THE REASON IS THE OPPOSITE SETTING.** v1
kept every FEATS category present anywhere on the grounds that a per-language category is cheap and
dropping it taxes exactly the low-resource languages the arm exists for. That held when all thirteen
languages were also TEST languages. Here the test languages were never seen in training, so a
category attested in one training treebank out of eighty cannot transfer by construction -- it can
only ever fire on that treebank -- while each channel adds a `width x nP x width` block to the
Maxout. `g2_feats_all` measures the cost of the cut rather than assuming it.
"""
import argparse
import collections
import glob
import json
import os
import pathlib

from thinc.api import Config

ARMS = ("g2_base", "g2_typ", "g2_typ_ctl", "g2_typ_der", "g2_nofeats",
        "g2_langid", "g2_typ12", "g2_feats_all", "g2_langemb")


def feats_inventory(corpus_dir, min_langs=5, min_token_pct=0.0):
    """(feature list, row counts, report) over the prepared TRAIN CoNLL-U. v1's, plus a token floor."""
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
                    if not k:
                        continue
                    vals[k].add(v)
                    langs[k].add(lang)
                    tokens[k] += 1
    keep = sorted(k for k in vals
                  if len(langs[k]) >= min_langs and 100 * tokens[k] / total >= min_token_pct)
    rows = []
    for k in keep:
        n = 4 * len(vals[k])
        r = 8
        while r < n:
            r *= 2
        rows.append(r)
    report = [{"feat": k, "values": len(vals[k]), "langs": len(langs[k]),
               "token_pct": round(100 * tokens[k] / total, 3), "rows": r}
              for k, r in zip(keep, rows)]
    dropped = [{"feat": k, "langs": len(langs[k]), "token_pct": round(100 * tokens[k] / total, 3)}
               for k in sorted(vals) if k not in set(keep)]
    return keep, rows, report, dropped


def build(arm, corpus, typology, feats, feat_rows, langs, seed, width, depth,
          hidden_width, max_steps, eval_frequency, patience, spare=32, lang_dim=0):
    cfg = {
        "paths": {"corpus": corpus, "typology": typology, "vectors": None, "init_tok2vec": None},
        "system": {"gpu_allocator": None, "seed": seed},
        "nlp": {
            # `xx` is spaCy's MultiLanguage. The arm has NO tokenizer of its own and never runs one:
            # the reader hands it gold tokens and at inference the caller supplies them.
            # `spacy.Tokenizer.v1` is here only because the config schema requires a tokenizer, and
            # reaching it at all would mean the input regime was wrong.
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
                        "@architectures": "sud.GenericEmbed.v2",
                        "width": "${components.tok2vec.model.encode.width}",
                        "upos_rows": 64,
                        "feats": feats, "feat_rows": feat_rows,
                        "typology": None, "typology_shuffle": False,
                        "typology_constant": False, "typology_dim": 8,
                        "lang_id": False, "langs": [],
                        "lang_embed": False, "lang_embed_rows": 0,
                        "lang_embed_dim": 0, "lang_slots": {},
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
                # 1, never the default 30. A frequency floor DELETES labels silently, with their
                # recall pinned to zero (`docs/dravidian.md`: 7 of ta TTB's 19 deprels). This arm's
                # inventory has a long tail -- `goeswith` at 60 tokens in 1.5 M -- and every one of
                # those labels is attested in some test language.
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
    ty = "${paths.typology}"
    if arm == "g2_base":
        pass                                        # no typology block at all
    elif arm == "g2_typ":
        embed["typology"] = ty
    elif arm == "g2_typ_ctl":
        embed.update({"typology": ty, "typology_constant": True})
    elif arm == "g2_typ_der":
        embed.update({"typology": ty, "typology_shuffle": True})
    elif arm == "g2_nofeats":
        embed.update({"typology": ty, "feats": [], "feat_rows": []})
    elif arm == "g2_langid":
        embed.update({"lang_id": True, "langs": langs})
    elif arm == "g2_typ12":
        embed.update({"typology": ty, "typology_dim": 12})
    elif arm == "g2_langemb":
        # A FREE per-language vector, constant across a document, in place of the four hand-picked
        # bits. Spare rows are left unallocated so an unseen language can be given one and fitted
        # on a small sample -- `lang_id`'s one-hot cannot do that, since its input width is the
        # number of training languages.
        embed.update({"lang_embed": True,
                      "lang_slots": {lg: i for i, lg in enumerate(langs)},
                      "lang_embed_rows": len(langs) + spare,
                      "lang_embed_dim": lang_dim})
    elif arm == "g2_feats_all":
        embed["typology"] = ty                      # feats/feat_rows are passed in wider
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
    ap.add_argument("--lang-dim", type=int, default=0,
                    help="width of the per-language vector before it is projected to the block "
                         "width; 0 means use the block width itself")
    ap.add_argument("--spare", type=int, default=32,
                    help="unallocated embedding rows for languages met after training")
    ap.add_argument("--outdir", default="configs")
    a = ap.parse_args()

    feats, rows, report, dropped = feats_inventory(a.corpus_conllu, a.min_langs, a.min_token_pct)
    wide, wide_rows, _, _ = feats_inventory(a.corpus_conllu, 1, 0.0)
    print(f"FEATS channels at --min-langs {a.min_langs}: {len(feats)} "
          f"(at 1 it would be {len(wide)})")
    print(f"{'feat':22s} {'vals':>5s} {'langs':>6s} {'tok%':>7s} {'rows':>5s}")
    for r in report:
        print(f"  {r['feat']:20s} {r['values']:5d} {r['langs']:6d} {r['token_pct']:7.3f} "
              f"{r['rows']:5d}")
    print(f"\ndropped {len(dropped)} categories below the threshold; the ten most frequent:")
    for r in sorted(dropped, key=lambda r: -r["token_pct"])[:10]:
        print(f"  {r['feat']:20s} langs {r['langs']:3d}  {r['token_pct']:6.3f} % of tokens")

    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))
    train_langs = sorted(k for k, v in man["languages"].items() if v["pool"] == "train")
    print(f"\nlang_id inventory: {len(train_langs)} training languages")

    # n_blocks drives the Maxout, which is where the parameters actually are.
    for n, label in ((len(feats) + 1, f"min-langs {a.min_langs}"), (len(wide) + 1, "min-langs 1")):
        print(f"  {label:14s} n_blocks={n + 1:4d} (with typology)  "
              f"Maxout {a.width * 3 * a.width * (n + 1):,} params")

    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for arm in a.arm:
        f, r = (wide, wide_rows) if arm == "g2_feats_all" else (feats, rows)
        cfg = build(arm, a.corpus, a.typology, f, r, train_langs, a.seed, a.width, a.depth,
                    a.hidden_width, a.max_steps, a.eval_frequency, a.patience, a.spare, a.lang_dim)
        out = outdir / f"config_{arm}{a.suffix}.cfg"
        # interpolate=False: the default resolves ${paths.corpus} to null and silently breaks CLI
        # path overrides (CLAUDE.md; this caused E913).
        Config(cfg).to_disk(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
