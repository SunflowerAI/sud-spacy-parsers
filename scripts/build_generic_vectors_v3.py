#!/usr/bin/env python3
"""Merge the per-language aligned assets into ONE table the v3 parser can hold in memory.

Forked from v1's `build_generic_vectors.py`. The merge, the fold-at-build-time rule and the
`lang\\tkey` layout are unchanged; what differs is that THREE ROLES now share one table and they
need three different vocabularies.

  training languages   the LEMMA types their treebanks actually contain. Rows the parser will
                       meet, and nothing else -- this is the channel it learns on.
  ENGLISH              a large FREQUENCY HEAD, because English is not a training language here. It
                       is the deployment lookup, the table an English gloss is resolved against,
                       and a gloss is an arbitrary English word rather than anything the English
                       treebank sample happens to contain. Keyed off en's own 5 617-token sample it
                       would miss most glosses, and every miss reaches the model as OOV -- which is
                       the failure mode this whole channel exists to avoid.
  test languages       a frequency head and NO treebank vocabulary. Their rows exist only for the
                       diagnostic upper bound ("how much better would a real aligned table have
                       been than a gloss?"), and a deployer has no treebank by construction. Their
                       `.conllu` files are never opened -- enforced in the manifest, not advised.

⚠ THE ROWS ARE NOT RE-NORMALISED, RE-PROJECTED OR RE-FITTED, AND MUST NOT BE. `docs/aligned-
vectors.md` measures what a per-language transformation costs: same anchors, same dimensionality, a
per-language PCA instead of the one shared basis, and retrieval goes from 63.8 % @1 to 0.0 %. This
script SELECTS; it does not transform.

⚠ KEYED OFF THE TREEBANKS, NEVER OFF THE SAMPLED CORPUS. Keying it off `corpus_generic_v2/` would
tie the table to whatever budget `prep_generic_v2.py` last ran with: change the budget and the new
sample's fresh types have no rows, so they arrive as OOV. Nothing raises, the model trains, and it
is simply worse -- the shape of every "loads cleanly and is wrong" defect in CLAUDE.md.

    .venv/bin/python scripts/build_generic_vectors_v3.py --assets release_vectors_v3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aligned_vectors import AlignedVectors     # noqa: E402


def treebank_keys(paths, key_attr):
    """Every distinct lookup key these files contribute, with token counts for the coverage report.

    A `_` LEMMA column falls back to the FORM, never to a literal underscore: spaCy keeps CoNLL-U
    `_` as a literal string, which once taught a Sanskrit transducer `FORM -> "_"` on 5 043 tokens.
    """
    col = 2 if key_attr == "lemma" else 1
    keys = collections.Counter()
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"missing source treebank: {p}")
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8 or "-" in f[0] or "." in f[0]:
                    continue
                k = f[col] if f[col] not in ("_", "") else f[1]
                if k and k != "_":
                    keys[k] += 1
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="assets_vec/sources_v3.json")
    ap.add_argument("--assets", default="release_vectors_v3",
                    help="directory of sud_vec_<lang>_128d.npz")
    ap.add_argument("--out", default="assets_vec/generic_vec_v3.npz")
    ap.add_argument("--top-k", type=int, default=0,
                    help="extra frequency head for TRAINING languages (0 = treebank types only)")
    ap.add_argument("--en-top-k", type=int, default=200000,
                    help="frequency head for ENGLISH, the gloss lookup. Large on purpose: a gloss "
                         "is an arbitrary English word, not a treebank type.")
    ap.add_argument("--test-top-k", type=int, default=50000,
                    help="frequency head for TEST languages, which contribute no treebank types")
    ap.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    a = ap.parse_args()

    src = json.load(open(a.sources, encoding="utf-8"))
    langs = src["languages"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    all_keys, blocks, meta_langs = [], [], {}
    print(f"{'lang':5} {'pool':5} {'key':6} {'types':>7} {'in tbl':>7} {'tok cov':>8} {'head':>7} {'rows':>8}")
    for lang in sorted(langs):
        s = langs[lang]
        path = os.path.join(a.assets, f"sud_vec_{lang}_128d.npz")
        if not os.path.exists(path):
            sys.exit(f"missing aligned asset: {path}")
        av = AlignedVectors.load(path)

        counts = treebank_keys(s["treebank"], av.key_attr) if s["treebank"] else collections.Counter()
        wanted, hit_tokens, tot_tokens = {}, 0, 0
        for raw, n in counts.items():
            tot_tokens += n
            folded = av.fold(raw)
            if folded in av._index:
                wanted[folded] = av._index[folded]
                hit_tokens += n
        n_tb = len(wanted)

        head = a.en_top_k if lang == "en" else (a.test_top_k if s["pool"] == "test" else a.top_k)
        if head:
            # Assets are emitted in source-frequency order, so the first N rows ARE the N most
            # frequent -- the assumption `docs/aligned-vectors.md` scores its coverage table on.
            for i in range(min(head, len(av.keys))):
                wanted.setdefault(av.keys[i], i)

        order = sorted(wanted.items(), key=lambda kv: kv[1])
        rows = np.asarray([i for _, i in order], dtype="int64")
        blocks.append(av.vectors[rows].astype(a.dtype))
        all_keys.extend(f"{lang}\t{k}" for k, _ in order)
        meta_langs[lang] = {
            "key_attr": av.key_attr,
            "lowercased": bool(av.lower),
            "key_norm": av.meta.get("key_norm"),
            "pool": s["pool"],
            "treebank_types": n_tb,
            "token_coverage": round(hit_tokens / tot_tokens, 4) if tot_tokens else None,
            "head": head,
            "rows": len(order),
        }
        cov = f"{hit_tokens / tot_tokens:7.1%}" if tot_tokens else "      -"
        print(f"{lang:5} {s['pool']:5} {av.key_attr:6} {len(counts):7d} {n_tb:7d} {cov} "
              f"{head:7d} {len(order):8d}")

    V = np.vstack(blocks)
    meta = dict(languages=meta_langs, dim=int(V.shape[1]), dtype=a.dtype,
                assets=a.assets, sources=a.sources,
                note=("en carries a large frequency head because it is the DEPLOYMENT LOOKUP for "
                      "English glosses, not a training language. Test languages carry a head and "
                      "no treebank vocabulary."))
    np.savez_compressed(a.out, keys="\n".join(all_keys), vectors=V, meta=json.dumps(meta))
    mb = os.path.getsize(a.out) / 1e6
    print(f"\nwrote {a.out}: {V.shape[0]:,} rows x {V.shape[1]}d, {mb:.1f} MB")
    tr = [l for l, m in meta_langs.items() if m["pool"] == "train"]
    covs = [m["token_coverage"] for m in meta_langs.values() if m["token_coverage"] is not None]
    if covs:
        print(f"train token coverage: mean {sum(covs)/len(covs):.1%}, "
              f"min {min(covs):.1%}, over {len(tr)} languages")


if __name__ == "__main__":
    main()
