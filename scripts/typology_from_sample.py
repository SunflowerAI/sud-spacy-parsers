#!/usr/bin/env python3
"""Derive a test language's typological profile from a SMALL hand-annotated sample.

The deployable middle ground between the two conditions already measured, both of which failed for
opposite reasons:

    Grambank/WALS   free, needs no data at all -- but agrees with the treebank on 52-71 % of
                    fields, and at that accuracy the channel scores no better than a deliberately
                    deranged profile (53.28 vs 53.40 macro LAS).
    the treebank    accurate, and worth +1.68 over the external profile -- but it is an ORACLE:
                    knowing it presupposes the annotated corpus the whole exercise is trying to
                    do without.

This script simulates the realistic case: a linguist annotates N sentences of the target language
before any automation is allowed, and the profile is read off those. The question it answers is how
large N has to be.

⚠ **THE SAMPLE MUST NOT BE PART OF THE SCORED TEST SET.** It is drawn from the language's `train`
split -- data the experiment otherwise never touches, since these are test languages. Five of the
twenty test languages are test-only treebanks with no such data and cannot take part; they are
reported as excluded rather than quietly profiled from the data they are scored on.

⚠ The sample is drawn at RANDOM (seeded) rather than as a contiguous passage. That is the
optimistic reading: a linguist working through one text would see a narrower range of
constructions, so treat these sample sizes as a lower bound on what real annotation would need.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_typology_v2 import FIELDS, bits_from, measure  # noqa: E402
from prep_generic import read_conllu  # noqa: E402


def sample_sentences(paths, n, seed):
    sents = [s for p in paths for s in read_conllu(p)]
    rng = random.Random(seed)
    if n and n < len(sents):
        idx = sorted(rng.sample(range(len(sents)), n))
        sents = [sents[i] for i in idx]
    return sents


def write_temp(sents, path):
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            for row in s.rows:
                fh.write("\t".join(row) + "\n")
            fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--oracle", default="assets_typ/typology_treebank.json")
    ap.add_argument("--sizes", type=int, nargs="*", default=[25, 50, 100, 200, 500])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="assets_typ/sampled")
    ap.add_argument("--scratch", default="/tmp/typ_sample.conllu")
    # Small samples cannot meet the full-corpus minimums, so they are scaled with N. The
    # alternative -- keeping them -- makes every field come out `00` and measures nothing.
    ap.add_argument("--min-arcs-frac", type=float, default=0.05,
                    help="min arcs as a fraction of sampled tokens, floored at 10")
    a = ap.parse_args()

    inv = {c["lcode"] or c["lang_name"]: c for c in
           json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))["corpora"]}
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    oracle = json.loads(pathlib.Path(a.oracle).read_text(encoding="utf-8"))["languages"]
    test = sorted(k for k, v in man.items() if v["pool"] == "test")

    usable, skipped = [], []
    for k in test:
        paths = inv[k]["paths"].get("train") or inv[k]["paths"].get("dev")
        (usable if paths else skipped).append(k)
    print(f"{len(usable)} test languages can be profiled from unscored data; "
          f"{len(skipped)} cannot ({' '.join(skipped)})")

    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    scratch = pathlib.Path(a.scratch)

    for n in a.sizes:
        table, stats = {}, collections.Counter()
        for k in usable:
            paths = inv[k]["paths"].get("train") or inv[k]["paths"].get("dev")
            sents = sample_sentences(paths, n, a.seed)
            write_temp(sents, scratch)
            ntok = sum(len(s) for s in sents)
            m = measure([str(scratch)])
            # Scaled with N, but never STRICTER than the full-corpus defaults: at 1 000 sentences
            # a proportional floor demanded 855 arcs against the oracle's own 100, so large
            # samples were held to a harder standard than the oracle and the agreement curve
            # plateaued at 0.86 for a reason that had nothing to do with the data.
            prop = max(10, int(a.min_arcs_frac * ntok))
            bits, _ = bits_from(m,
                                min_arcs=min(prop, 100), min_verbs=min(prop, 200),
                                min_nouns=min(prop, 500), gender_floor=0.20)
            b = [bits[f] for f in FIELDS]
            table[k] = {"bits": b, "sampled_sents": len(sents), "sampled_tokens": ntok,
                        "sources": {f: f"sample:{n}sents" for f in FIELDS}}
            ob = oracle[k]["bits"]
            for i in range(len(FIELDS)):
                stats["bit_total"] += 1
                stats["bit_match"] += int(b[i] == ob[i])
            for i in (0, 2, 4, 6):
                stats["field_total"] += 1
                stats["field_match"] += int(b[i] == ob[i] and b[i + 1] == ob[i + 1])
                stats["field_unknown"] += int(b[i] == 0 and b[i + 1] == 0)
        p = out / f"typology_sample_{n}.json"
        json.dump({"meta": {"n_sents": n, "seed": a.seed, "source": "sample"},
                   "languages": table}, open(p, "w", encoding="utf-8"), indent=1)
        bm = stats["bit_match"] / max(stats["bit_total"], 1)
        fm = stats["field_match"] / max(stats["field_total"], 1)
        fu = stats["field_unknown"] / max(stats["field_total"], 1)
        summary[n] = {"bit_agree": bm, "field_agree": fm, "field_unknown": fu}
        tok = sum(v["sampled_tokens"] for v in table.values()) / len(table)
        print(f"  {n:4d} sents (~{tok:6.0f} tok/lang):  bits {bm:.2f}   whole fields {fm:.2f}   "
              f"unknown fields {fu:.2f}   -> {p}")

    print("\nFor reference, the databases against the same oracle: bits ~0.72, whole fields ~0.62")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
