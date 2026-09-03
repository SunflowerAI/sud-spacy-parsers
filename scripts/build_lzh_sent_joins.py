#!/usr/bin/env python3
"""Harvest the relation that joins a following clause, from the treebank's own annotation.

`sent_join` merges sentences the parser opened where the reading convention says there is no
boundary. Merging means giving the second sentence's root a head AND A LABEL, and the label is not
a matter of taste — the treebank annotates this configuration tens of thousands of times. This
script reads it off, `cross_unit_rules.py`-style: derive on train, keep only what clears a
dominance and a count threshold, and let everything else fall through to a single declared default.

WHAT THE DATA SAYS (Kyoto train, punctuation-restored, rule-merged):

  * Inside a BALANCED quoted span there is essentially ONE attachment point: 1 743 of 1 755 spans
    have exactly one token whose head lies outside the span, and it is `comp:obj` of the speech verb
    (1 641 of 1 755; head 曰 1 504, 問 29, 云 28, 謂 18). Of the 13 further attachments that do
    exist, 12 also attach to 曰. So a second quoted clause is NOT chained to the first — it hangs
    off the SPEECH VERB, with the same relation. `sent_join` implements that directly and does not
    need this table for it; the table is the fallback for a quote with no external governor.
  * Elsewhere the relation is sharply conditioned on the UPOS OF THE PREVIOUS UNIT'S HEAD, which is
    why this table is keyed on it rather than being one constant:

        after a PAUSE mark          prev head VERB  -> comp:obj 69%   NOUN  -> conj:coord 87%
                                    prev head PROPN -> conj:coord 95%  AUX  -> comp:obj 47%
        after a SENTENCE-FINAL mark prev head PROPN -> conj:coord 98%  NOUN -> conj:coord 90%
                                    prev head VERB  -> parataxis 33% / comp:obj 32% / conj:coord 19%

    Note the VERB row after a full stop: no relation clears 50 %, so at the default thresholds it is
    NOT harvested and falls through to the default. That is the point of a threshold — the cell is
    genuinely undecided and a majority vote there would be memorising noise.

⚠ AND THE HONEST CAVEAT: **31.2 % of gold sentence blocks END at a pause mark.** Kyoto does break
sentences at commas, frequently. Refusing to (`pause_join`) is a reading convention imposed at
inference, not treebank fidelity, and the treebank's own gold will score it down accordingly.

Usage:
    build_lzh_sent_joins.py --out models/lzh_sent_joins.json [--min-dominance 0.5] [--min-count 20]
"""
import argparse
import collections
import json
import pathlib

TRAIN = ("assets_lzh/SUD_Classical_Chinese-Kyoto/"
         "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")
PAUSE = "，、；：,;:"
FINAL = "。！？!?."
OTHER_MARKS = "「』」『（）〔〕《》【】…"


def blocks(path):
    cur = []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                yield cur
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f)
    if cur:
        yield cur


def unit_head(b, lo, hi):
    """The head of a contiguous content run: the token whose own head lies outside it."""
    for i in range(lo, hi + 1):
        h = int(b[i][6])
        if h == 0 or not (lo <= h - 1 <= hi):
            return i
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    # TWO BARS, and they exist because one bar cannot serve both ends of this distribution.
    # `cross_unit_rules.py` uses >=90 % on >=20, which is right there because a wrong rule writes
    # false structure into the TRAINING data. Here a rule only labels an arc the parser did not
    # produce at all, so a 69 %-accurate label on the 6 545-example `pause`+VERB cell plainly beats
    # falling through to a default that is right 17 % of the time. But that argument does not
    # licence a 50 %-of-48 cell. So: a small cell must be nearly unanimous, and a moderate majority
    # is accepted only where the evidence is large.
    ap.add_argument("--min-dominance", type=float, default=0.8,
                    help="dominance required of a cell with >= --min-count examples")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--big-dominance", type=float, default=0.5,
                    help="dominance required of a cell with >= --big-count examples")
    ap.add_argument("--big-count", type=int, default=500)
    ap.add_argument("--default", default="conj:coord")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tally = collections.defaultdict(collections.Counter)
    for b in blocks(a.train):
        forms = [f[1] for f in b]
        units, kinds, cu = [], [], []
        for i, x in enumerate(forms):
            if x in PAUSE or x in FINAL:
                if cu:
                    units.append(cu)
                    kinds.append("pause" if x in PAUSE else "final")
                cu = []
            elif x in OTHER_MARKS:
                continue
            else:
                cu.append(i)
        if cu:
            units.append(cu)
            kinds.append(None)
        for k in range(1, len(units)):
            prev, this = units[k - 1], units[k]
            ph = unit_head(b, prev[0], prev[-1])
            th = unit_head(b, this[0], this[-1])
            h = int(b[th][6])
            if h == 0 or h - 1 != ph:          # only the configuration the merge actually creates
                continue
            tally[(kinds[k - 1], b[ph][3])][b[th][7]] += 1

    table, rejected = {}, []
    for (kind, pos), c in sorted(tally.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        dep, k = c.most_common(1)[0]
        dom = k / n
        if (n >= a.min_count and dom >= a.min_dominance) or \
                (n >= a.big_count and dom >= a.big_dominance):
            table[f"{kind}|{pos}"] = dep
        else:
            rejected.append((kind, pos, n, dep, dom))
    print(f"harvested {len(table)} cells (>= {a.min_dominance:.0%} on >= {a.min_count}, "
          f"or >= {a.big_dominance:.0%} on >= {a.big_count}); default {a.default!r}")
    for key, dep in sorted(table.items()):
        kind, pos = key.split("|")
        n = sum(tally[(kind, pos)].values())
        share = tally[(kind, pos)][dep] / n
        print(f"   {kind:<6} prev head {pos:<6} -> {dep:<12} {share:5.0%} of {n}")
    for kind, pos, n, dep, share in rejected:
        if n >= a.min_count:
            print(f"   REJECTED {kind:<6} prev head {pos:<6}: best {dep} only {share:.0%} of {n}"
                  f" — falls through to the default")

    payload = {"__meta__": {"train": a.train, "min_dominance": a.min_dominance,
                            "min_count": a.min_count},
               "default": a.default, "table": table}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
