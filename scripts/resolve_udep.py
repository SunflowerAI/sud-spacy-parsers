"""Resolve SUD's underspecified `udep` from the UPOS of the head and the dependent.

cxSUD's converter wants already-disambiguated SUD (see the plan), but `udep` survives
SUD-spaCy's relabelling on 12,927 tokens of EWT+GUM train -- 9,254 sentences, 28% of the
corpus -- and treating it as unanalysed material leaves those tokens outside the
constructional analysis entirely.

The configuration (head UPOS, dependent UPOS) turns out to decide it. Two sources, used in
order of how much they can be trusted:

  1. RULED -- the 1,025 udep tokens SUD-spaCy's own `udep_ruled` pass resolved. Every
     configuration in that set is 100% pure and 5-fold cross-validation is perfect, which
     almost certainly means their pass keyed on these same features: this reproduces a
     deliberate disambiguation rather than discovering a fact. Covers 42.3% of udeps.
  2. ATTESTED -- for configurations that pass never resolved, how NON-udep tokens in the
     same configuration are labelled across the corpus. Much more data, but selection
     bias cuts against it: a token is udep precisely because SUD found it hard to label,
     so it may be atypical of its configuration. Applied only above a purity threshold.

Anything left unresolved keeps `udep` and is treated as a passive attachment -- claiming
nothing, which is the right answer when the evidence does not support a claim.
"""

from __future__ import annotations

import argparse
import collections
import re

SENT_ID = re.compile(r"^# sent_id = (.+)$", re.M)


def sentences(path):
    for blk in open(path).read().split("\n\n"):
        if not blk.strip():
            continue
        sid = SENT_ID.search(blk)
        toks = {}
        for line in blk.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) > 7 and "-" not in f[0] and "." not in f[0]:
                toks[f[0]] = f
        yield (sid.group(1) if sid else None), blk, toks


def head_upos(toks, f):
    return toks[f[6]][3] if f[6] in toks else "ROOT"


def kept_udep(toks, f):
    """`udep` tokens this pass must NOT resolve, by decision rather than by lack of evidence.

    * noun-level `of` keeps the treebank's own `udep` (user decision, 2026-10-25): the (NOUN, ADP)
      majority would otherwise turn every one of them into `mod`;
    * the partitive `of` under a DET (`some of the links`) stays `udep`, SUD's documented default
      for partitives (docs/udep-relabel.md; cxSUD restored it by hand on train on 2026-10-01).
    """
    if f[3] != "ADP" or f[2].lower() != "of":
        return None
    h = head_upos(toks, f)
    if h in ("NOUN", "PROPN"):
        return "noun-level of kept as udep"
    if h == "DET":
        return "partitive of kept as udep"
    return None


def build_table(raw_path, ruled_path, threshold):
    ruled = {s: t for s, _, t in sentences(ruled_path) if s} if ruled_path else {}
    from_ruled = collections.defaultdict(collections.Counter)
    attested = collections.defaultdict(collections.Counter)
    for sid, _, toks in sentences(raw_path):
        for tid, f in toks.items():
            base = f[7].split("@")[0]
            key = (head_upos(toks, f), f[3])
            if base == "udep":
                new = ruled.get(sid, {}).get(tid)
                if new is not None and not new[7].startswith("udep"):
                    from_ruled[key][new[7].split("@")[0]] += 1
            elif base != "root":
                attested[key][base] += 1

    table = {}
    for key, c in from_ruled.items():
        table[key] = (c.most_common(1)[0][0], "ruled", c.most_common(1)[0][1] / sum(c.values()))
    for key, c in attested.items():
        if key in table:
            continue
        rel, n = c.most_common(1)[0]
        purity = n / sum(c.values())
        if purity >= threshold and rel != "udep":
            table[key] = (rel, "attested", purity)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sud_conllu")
    ap.add_argument("--ruled", help="SUD-spaCy's *.udep_ruled.conllu, for source 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum purity for the attested fallback (default 0.5)")
    ap.add_argument("--table-from", nargs=2, metavar=("TRAIN_CONLLU", "TRAIN_RULED"),
                    help="derive the table from these (the TRAIN split and its udep_ruled file) "
                         "instead of from the input. Use it for dev and test: a table built from each "
                         "split on its own labelled the same configuration differently by split "
                         "(dev resolved PRON<-ADP to mod while train said orphan; 99 dev / 143 test "
                         "tokens disagreed).")
    args = ap.parse_args()

    if args.table_from:
        table = build_table(args.table_from[0], args.table_from[1], args.threshold)
    else:
        table = build_table(args.sud_conllu, args.ruled, args.threshold)
    stats = collections.Counter()
    out = []
    for _, blk, toks in sentences(args.sud_conllu):
        lines = []
        for line in blk.split("\n"):
            if line.startswith("#") or not line.strip():
                lines.append(line)
                continue
            f = line.split("\t")
            if len(f) > 7 and f[7].split("@")[0] == "udep" and kept_udep(toks, f):
                stats[kept_udep(toks, f)] += 1
            elif len(f) > 7 and f[7].split("@")[0] == "udep":
                hit = table.get((head_upos(toks, f), f[3]))
                if hit:
                    rel, source, _ = hit
                    # keep any @subtype the original carried
                    _, _, sub = f[7].partition("@")
                    f[7] = f"{rel}@{sub}" if sub else rel
                    line = "\t".join(f)
                    stats[source] += 1
                else:
                    stats["left as udep"] += 1
            lines.append(line)
        out.append("\n".join(lines))
    with open(args.out, "w") as fh:
        fh.write("\n\n".join(out) + "\n")
    total = sum(stats.values())
    print(f"{args.out}: {total} udep tokens, " +
          ", ".join(f"{k} {v} ({v/max(total,1):.1%})" for k, v in stats.most_common()))


if __name__ == "__main__":
    main()
