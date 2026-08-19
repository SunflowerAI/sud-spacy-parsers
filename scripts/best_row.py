#!/usr/bin/env python3
"""Read a `spacy train` log by COLUMN NAME instead of by position.

Every arm in this project logs a different column set — the multi-task sa arm has five loss columns,
the SUD layer has six MISC columns, lzh's base has three — so `awk '{print $8}'` means DEP_UAS in one
log, DEP_LAS in another and MORPH_ACC in a third. That has produced wrong numbers repeatedly here,
and silently: every candidate column is a plausible-looking percentage.

The header cannot simply be whitespace-split — spaCy writes `LOSS TOK2VEC` and truncates to
`LOSS SUD_S...`, so a loss column is one or two tokens and two of them can carry the SAME truncated
name. Only the metric columns are addressed by name; the loss columns are counted, not named.

    best_row.py LOG [--by DEP_LAS] [--cols DEP_LAS,SENTS_F]     # best row by a metric
    best_row.py LOG --list                                      # what columns this log has
"""
import argparse
import re

# spaCy pads every column, so the header splits on RUNS of 2+ spaces — that keeps `LOSS TOK2VEC`
# and `LOSS SUD_S...` as ONE label each. Splitting on single whitespace does not: it turns
# `LOSS TOK2VEC` into two tokens and silently shifts every metric one column left.
_SPLIT = re.compile(r"\s{2,}")


def parse(path):
    header = None
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if header is None:
            if line.startswith("E ") and "#" in line:
                labels = [t for t in _SPLIT.split(line.strip()) if t]
                # drop E and #, then the loss columns; what remains are the metrics, in order
                header = [t for t in labels[2:] if not t.startswith("LOSS")]
            continue
        f = line.split()
        if len(f) < 3 or not f[0].isdigit():
            continue
        try:
            vals = [float(x) for x in f[1:]]
        except ValueError:
            continue
        rows.append((int(f[0]), vals))
    if header is None or not rows:
        raise SystemExit(f"{path}: no table found")
    n = len(rows[0][1])
    # vals = [step, *losses, *metrics]; the metrics are the LAST len(header) entries, so the number
    # of loss columns never has to be counted.
    if len(header) > n:
        raise SystemExit(f"{path}: header has {len(header)} metrics but rows have {n} values")
    idx = {name: n - len(header) + i for i, name in enumerate(header)}
    return header, idx, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--by", default="DEP_LAS")
    ap.add_argument("--cols", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    header, idx, rows = parse(a.log)
    if a.list:
        print(f"{a.log}: {', '.join(header)}")
        return
    if a.by not in idx:
        raise SystemExit(f"{a.log} has no column {a.by}; it has: {', '.join(header)}")
    cols = a.cols.split(",") if a.cols else header
    best = max(rows, key=lambda r: r[1][idx[a.by]])
    epoch, vals = best
    shown = "  ".join(f"{c} {vals[idx[c]]:.2f}" for c in cols if c in idx)
    print(f"{a.log}  best by {a.by} @ epoch {epoch} step {int(vals[0])}:  {shown}")


if __name__ == "__main__":
    main()
