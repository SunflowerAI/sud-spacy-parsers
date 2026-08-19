#!/usr/bin/env python3
"""Sub-character channels for the lzh segmenter: Kangxi radical, and Qieyun 音韻地位.

WHY THESE TWO, AND WHY QIEYUN IS NOT RULED OUT BY THE EARLIER PROBE. `NEGATIVE-RESULTS.md` records
a sub-character probe where the radical (57.00) beat Qieyun (48.06, against a 44.59 null) at
predicting LEXICAL CLASS. That does not transfer here: a transliteration is chosen for its SOUND and
its characters' meanings are irrelevant, so the graphic channel is the one expected to mislead. The
character-pair probe bears this out -- on bigrams never merged in training, Qieyun 0.092 vs radical
0.055 -- i.e. the ordering reverses on this task.

RADICAL, not IDS. Also from the earlier probe: the Kangxi radical (Unihan `kRSUnicode`, Unicode
licence) beat full IDS (cjkvi-ids, GPLv2) and adding IDS on top bought nothing. So the stronger
channel is also the licence-clean one, and lzh stays CC BY-SA.

⚠ QIEYUN IS A BAG OF READINGS AND THIS TAKES ONLY THE FIRST. 23.7 % of the 19,586 covered
characters are polyphonic and the reading cannot be chosen at inference, so a single categorical id
is a lossy summary. Recorded rather than hidden: if the channel pays, a multi-hot encoding is the
obvious next thing to try; if it does not, this is one reason why.
"""
import csv, pathlib

RADICAL = "assets_unihan/Unihan_IRGSources.txt"
QIEYUN = "assets_qieyun/guangyun.csv"

def radical_table(path=RADICAL):
    t = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#") or "kRSUnicode" not in line: continue
        cp, _, val = line.rstrip("\n").split("\t")[:3]
        t[chr(int(cp[2:], 16))] = val.split()[0].split(".")[0]
    return t

def qieyun_table(path=QIEYUN):
    t = {}
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ch, code = row.get("字頭"), row.get("音韻地位")
            if ch and code and ch not in t:      # first reading only -- see the warning above
                t[ch] = code
    return t

TABLES = {"radical": radical_table, "qieyun": qieyun_table}

def build(names):
    out = []
    for n in names:
        if n not in TABLES: raise SystemExit(f"unknown channel {n!r}; have {sorted(TABLES)}")
        t = TABLES[n]()
        print(f"  channel {n}: {len(t):,} characters, {len(set(t.values())):,} categories")
        out.append((n, t))
    return out

if __name__ == "__main__":
    build(["radical", "qieyun"])
