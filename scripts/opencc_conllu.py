#!/usr/bin/env python
"""Duplicate a CoNLL-U treebank in the other Han script via OpenCC.

OpenCC ``s2t``/``t2s`` are character-level and length-preserving, so converting
each FORM (and LEMMA, and every ``text`` comment) independently keeps the
tokenisation, the 1:1 token alignment and every head/deprel/XPOS untouched —
only the script changes. ``id`` comments (``sent_id``, ``newdoc id``) get a
suffix so the converted copy can be concatenated with the original without
collisions.

Generalises ``zh_simp2trad.py`` (which is the ``--config s2t`` case) and is used
for lzh with ``--config t2s``.

Usage:
    opencc_conllu.py IN.conllu OUT.conllu [--config t2s] [--suffix -hans]
"""
import argparse
import opencc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--config", default="s2t", help="OpenCC config, e.g. s2t or t2s")
    ap.add_argument("--suffix", default="-hant", help="appended to sent_id / newdoc id")
    args = ap.parse_args()

    cc = opencc.OpenCC(args.config)

    def conv(s: str) -> str:
        return s if s == "_" else cc.convert(s)

    out = []
    for line in open(args.src, encoding="utf-8"):
        if line.startswith("#"):
            body = line[1:].lstrip()
            if body.startswith("text") or body.startswith("newpar text"):
                key, sep, val = line.partition("=")
                out.append(f"{key}{sep}{cc.convert(val.rstrip(chr(10)))}\n")
            elif body.startswith("sent_id") or body.startswith("newdoc id"):
                out.append(line.rstrip("\n") + args.suffix + "\n")
            else:
                out.append(line)
            continue
        if not line.strip():
            out.append(line)
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) == 10:
            cols[1] = conv(cols[1])  # FORM
            cols[2] = conv(cols[2])  # LEMMA
        out.append("\t".join(cols) + "\n")

    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
