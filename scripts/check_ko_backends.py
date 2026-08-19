#!/usr/bin/env python3
"""Do the two Korean analyser backends give the parser the SAME channel?

This is a release blocker, not a curiosity. `sud.KoAnalyserEmbed.v1` stamps the backend into the
model bytes and REFUSES a mismatch on load, because an arm fed a channel it never saw parses quietly
worse. The arms were trained on this machine against `natto-py` + Homebrew mecab-ko; the wheel would
declare `python-mecab-ko`, which vendors its own mecab-ko-dic. If those two disagree, the shipped
model must be retrained against the one it declares. If they agree token for token, then the
fingerprint is naming the wrong thing — the DICTIONARY is the channel and the binding is not — and
refusing on the binding would reject a perfectly good install.

So: run every eojeol in the treebank through both, and compare the full (morpheme, tag) sequence.

    .venv/bin/pip install --target /tmp/kotest python-mecab-ko
    PYTHONPATH=/tmp/kotest .venv/bin/python scripts/check_ko_backends.py

Needs BOTH backends importable at once; `KO_ANALYSER_BACKEND` picks between them per subprocess.
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONLLU = ("assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu",
                  "assets_ko/SUD_Korean-GSD/ko_gsd-sud-test.relabeled_ext.conllu")


def forms(paths) -> list:
    seen = {}
    for p in paths:
        for line in pathlib.Path(p).open(encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            seen.setdefault(f[1], None)
    return list(seen)


def analyse_with(backend: str, words: list) -> list:
    """One subprocess per backend: both bind a global MeCab and only one can be `the` backend in a
    process, so importing them together and hoping is not a test."""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import ko_analyser\n"
        "words = json.load(sys.stdin)\n"
        "out = [ko_analyser.analyse(w) for w in words]\n"
        "json.dump({'backend': ko_analyser.fingerprint(), 'out': out}, sys.stdout)\n"
    )
    env = dict(os.environ, KO_ANALYSER_BACKEND=backend)
    r = subprocess.run([sys.executable, "-c", code], input=__import__("json").dumps(words),
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"{backend} failed:\n{r.stderr[-2000:]}")
    return __import__("json").loads(r.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu", nargs="*", default=list(DEFAULT_CONLLU))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    words = forms(args.conllu)
    if args.limit:
        words = words[:args.limit]
    print(f"{len(words)} distinct eojeol from {len(args.conllu)} file(s)")

    a = analyse_with("natto-py", words)
    b = analyse_with("python-mecab-ko", words)
    print(f"  A: {a['backend']}")
    print(f"  B: {b['backend']}")

    same = stem_same = tag_same = 0
    diffs = []
    for w, x, y in zip(words, a["out"], b["out"]):
        x = [tuple(m) for m in x]
        y = [tuple(m) for m in y]
        same += x == y
        stem_same += (x[0][0] if x else None) == (y[0][0] if y else None)
        tag_same += [t for _, t in x] == [t for _, t in y]
        if x != y and len(diffs) < 10:
            diffs.append((w, x, y))
    n = len(words)
    print(f"\nidentical analysis      {same}/{n} = {same/n:.4%}")
    print(f"identical first morpheme {stem_same}/{n} = {stem_same/n:.4%}   <- the lexical key")
    print(f"identical tag sequence   {tag_same}/{n} = {tag_same/n:.4%}   <- the multi-hot block")
    for w, x, y in diffs:
        print(f"    {w}\n      natto : {'+'.join(m for m, _ in x)}  [{'+'.join(t for _, t in x)}]"
              f"\n      pymecab: {'+'.join(m for m, _ in y)}  [{'+'.join(t for _, t in y)}]")
    if same == n:
        print("\nIDENTICAL. The dictionary is the channel and the binding is not, so the "
              "fingerprint must name the dictionary.")
    else:
        print(f"\nDIFFERENT on {n - same} forms. An arm trained against one must NOT be shipped "
              f"declaring the other.")


if __name__ == "__main__":
    main()
