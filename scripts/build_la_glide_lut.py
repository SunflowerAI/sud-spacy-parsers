#!/usr/bin/env python3
"""Harvest which ``u`` letters are the CONSONANT (``v``), for the Latin orthography augmenter.

``scripts/la_orth.py`` needs to rewrite a Latin word between the two live spelling conventions --
``uita``/``vita``, ``seruus``/``servus`` -- which means knowing, for a word spelled with ``u``
throughout, which of its ``u``s stand for the glide. That is not derivable by suffix rule: after
``l``/``r``/``n`` both readings occur (``silua`` -> *silva*, but ``minuere`` stays vocalic), and it
is a lexical fact, not a phonotactic one.

It does not have to be guessed, though, because the treebanks disagree with each other about it in
a useful way. **ITTB writes ``u`` throughout** (0 ``v`` in 390 787 tokens); **PROIEL and Perseus
write ``v``** (13 516 tokens). So the v-writing half is a labelled corpus for the u-writing half:
every token PROIEL spells ``servus`` says that the u-spelling ``seruus`` is glide at position 3.

Two tables come out, and the second only exists to cover what the first misses:

  ``lex``  form (lowercased, macron-stripped, v->u) -> a bitmask over its ``u`` positions.
           Ambiguity is resolved by majority; the disagreements are single-token spelling noise
           (one ``qvae`` against 1 034 ``quae``), not real variation.
  ``ctx``  (prev-prev, prev, next) character context -> glide or not, kept only where the evidence
           is dominant. The residue rule, for words the harvest never saw.

Both are measured, held out by TYPE (see ``--report``), so the fallback's error rate is known
rather than assumed. Train-time only: nothing here ships in a wheel.

    .venv/bin/python scripts/build_la_glide_lut.py            # writes scripts/la_glide_lut.json.gz
    .venv/bin/python scripts/build_la_glide_lut.py --report   # + held-out evaluation of ``ctx``
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "la_glide_lut.json.gz"

# The v-writing treebanks are the evidence; ITTB (u only) is what the tables are FOR.
SOURCES = [
    ROOT / "assets_la2/SUD_Latin-PROIEL/la_proiel-sud-train.conllu",
    ROOT / "assets_la/SUD_Latin-Perseus/la_perseus-sud-train.conllu",
]
TARGET = ROOT / "assets_la/SUD_Latin-ITTB/la_ittb-sud-train.conllu"

MIN_CTX = 20      # contexts rarer than this are not evidence
# Plain majority, NOT this project's usual 0.90 dominance bar. Raising it to 0.90 drops held-out
# accuracy 98.0 -> 94.8, because word-initial ``u`` + ``e`` sits at 0.89 -- glide in every real
# word (uerbum, uerum, uester) and non-glide only in the v-writing treebanks' own spelling slips.
# The bar is right when a rule COMMITS an annotation; here it only picks which of two attested
# spellings to show the model, so majority evidence is the appropriate standard.
MIN_DOMINANCE = 0.50


def strip_macrons(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in "̄̆"))


def key_form(form: str) -> str:
    return strip_macrons(form).lower().replace("v", "u")


def read_forms(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if not f[0].isdigit():
                continue
            out.append(f[1])
    return out


def harvest(paths) -> dict[str, tuple[frozenset, int]]:
    """form -> (majority glide-position set, token count)."""
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing source treebank: {p}")
        for w in read_forms(p):
            lw = strip_macrons(w).lower()
            votes[lw.replace("v", "u")][frozenset(i for i, c in enumerate(lw) if c == "v")] += 1
    return {k: (c.most_common(1)[0][0], sum(c.values())) for k, c in votes.items()}


def context(word: str, i: int) -> str:
    """Context of the ``u`` at ``i``: (prev-prev if prev is u else '.', prev, next).

    The prev-prev slot only matters through a preceding ``u`` (``uua`` -> *uva*: the first is the
    glide, the second is not), so it is collapsed to '.' elsewhere to keep the table small.
    """
    prv = word[i - 1] if i > 0 else "^"
    nxt = word[i + 1] if i + 1 < len(word) else "$"
    prv2 = (word[i - 2] if i > 1 else "^") if prv == "u" else "."
    return f"{prv2}{prv}{nxt}"


def ctx_table(types: dict, keys) -> dict[str, int]:
    stat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for k in keys:
        mask, n = types[k]
        for i, c in enumerate(k):
            if c != "u":
                continue
            s = stat[context(k, i)]
            s[1] += n
            if i in mask:
                s[0] += n
    return {c: int(g / t > 0.5) for c, (g, t) in stat.items()
            if t >= MIN_CTX and max(g / t, 1 - g / t) >= MIN_DOMINANCE}


def report(types: dict) -> None:
    """Held-out evaluation of the context rule, split by TYPE so no word is in both halves."""
    rng = random.Random(0)
    keys = sorted(types)
    rng.shuffle(keys)
    cut = int(len(keys) * 0.8)
    tbl = ctx_table(types, keys[:cut])
    ok = tot = fired = 0
    errs = collections.Counter()
    for k in keys[cut:]:
        mask, n = types[k]
        for i, c in enumerate(k):
            if c != "u":
                continue
            pred = bool(tbl.get(context(k, i), 0))
            fired += n if context(k, i) in tbl else 0
            tot += n
            if pred == (i in mask):
                ok += n
            else:
                errs[(k, i, i in mask)] += n
    print(f"  held-out u positions (token-weighted): {tot}")
    print(f"  covered by a kept context            : {fired / tot:.4f}")
    print(f"  accuracy                             : {ok / tot:.4f}")
    print("  top disagreements (word, position, gold-is-glide):")
    for (w, i, g), n in errs.most_common(10):
        print(f"    {n:6d}  {w:<18s} pos {i}  gold_glide={g}")


def coverage(types: dict) -> None:
    """How much of the u-writing treebank the harvest answers for outright."""
    if not TARGET.exists():
        return
    counts = collections.Counter(strip_macrons(w).lower() for w in read_forms(TARGET))
    u_tokens = {w: n for w, n in counts.items() if "u" in w}
    tot = sum(u_tokens.values())
    cov = sum(n for w, n in u_tokens.items() if w in types)
    glide = sum(n for w, n in u_tokens.items() if w in types and types[w][0])
    print(f"  ITTB tokens containing u : {tot}")
    print(f"  answered by the harvest  : {cov / tot:.4f}")
    print(f"  ... of those, glide-bearing: {glide / cov:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--report", action="store_true", help="held-out evaluation of the context rule")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    types = harvest(SOURCES)
    print(f"harvested {len(types)} types from {len(SOURCES)} v-writing treebanks")
    coverage(types)
    if args.report:
        print("context-rule fallback, held out by type (80/20):")
        report(types)

    lex = {k: "".join("1" if i in mask else "0"
                      for i, c in enumerate(k) if c == "u")
           for k, (mask, _) in types.items() if "u" in k}
    data = {"lex": lex, "ctx": ctx_table(types, types.keys())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    print(f"wrote {args.out}  ({len(lex)} forms, {len(data['ctx'])} contexts, "
          f"{args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
