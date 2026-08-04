#!/usr/bin/env python3
"""Strict token F of the Latin tokeniser on raw `# text`, with and without the -que rule.

Only ITTB and Perseus can measure this: PROIEL respaces its own `# text` (`ne que mittatis`),
so its enclitics are already separated and the fused spelling never appears.

`--gold-preproc` hides this failure completely -- every published la metric bypasses the
tokeniser -- so this script is the only place the enclitic split is visible.

    .venv/bin/python scripts/eval_la_tokenizer.py [--split test]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spacy                                            # noqa: E402
import la_tokenizer                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CORPORA = [
    ("ITTB", "assets_la/SUD_Latin-ITTB/la_ittb-sud-%s.conllu"),
    ("Perseus", "assets_la/SUD_Latin-Perseus/la_perseus-sud-%s.conllu"),
]


def sentences(path: Path):
    """(raw text, gold SYNTACTIC tokens) -- range lines skipped, sub-tokens kept."""
    text, toks = None, []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.startswith("# text ="):
            text = line.split("=", 1)[1].strip()
            continue
        if not line.strip():
            if text and toks:
                yield text, toks
            text, toks = None, []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if not re.match(r"^\d+$", cols[0]):
            continue
        toks.append(cols[1])
    if text and toks:
        yield text, toks


def matched(gold, pred) -> int:
    """Tokens in aligned equal blocks -- spaCy's own strict token-accuracy notion."""
    sm = difflib.SequenceMatcher(a=gold, b=pred, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def f1(correct, gold, pred) -> float:
    p = correct / pred if pred else 0.0
    r = correct / gold if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    plain = spacy.blank("la")
    fixed = spacy.blank("la")
    fixed.tokenizer = la_tokenizer.make_la_enclitic_tokenizer()(fixed)

    print(f"{'corpus':10s} {'gold':>7s}  {'stock la':>9s}  {'+ -que rule':>12s}  {'Δ':>7s}")
    for name, template in CORPORA:
        path = ROOT / (template % args.split)
        if not path.exists():
            continue
        gold_n = a_pred = a_corr = b_pred = b_corr = 0
        for text, gold in sentences(path):
            a = [t.text for t in plain(text) if not t.is_space]
            b = [t.text for t in fixed(text) if not t.is_space]
            gold_n += len(gold)
            a_pred += len(a)
            a_corr += matched(gold, a)
            b_pred += len(b)
            b_corr += matched(gold, b)
        before, after = f1(a_corr, gold_n, a_pred), f1(b_corr, gold_n, b_pred)
        print(f"{name:10s} {gold_n:7d}  {before:9.4f}  {after:12.4f}  {after - before:+7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
