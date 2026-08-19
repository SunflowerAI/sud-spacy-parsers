#!/usr/bin/env python3
"""Strict token F of the Tamil tokeniser on raw `# text`, against the rule tokeniser it replaces.

THE ONLY NUMBER THAT SEES A TOKENISER. Every parsing figure in this project passes
`--gold-preproc`, which hands the model gold tokens and never calls the tokeniser at all, and
`sud.GoldTokCorpus.v1` trains the parser on gold tokens too — so LAS, UAS and TAG are all blind to
whether the wheel can segment Tamil. `docs/lzh-tokenisation.md` records the same blind spot costing
lzh a tokeniser that split 孔子 for a whole generation.

The baseline is spaCy's rule tokeniser, which splits on whitespace and punctuation only. Its
ceiling is arithmetic rather than empirical: the treebank has 13 043 syntactic words in 11 171
orthographic ones, so a tokeniser that never splits a word can reach at most ~0.92 F however good
it otherwise is.

⚠ SCORED WITH `difflib.SequenceMatcher` over the token SEQUENCE, which is spaCy's own strict
token-accuracy notion (`eval_la_tokenizer.py` uses the same helper) — NOT with character offsets.
Offsets are unavailable here on purpose: this tokeniser REWRITES, so `கஷ்டப்படுகிறான்` comes back as
`கஷ்ட` + `படுகிறான்` and the pieces do not have spans in the input string. A sequence match is the
right notion for that, and it is the same one the id/zh/lzh segmenters were scored with.

    eval_ta_tokenizer.py --model models/ta_seg_char --conllu assets_ta/ta_ttb_mwtt-sud-test.conllu
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                            # noqa: E402
from ta_tokenizer import TamilSandhiTokenizer           # noqa: E402


def sentences(path):
    """Yield (raw text, [gold syntactic word, ...]) per sentence.

    The raw text is the treebank's own `# text`, i.e. what a user would actually type — NOT the
    concatenation of the FORMs, which for Tamil is a different string (that is the whole problem).
    """
    text, words = None, []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# text ="):
            text = line.split("=", 1)[1].strip()
            continue
        if not line.strip():
            if text and words:
                yield text, words
            text, words = None, []
            continue
        cols = line.split("\t")
        if len(cols) == 10 and cols[0].isdigit():
            words.append(cols[1])
    if text and words:
        yield text, words


def matched(gold, pred) -> int:
    """Tokens in aligned equal blocks — spaCy's own strict token-accuracy notion."""
    sm = difflib.SequenceMatcher(a=gold, b=pred, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def f1(correct, gold, pred) -> float:
    p = correct / pred if pred else 0.0
    r = correct / gold if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ta_seg_char")
    ap.add_argument("--conllu", default="assets_ta/ta_ttb_mwtt-sud-test.conllu")
    args = ap.parse_args()

    data = list(sentences(args.conllu))
    if not data:
        print(f"no `# text` sentences in {args.conllu}")
        return 1

    rule = spacy.blank("ta")
    trained = spacy.blank("ta")
    tok = TamilSandhiTokenizer(trained.vocab)
    tok.load_segmenter(args.model)
    trained.tokenizer = tok

    rows = []
    for name, nlp in (("rule (baseline)", rule), ("trained segmenter", trained)):
        g = p = c = 0
        for text, gold in data:
            pred = [t.text for t in nlp(text)]
            g += len(gold)
            p += len(pred)
            c += matched(gold, pred)
        rows.append((name, c, g, p, f1(c, g, p)))

    print(f"{args.conllu}: {len(data)} sentences, {rows[0][2]} gold syntactic words")
    print(f"{'tokenizer':22s} {'correct':>8s} {'pred':>7s} {'P':>7s} {'R':>7s} {'F':>7s}")
    for name, c, g, p, f in rows:
        print(f"{name:22s} {c:8d} {p:7d} {c/p if p else 0:7.4f} {c/g if g else 0:7.4f} {f:7.4f}")

    gain = rows[1][4] - rows[0][4]
    print(f"\nthe split is worth {gain:+.4f} F over the rule tokeniser "
          f"({rows[0][4]:.4f} -> {rows[1][4]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
