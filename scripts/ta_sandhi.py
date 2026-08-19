#!/usr/bin/env python3
"""Tamil orthography: the akṣara arithmetic that turns MWT splitting into plain segmentation.

THE PROBLEM. SUD_Tamil-TTB splits 835 orthographic words into 1 781 syntactic words, and **94.2 %
of those splits REWRITE at the seam** rather than cutting cleanly:

    நிலையத்துக்குக்கான   ->  நிலையத்துக்குக்க்  +  ஆன
    துறைகளையும்          ->  துறைகளைய்         +  உம்
    வந்துள்ளதாக           ->  வந்த்  +  உள்ளத்  +  ஆக

So the concatenation of the parts is NOT the surface, and every tool in this project that assumes
it is — `make_seg_pairs.py` drops such rows by design, `sud.CharSegTokenizer.v1` can only cut —
sees a treebank whose FORMs are not a segmentation of its own text.

THE OBSERVATION THAT DISSOLVES IT. Tamil is an abugida: the character கா is not "k" followed by
"ā", it is one akṣara spelling க் + ஆ. The treebank's split points fall INSIDE such characters, and
that is all that is happening. Decompose every akṣara into consonant + virāma + INDEPENDENT vowel
and the rewriting disappears:

    decompose(நிலையத்துக்குக்கான) == decompose(நிலையத்துக்குக்க்) + decompose(ஆன)

Measured over both treebanks: `recompose(decompose(w)) == w` on **13 043 of 13 043 tokens**, and the
gold parts are a clean segmentation of the decomposed surface on **842 of 878 MWT ranges (95.90 %)**
against 5.8 % on the raw surface. So the tokeniser becomes: decompose, cut, recompose — and the
existing trained character segmenter does the middle step unchanged.

THE 4.1 % RESIDUE IS REAL SANDHI, and it is a short named list rather than a long tail:

  * **Gemination (வலிமிகல்).** A hard consonant doubles at the seam: `கஷ்ட` + `படுகிறான்` ->
    `கஷ்டப்படுகிறான்`, `பணத்துக்கு` + `தான்` -> `பணத்துக்குத்தான்`.
  * **u-elision.** A final உ drops before an initial vowel: `கொண்டு` + `இருக்கிறது` ->
    `கொண்டிருக்கிறது`.
  * **ல்/ற் assimilation.** `அடுத்தால்` + `போல` -> `அடுத்தாற்போல`.
  * **Suppletion**, which is not orthographic at all: the treebank writes `*இந்த` (asterisked,
    marking an abstract form) for the `இவ்` of `இவ்விரண்டு`. No rule reaches that and none should
    try — it is a lemma, not a spelling.

⚠ THE DECOMPOSED FORM IS A TRAINING AND INFERENCE REPRESENTATION, NEVER A STORED ONE. The parser
corpora keep the treebank's real FORMs; only the segmenter's training pairs are decomposed, and the
tokeniser decomposes its input and recomposes its output. `scripts/ta_tokenizer.py` records the
regime in the bundled `vocab.json` (`reads_decomposed`) and reads it back, because standing hazard
10 in CLAUDE.md is exactly this mistake made twice already — a CSLiser trained on spaced text fed
space-split chunks for a whole generation, at −4.83 F.

    ta_sandhi.py --check                                   # round trip + segmentation rates
    ta_sandhi.py --conllu IN.conllu --out OUT.conllu        # a decomposed copy, for the segmenter
"""
from __future__ import annotations

import argparse
import pathlib

VIRAMA = "்"                      # ் puḷḷi

#: dependent vowel sign -> independent vowel. `அ` has no sign: it is the inherent vowel.
SIGN_TO_INDEPENDENT = {
    "ா": "ஆ", "ி": "இ", "ீ": "ஈ",   # ā i ī
    "ு": "உ", "ூ": "ஊ", "ெ": "எ",   # u ū e
    "ே": "ஏ", "ை": "ஐ", "ொ": "ஒ",   # ē ai o
    "ோ": "ஓ", "ௌ": "ஔ",                       # ō au
}
INDEPENDENT_TO_SIGN = {v: k for k, v in SIGN_TO_INDEPENDENT.items()}
INHERENT = "அ"                    # அ, the inherent vowel
INHERENT_U = "உ"                  # உ, the vowel that elides before another vowel
INDEPENDENT_TO_SIGN[INHERENT] = ""

#: Tamil consonant letters, U+0B95..U+0BB9. Each already spells consonant + inherent அ.
CONSONANTS = frozenset(chr(c) for c in range(0x0B95, 0x0BBA))
INDEPENDENT_VOWELS = frozenset(INDEPENDENT_TO_SIGN)

#: The hard consonants that double at a morpheme seam (வலிமிகல்).
HARD = ("க", "ச", "ட", "த", "ப", "ற")   # k c ṭ t p ṟ


def decompose(text: str) -> str:
    """Every akṣara as consonant + virāma + independent vowel. Exactly invertible by `recompose`."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in CONSONANTS:
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:                       # bare consonant, already decomposed
                out.append(ch)
                out.append(VIRAMA)
                i += 2
            elif nxt in SIGN_TO_INDEPENDENT:        # consonant + vowel sign
                out.append(ch)
                out.append(VIRAMA)
                out.append(SIGN_TO_INDEPENDENT[nxt])
                i += 2
            else:                                   # inherent அ
                out.append(ch)
                out.append(VIRAMA)
                out.append(INHERENT)
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def recompose(text: str) -> str:
    """Inverse of `decompose`. Verified exact on every token of both Tamil treebanks."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] in CONSONANTS and text[i + 1] == VIRAMA:
            nxt = text[i + 2] if i + 2 < n else ""
            if nxt in INDEPENDENT_TO_SIGN:
                out.append(text[i] + INDEPENDENT_TO_SIGN[nxt])
                i += 3
            else:
                out.append(text[i] + VIRAMA)
                i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _seam(left: str, right: str, geminate: bool) -> str:
    """Join two DECOMPOSED strings at one morpheme seam.

    Both rules are stated in decomposed space, which is the only space they are simple in. Written
    on the composed string, u-elision removes the vowel SIGN and silently leaves the consonant with
    its inherent அ — `கொண்டு` + `இருக்கிறது` came out `கொண்டஇருக்கிறது` instead of
    `கொண்டிருக்கிறது`. That was this module's first version and it recovered nothing.
    """
    if not left:
        return right
    if not right:
        return left
    # u-elision: a final உ drops before an initial vowel.
    if left.endswith(INHERENT_U) and right[0] in INDEPENDENT_VOWELS:
        return left[:-1] + right
    # gemination (வலிமிகல்): a hard consonant doubles after a vowel-final word.
    if geminate and right[0] in HARD and left and left[-1] in INDEPENDENT_VOWELS:
        return left + right[0] + VIRAMA + right
    return left + right


def join_variants(parts: list[str]) -> list[str]:
    """Every surface these parts could re-form, plain first.

    A LIST rather than a value, and that is the honest shape. Gemination is not predictable from
    the string: Tamil grammar calls it வலி மிகும் / மிகாது and it turns on the morphology and the
    lexeme, not on the phonology alone — `கஷ்ட` + `படுகிறான்` geminates, and plenty of the 842
    ranges that join cleanly have a hard-initial second part and do not. So the deterministic
    direction gives the plain join and the alternatives are offered, never chosen here.
    """
    out = []
    for geminate in (False, True):
        acc = decompose(parts[0]) if parts else ""
        for nxt in parts[1:]:
            acc = _seam(acc, decompose(nxt), geminate)
        surface = recompose(acc)
        if surface not in out:
            out.append(surface)
    return out


def join(parts: list[str]) -> str:
    """The deterministic join: plain concatenation plus u-elision, no gemination."""
    return join_variants(parts)[0] if parts else ""


# ---------------------------------------------------------------------------- CLI


def _iter_conllu(path):
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        yield line


def write_decomposed(src: str, dst: str) -> None:
    """A copy with FORM decomposed. Every other column, and every comment, is untouched.

    Range rows are decomposed too: `make_seg_pairs.py` reads their SPACING, and a later reader that
    wanted their FORM should see the same representation as the words under them.
    """
    out, n = [], 0
    for line in _iter_conllu(src):
        cols = line.split("\t")
        if len(cols) == 10 and (cols[0].isdigit() or "-" in cols[0]):
            cols[1] = decompose(cols[1])
            n += 1
            line = "\t".join(cols)
        out.append(line)
    pathlib.Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {dst}  ({n} FORMs decomposed)")


def check(paths: list[str]) -> None:
    import glob
    files = [f for p in paths for f in sorted(glob.glob(p))]
    n = bad = 0
    for path in files:
        for line in _iter_conllu(path):
            cols = line.split("\t")
            if len(cols) == 10 and (cols[0].isdigit() or "-" in cols[0]):
                n += 1
                bad += recompose(decompose(cols[1])) != cols[1]
    print(f"round trip exact on {n - bad}/{n} tokens ({(n - bad) / max(n, 1):.4%})")

    ok = fail = 0
    residue = []
    for path in files:
        lines = list(_iter_conllu(path))
        i = 0
        while i < len(lines):
            cols = lines[i].split("\t")
            if len(cols) == 10 and "-" in cols[0] and cols[0][0].isdigit():
                end = int(cols[0].split("-")[1])
                surface, parts, j = cols[1], [], i + 1
                while j < len(lines):
                    d = lines[j].split("\t")
                    if len(d) == 10 and d[0].isdigit() and int(d[0]) <= end:
                        parts.append(d[1])
                        j += 1
                    else:
                        break
                if "".join(decompose(p) for p in parts) == decompose(surface):
                    ok += 1
                else:
                    fail += 1
                    variants = join_variants(parts)
                    if surface in variants:
                        residue.append((f"rule recovers ({'plain' if surface == variants[0] else 'geminate'})",
                                        surface, parts))
                    else:
                        residue.append(("UNRECOVERED", surface, parts))
                i = j
                continue
            i += 1
    print(f"parts are a clean segmentation of the DECOMPOSED surface: "
          f"{ok}/{ok + fail} = {ok / max(ok + fail, 1):.2%}")
    got = sum(1 for kind, *_ in residue if kind.startswith("rule recovers"))
    print(f"of the {fail} that are not, the join rules recover {got}; "
          f"{fail - got} remain (suppletion and assimilation)")
    for kind, surface, parts in residue[:8]:
        print(f"   {kind:14s} {surface}  ->  {' + '.join(parts)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu")
    ap.add_argument("--out")
    ap.add_argument("--check", nargs="*", default=None,
                    help="glob(s) of CoNLL-U to verify; defaults to both Tamil treebanks")
    args = ap.parse_args()
    if args.check is not None:
        check(args.check or ["assets_ta/SUD_Tamil-*/*.conllu"])
    if args.conllu:
        write_decomposed(args.conllu, args.out or args.conllu.replace(".conllu", ".decomp.conllu"))


if __name__ == "__main__":
    main()
