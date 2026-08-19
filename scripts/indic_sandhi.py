#!/usr/bin/env python3
"""Akṣara decomposition for Indic abugidas, shared by Tamil and Telugu.

`scripts/ta_sandhi.py` discovered the useful fact for Tamil and this module generalises it: in an
abugida a consonant letter already spells consonant + inherent vowel, so a morpheme boundary can
fall INSIDE a character. Decompose every akṣara into consonant + virāma + INDEPENDENT vowel and the
boundary becomes an ordinary position in a string:

    Tamil    நிலையத்துக்குக்கான  ->  நிலையத்துக்குக்க்  +  ஆன
    Telugu   ఇళ్ళున్నాయి          ->  ఇళ్ళు             +  ఉన్నాయి

Both directions are exact, and that is checked rather than assumed — `recompose(decompose(w)) == w`
on **13 043 / 13 043** Tamil tokens and **6 465 / 6 465** Telugu tokens.

COMBINING MARKS PASS THROUGH UNTOUCHED, which is why the round trip holds for Telugu at all.
Anusvāra (ం), visarga (ః) and candrabindu (ఁ) are neither consonants nor vowel signs, so
`decompose` emits them where it finds them and `recompose` — which only ever merges a
consonant + virāma + independent-vowel triple — steps over them. `డబ్బంతా` therefore decomposes with
the ం sitting between the two words' material and recomposes into `డబ్బు` + `అంతా` correctly.

⚠ **THE ENUNCIATIVE VOWEL IS A PROPERTY OF THE SCRIPT'S LANGUAGE, NOT OF THE SCRIPT.** Telugu adds
a euphonic `-u` to consonant-final words and elides it before a vowel-initial word, which is what
fuses two words into one orthographic word. Tamil's equivalent `-u` behaves differently and Tamil's
splits are mostly plain resyllabification. So `Script.enunciative` is set per language and a caller
that reconstructs a word must use it rather than assuming.
"""
from __future__ import annotations


class Script:
    """Everything the decomposition needs to know about one abugida.

    ⚠ A PLAIN CLASS, DELIBERATELY, WHERE THE REST OF THIS PROJECT USES `@dataclass`. spaCy loads a
    `--code` module BY PATH and does not put it in `sys.modules`, so `dataclasses` cannot look the
    module up to resolve its own (string, under `from __future__ import annotations`) field
    annotations and dies with `AttributeError: 'NoneType' object has no attribute '__dict__'`.
    That surfaces at `spacy package` time, not at training time, so it fails only once a wheel is
    being built — and this module has to load INSIDE the wheel, where the same import path applies.
    """

    def __init__(self, name, virama, inherent, sign_to_independent, consonants, enunciative):
        self.name = name
        self.virama = virama
        self.inherent = inherent             # the vowel a bare consonant letter carries
        self.sign_to_independent = sign_to_independent   # dependent sign -> independent vowel
        self.consonants = consonants
        self.enunciative = enunciative       # the euphonic final vowel that elides, or None

    @property
    def independent_to_sign(self) -> dict:
        out = {v: k for k, v in self.sign_to_independent.items()}
        out[self.inherent] = ""
        return out

    @property
    def independent_vowels(self) -> frozenset:
        return frozenset(self.independent_to_sign)


def _script(name, first_cons, last_cons, virama, inherent, pairs, enunciative, extra_cons=()):
    cons = frozenset(chr(c) for c in range(first_cons, last_cons + 1)) | frozenset(extra_cons)
    return Script(name=name, virama=virama, inherent=inherent,
                  sign_to_independent=dict(pairs), consonants=cons, enunciative=enunciative)


TAMIL = _script(
    "Tamil", 0x0B95, 0x0BB9, "்", "அ",
    [("ா", "ஆ"), ("ி", "இ"), ("ீ", "ஈ"),
     ("ு", "உ"), ("ூ", "ஊ"), ("ெ", "எ"),
     ("ே", "ஏ"), ("ை", "ஐ"), ("ொ", "ஒ"),
     ("ோ", "ஓ"), ("ௌ", "ஔ")],
    enunciative="உ",               # உ
)

TELUGU = _script(
    "Telugu", 0x0C15, 0x0C39, "్", "అ",
    [("ా", "ఆ"), ("ి", "ఇ"), ("ీ", "ఈ"),
     ("ు", "ఉ"), ("ూ", "ఊ"), ("ృ", "ఋ"),
     ("ె", "ఎ"), ("ే", "ఏ"), ("ై", "ఐ"),
     ("ొ", "ఒ"), ("ో", "ఓ"), ("ౌ", "ఔ")],
    enunciative="ఉ",               # ఉ — the euphonic vowel that elides
    extra_cons=("ౘ", "ౙ", "ౚ"),
)

SCRIPTS = {"ta": TAMIL, "te": TELUGU}


def decompose(text: str, script: Script) -> str:
    """Every akṣara as consonant + virāma + independent vowel. Exactly invertible by `recompose`."""
    signs = script.sign_to_independent
    cons = script.consonants
    virama = script.virama
    inherent = script.inherent
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in cons:
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == virama:                       # bare consonant, already decomposed
                out.append(ch)
                out.append(virama)
                i += 2
            elif nxt in signs:                      # consonant + vowel sign
                out.append(ch)
                out.append(virama)
                out.append(signs[nxt])
                i += 2
            else:                                   # inherent vowel
                out.append(ch)
                out.append(virama)
                out.append(inherent)
                i += 1
        else:
            out.append(ch)                          # anusvāra, visarga, punctuation, digits, Latin
            i += 1
    return "".join(out)


def recompose(text: str, script: Script) -> str:
    """Inverse of `decompose`. Verified exact on every token of both treebank families."""
    to_sign = script.independent_to_sign
    cons = script.consonants
    virama = script.virama
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] in cons and text[i + 1] == virama:
            nxt = text[i + 2] if i + 2 < n else ""
            if nxt in to_sign:
                out.append(text[i] + to_sign[nxt])
                i += 3
            else:
                out.append(text[i] + virama)
                i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def restore_enunciative(decomposed_left: str, script: Script) -> str:
    """Re-form the FIRST word of a fused pair, putting back the vowel that elided.

    A word of these languages does not end in a bare consonant — that is precisely why the
    enunciative vowel exists — so a left part ending in a virāma is the signature of elision, and
    the reconstruction is to put the vowel back.
    """
    if script.enunciative is None:
        return recompose(decomposed_left, script)
    return recompose(decomposed_left + script.enunciative, script)


def joins_to(parts: list[str], script: Script) -> str:
    """Re-form the orthographic word from its syntactic words under enunciative elision.

    The deterministic direction, and the one that VERIFIES a split: a proposed split is only
    accepted if joining its parts reproduces the original surface exactly.
    """
    if not parts:
        return ""
    acc = decompose(parts[0], script)
    for nxt in parts[1:]:
        right = decompose(nxt, script)
        if (script.enunciative and acc.endswith(script.enunciative)
                and right[:1] and right[0] in script.independent_vowels):
            acc = acc[:-1]                          # the enunciative vowel elides before a vowel
        acc += right
    return recompose(acc, script)
