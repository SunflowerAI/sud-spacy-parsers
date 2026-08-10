#!/usr/bin/env python
"""Harvest, from ITTB, the tables that render an ITTB-conformant XPOS for any Latin token.

The three Latin treebanks the parser trains on use mutually-incompatible XPOS tagsets:
ITTB writes the Index Thomisticus composite code (`C1|grn1|casB|gen2|vgr1`, 1 914 types
over 390 787 train tokens), PROIEL a 23-value part-of-speech code (`Nb`, `V-`, `Df`), and
Perseus a 9-position morphology string (`v3spsa---`).  ITTB is by far the largest, so its
conventions are the ones to normalise towards -- and it is left untouched; only PROIEL and
Perseus are re-rendered (see normalise_la_xpos.py).

The composite tag decomposes into a LEXICAL head and a MORPHOLOGICAL tail, and the split is
what makes the rendering derivable at all:

    C1 | grn1 casB gen2 | vgr1
    ^^   ^^^^^^^^^^^^^^   ^^^^
    |    restatement of FEATS   graphical variant of the surface form
    flexional category: LETTER = declension/conjugation (lexical, a property of the lemma)
                        DIGIT  = which paradigm the form inflects by (morphological)

Measured on ITTB train, that reading holds: the letter is dominant at 98.43 % per
(lemma, UPOS), and the digit at 98.15 % per (UPOS, VerbForm, has-Case) -- a verb lemma takes
`3` when finite or infinitive and `2` when participial, which is why keying the whole head on
the lemma alone tops out at 94.5 % and splitting it reaches 95.7-96.1 %.

Four tables, each with a backoff ladder:

  TAIL    (UPOS, FEATS-signature) -> tail string, at six levels of feature restriction.  The
          tail is taken VERBATIM from ITTB rather than composed field by field, so a covered
          token is guaranteed a well-formed tag in ITTB's own field order.
  LETTER  (lemma, UPOS) -> lemma -> (UPOS, lemma suffix) -> (UPOS) majority.  The suffix rung
          is the one that matters for the other two treebanks: 17.8 % of PROIEL and 23.4 % of
          Perseus tokens have a lemma ITTB never saw, and on ITTB's own unseen-lemma tokens a
          UPOS majority gets the letter right 41 % / 45 % of the time (dev / test) against the
          suffix model's 83 % / 90 %.  Declension IS a property of the stem ending, so this is
          the feature the majority baseline was missing, not a lucky correlation.
  DIGIT   (UPOS, VerbForm, has-Case) -> (UPOS).
  VGR     (folded form, UPOS) -> none.  The graphical-variant marker is a fact about the
          surface spelling, so it is keyed on the form and simply omitted when unattested,
          which is what an unmarked (standard) spelling carries anyway.  Keying it on the form
          rather than the lemma is worth 4.5 points of exact agreement (89.2 -> 93.7 on test).
          The key is FOLDED -- length marks stripped, ligatures expanded, j/v -> i/u -- because
          the released arm is the orthographically augmented one, which respells every token
          each epoch: a tag is a property of the word, not of the edition it is printed in, so
          `vitae`, `vītae` and `vītæ` must render one XPOS.  Folding to the i/u spelling is the
          right direction because ITTB, the treebank being harvested, writes `u` throughout.

Writing the map also self-evaluates it: --report re-derives ITTB's own dev/test XPOS from
(lemma, UPOS, FEATS) alone and prints exact/head/tail agreement, which is the honest ceiling
on the tags this map manufactures for PROIEL and Perseus.

    build_la_xpos_map.py [--out la_xpos_map.json] [--report]
"""
import argparse, collections, json, re, unicodedata

FIELD_FAMILY = re.compile(r"^[a-z]+")
HEAD_SPLIT = re.compile(r"^([A-Za-z]+)(\d*)$")

# Six restrictions of the FEATS bundle, most specific first.  ITTB's tail restates morphology
# only, so keys outside these sets (InflClass, Compound, Shared, NameType ...) are noise for
# this purpose and are dropped at every level.
LEVELS = [
    {"Case", "Number", "Gender", "Degree", "Mood", "Tense", "Aspect", "Voice", "VerbForm",
     "Person", "PronType", "NumType", "Poss", "Polarity"},
    {"Case", "Number", "Gender", "Degree", "Mood", "Tense", "Aspect", "Voice", "VerbForm",
     "Person"},
    {"Case", "Number", "Gender", "Mood", "Tense", "VerbForm", "Person"},
    {"Case", "Number", "Mood", "Tense", "VerbForm"},
    {"Case", "Number", "VerbForm"},
    set(),
]
SUFFIX_LENGTHS = (5, 4, 3, 2, 1)
SUFFIX_MIN_COUNT = 10

_LENGTH_MARKS = ("\u0304", "\u0306")          # combining macron, combining breve
_UNLIGATURE = {"\u00e6": "ae", "\u0153": "oe", "\u00c6": "ae", "\u0152": "oe"}


def fold(form):
    """Lowercase and strip every orthographic axis the augmenter varies (see VGR above)."""
    s = unicodedata.normalize("NFD", form.lower())
    s = "".join(c for c in s if c not in _LENGTH_MARKS)
    s = unicodedata.normalize("NFC", s)
    s = "".join(_UNLIGATURE.get(c, c) for c in s)
    return s.replace("j", "i").replace("v", "u")


def rows(path):
    """Yield token rows (10 CoNLL-U columns); range/empty-node lines are skipped."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            cols = line.rstrip("\n").split("\t")
            if cols[0].isdigit():
                yield cols


def family(field):
    m = FIELD_FAMILY.match(field)
    return m.group(0) if m else field


def split_head(head):
    m = HEAD_SPLIT.match(head)
    return m.groups() if m else (head, "")


def feats_dict(feats):
    return dict(x.split("=", 1) for x in feats.split("|") if "=" in x) if feats != "_" else {}


def signature(feats, keep):
    d = feats_dict(feats)
    return "|".join(f"{k}={d[k]}" for k in sorted(d) if k in keep)


def digit_key(upos, feats):
    d = feats_dict(feats)
    return f"{upos}\t{d.get('VerbForm', '')}\t{int('Case' in d)}"


def build(ittb_train):
    tails = [collections.defaultdict(collections.Counter) for _ in LEVELS]
    letter_lu = collections.defaultdict(collections.Counter)
    letter_l = collections.defaultdict(collections.Counter)
    letter_suf = {n: collections.defaultdict(collections.Counter) for n in SUFFIX_LENGTHS}
    letter_u = collections.defaultdict(collections.Counter)
    digit = collections.defaultdict(collections.Counter)
    vgr = collections.defaultdict(collections.Counter)

    for c in rows(ittb_train):
        _, _, lemma, upos, xpos, feats = c[0], c[1], c[2].lower(), c[3], c[4], c[5]
        if xpos == "_":
            continue
        parts = xpos.split("|")
        letter, dig = split_head(parts[0])
        variants = [p for p in parts[1:] if family(p) == "vgr"]
        tail = "|".join(p for p in parts[1:] if family(p) != "vgr")

        for i, keep in enumerate(LEVELS):
            tails[i][f"{upos}\t{signature(feats, keep)}"][tail] += 1
        letter_lu[f"{lemma}\t{upos}"][letter] += 1
        letter_l[lemma][letter] += 1
        for n in SUFFIX_LENGTHS:
            letter_suf[n][f"{upos}\t{lemma[-n:]}"][letter] += 1
        letter_u[upos][letter] += 1
        digit[digit_key(upos, feats)][dig] += 1
        vgr[f"{fold(c[1])}\t{upos}"][variants[0] if variants else ""] += 1

    top = lambda d: {k: ctr.most_common(1)[0][0] for k, ctr in d.items()}
    return {
        "tails": [top(t) for t in tails],
        "letter_lemma_upos": top(letter_lu),
        "letter_lemma": top(letter_l),
        "letter_suffix": {str(n): {k: ctr.most_common(1)[0][0]
                                   for k, ctr in letter_suf[n].items()
                                   if sum(ctr.values()) >= SUFFIX_MIN_COUNT}
                          for n in SUFFIX_LENGTHS},
        "letter_upos": top(letter_u),
        "digit": top(digit),
        "vgr": {k: v for k, v in top(vgr).items() if v},
    }


class Renderer:
    """Render an ITTB-conformant XPOS from (form, lemma, UPOS, FEATS)."""

    def __init__(self, m):
        self.m = m

    def letter(self, lemma, upos):
        m = self.m
        for key, table in ((f"{lemma}\t{upos}", m["letter_lemma_upos"]),
                           (lemma, m["letter_lemma"])):
            if key in table:
                return table[key], "lexicon"
        for n in SUFFIX_LENGTHS:
            hit = m["letter_suffix"][str(n)].get(f"{upos}\t{lemma[-n:]}")
            if hit:
                return hit, "suffix"
        return m["letter_upos"].get(upos, "O"), "upos"

    def render(self, form, lemma, upos, feats):
        m = self.m
        lemma = lemma.lower()
        tail, tail_src = "", "none"
        for i, keep in enumerate(LEVELS):
            hit = m["tails"][i].get(f"{upos}\t{signature(feats, keep)}")
            if hit is not None:
                tail, tail_src = hit, f"L{i}"
                break
        letter, letter_src = self.letter(lemma, upos)
        dk = digit_key(upos, feats)
        dig = m["digit"].get(dk, m["digit"].get(f"{upos}\t\t0", ""))
        variant = m["vgr"].get(f"{fold(form)}\t{upos}", "")
        parts = [letter + dig] + ([tail] if tail else []) + ([variant] if variant else [])
        return "|".join(parts), tail_src, letter_src


def report(m, ittb_tmpl):
    r = Renderer(m)
    train_lemmas = {c[2].lower() for c in rows(ittb_tmpl % "train")}
    for split in ("dev", "test"):
        n = ok = ok_head = ok_tail = 0
        oov_n = oov_ok = 0
        srcs = collections.Counter()
        for c in rows(ittb_tmpl % split):
            if c[4] == "_":
                continue
            n += 1
            pred, ts, ls = r.render(c[1], c[2], c[3], c[5])
            srcs[(ts, ls)] += 1
            ok += pred == c[4]
            ok_head += pred.split("|")[0] == c[4].split("|")[0]
            strip = lambda x: "|".join(p for p in x.split("|")[1:] if family(p) != "vgr")
            ok_tail += strip(pred) == strip(c[4])
            if c[2].lower() not in train_lemmas:
                oov_n += 1
                oov_ok += split_head(pred.split("|")[0])[0] == split_head(c[4].split("|")[0])[0]
        print(f"ITTB {split}: n={n}  exact {100*ok/n:.2f}%  head {100*ok_head/n:.2f}%  "
              f"tail {100*ok_tail/n:.2f}%   unseen-lemma letter {100*oov_ok/max(oov_n,1):.2f}% "
              f"(n={oov_n})")
        print("   sources:", ", ".join(f"{k}={v}" for k, v in srcs.most_common(5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ittb", default="assets_la/SUD_Latin-ITTB/la_ittb-sud-%s.conllu")
    ap.add_argument("--out", default="assets_la/la_xpos_map.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    m = build(a.ittb % "train")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False)
    print(f"{a.out}: tails={[len(t) for t in m['tails']]} "
          f"letter(lemma,upos)={len(m['letter_lemma_upos'])} "
          f"letter(suffix)={sum(len(v) for v in m['letter_suffix'].values())} "
          f"digit={len(m['digit'])} vgr={len(m['vgr'])}")
    if a.report:
        report(m, a.ittb)


if __name__ == "__main__":
    main()
