#!/usr/bin/env python3
"""Rebuild the Sanskrit training representation on DCS's multiword-token convention.

WHAT CHANGED AND WHY. The previous representation (`revert_csl_sandhi.py` -> `*.csl_rev.conllu`)
ran `desandhi_csl` over EVERY token, normalising all of them toward the pre-pausal (pausa) form.
That is an approximation twice over: a word in mid-sentence is not at a pause, and the pausa form is
not what any Sanskrit treebank records. This follows the Digital Corpus of Sanskrit instead, whose
CoNLL-U readme defines the unit exactly: "A single string (= sequence of letters bounded by spaces)
can contain one or multiple words in Sanskrit. If it contains multiple words, the annotation follows
the guidelines for multiword annotation." So an **MWT is an ORTHOGRAPHIC word**, not a compound.

Measured on DCS (Rāmāyaṇa, 162 sentences / 1 080 tokens / 182 ranges), the convention is:

  * **MWT formation** — an orthographic word fuses on a **bound** junction (compound member,
    preverb, privative) or on **vowel coalescence** (`caitat` = `ca` + `etat`, `cāmantrya` =
    `ca` + `āmantrya`). It does NOT fuse at an avagraha (`ko 'nasūyakaḥ`, `samartho 'si` all keep
    the space) and does NOT fuse consonant-final + vowel-initial (`vahnir idraḥ` stays two words —
    only 2 such junctions in the DCS file, both the bound privative `an-`). That last point is
    where DCS parts company with the Devanagari-script treebanks: UFAL writes `वह्निरिद्रः` solid
    because the script cannot render a bare consonant before a vowel, but DCS is romanised and
    keeps the space.
  * **FORM** (`--forms dcs`, the parser corpus) — every token INSIDE an MWT is **unsandhied**,
    final member included (0 of 219 internal and 0 of 182 final DCS tokens carry sandhi); the range
    line already holds the fused surface, so members are free to be citation forms. A token that IS
    its own orthographic word keeps its **sandhied** surface (36.2 % of DCS's differ from unsandhied:
    `nāradaṃ`, `vālmīkir`, `ko`, `nv`), including a leading avagraha (`'nasūyakaḥ`).
    `--forms csl` instead writes the verbatim de-CSLized piece for every token: that is the INPUT
    side of the de-sandhifier, used to build its training corpus.
  * **`Unsandhied=` in MISC** keeps its native treebank values on every token, untouched. It is the
    gold for the FORMs above, and the supervision for a learned sandhi-reversal component.

The `# text` line stays the **CSL** string: it is what the tokeniser reads, and CSL carries the
syntactic word boundaries that let it split without solving segmentation first. The orthographic
rendering is emitted alongside as `# text_ortho` so the representation documents itself.

    restructure_sa_csl.py IN.sandhi.conllu OUT.conllu
"""
import argparse
import collections
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conllu_misc import misc_get, misc_set                        # noqa: E402
from external_sandhi import COALESCE_MARKS, COALESCE_SURFACE      # noqa: E402

MARKERS = ("'", '"')


def blocks(path):
    cm, tk = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cm or tk:
                yield cm, tk
            cm, tk = [], []
        elif line.startswith("#"):
            cm.append(line)
        else:
            tk.append(line.split("\t"))
    if cm or tk:
        yield cm, tk


def lead_mark(word):
    for m in COALESCE_MARKS:
        if word.startswith(m):
            return m
    return None


def orthographic_groups(forms):
    """Group token indices into orthographic words, per the DCS convention.

    Fuses on a bound junction (the left token carries the CSL join marker `-`) or on vowel
    coalescence (left ends in the elision marker `'`/`"` AND right opens with a coalescence mark).
    An avagraha — the RIGHT token opening with `'` while the left carries no marker — does NOT fuse.
    """
    groups, cur = [], [0]
    for i in range(len(forms) - 1):
        L, R = forms[i], forms[i + 1]
        bound = L.endswith("-") and len(L) > 1
        coal = bool(L) and L[-1] in MARKERS and lead_mark(R) is not None
        if bound or coal:
            cur.append(i + 1)
        else:
            groups.append(cur)
            cur = [i + 1]
    groups.append(cur)
    return groups


def group_surface(forms, idxs):
    """Render a group as its orthographic word: resolve coalescence, drop the join markers."""
    out = ""
    for i in idxs:
        w = forms[i]
        if out and out[-1] in MARKERS:
            m = lead_mark(w)
            if m:                                    # fused vowel replaces marker + mark
                out = out[:-1] + COALESCE_SURFACE[m]
                w = w[len(m):]
        out += w[:-1] if (w.endswith("-") and len(w) > 1) else w
    return out


def process(in_path, out_path, forms_mode="dcs"):
    stat = collections.Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for cm, rows in blocks(in_path):
            toks = [c for c in rows if not ("-" in c[0] and c[0].split("-")[0].isdigit())]
            forms = [unicodedata.normalize("NFC", c[1]) for c in toks]
            groups = orthographic_groups(forms)
            new_rows = []
            for g in groups:
                surf = group_surface(forms, g)
                if len(g) > 1:
                    a, b = toks[g[0]][0], toks[g[-1]][0]
                    new_rows.append([f"{a}-{b}", surf, "_", "_", "_", "_", "_", "_", "_", "_"])
                    stat["mwt"] += 1
                    stat["mwt_tokens"] += len(g)
                for n, i in enumerate(g):
                    c = list(toks[i])
                    w = forms[i]
                    piece = w[:-1] if (w.endswith("-") and len(w) > 1) else w
                    gold = misc_get(c[9], "Unsandhied")
                    if forms_mode == "csl" or len(g) == 1:
                        # `csl`: the verbatim de-CSLized piece for every token — the INPUT side of
                        # the de-sandhifier. Also what a standalone token keeps under `dcs`, since
                        # DCS leaves a token that is its own orthographic word sandhied.
                        c[1] = piece
                    else:
                        # `dcs`: a token inside an MWT is written unsandhied (0 of 219 internal and
                        # 0 of 182 final DCS tokens carry sandhi) — the range line already holds the
                        # fused surface, so members are free to be citation forms.
                        c[1] = gold if gold else piece
                    c[9] = misc_set(c[9], "SpaceAfter",
                                    "No" if (len(g) > 1 and n < len(g) - 1) else None)
                    stat["external" if len(g) == 1 else "in_mwt"] += 1
                    if gold is None:
                        stat["nogold"] += 1
                    new_rows.append(c)
            ortho = " ".join(group_surface(forms, g) for g in groups)
            for c in cm:
                out.write(c + "\n")
                if c.startswith("# text ="):
                    out.write("# text_ortho = " + ortho + "\n")
            for c in new_rows:
                out.write("\t".join(c) + "\n")
            out.write("\n")
    print(f"{in_path} -> {out_path}")
    print(f"  orthographic words: {stat['external']} single + {stat['mwt']} multiword "
          f"({stat['mwt_tokens']} tokens inside an MWT)"
          + (f"; {stat['nogold']} tokens without a gold Unsandhied" if stat["nogold"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--forms", choices=("dcs", "csl"), default="dcs",
                    help="dcs: MWT members unsandhied, standalone sandhied (the parser corpus). "
                         "csl: every token is the verbatim de-CSLized piece (the de-sandhifier's "
                         "input side).")
    a = ap.parse_args()
    process(a.inp, a.out, a.forms)


if __name__ == "__main__":
    main()
