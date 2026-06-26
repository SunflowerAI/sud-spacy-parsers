#!/usr/bin/env python
"""Prepare a Sanskrit SUD treebank for the `sa` pipeline in Clay-Sanskrit-Library (CSL)
notation.

Two transforms (see CLAUDE.md / conversation):

  1. **Transliterate** every Devanagari cell (FORM, LEMMA) to IAST, using the *same*
     normalisation as the runtime tokeniser (`sa_tokenizer.normalise`), so that
     `tokeniser(text) == treebank tokens`. The Vedic treebank is already IAST, so this
     is a no-op there.

  2. **Re-segment the multiword tokens (MWTs)** in a CSL variant that makes the internal
     structure recoverable *and* deterministically re-tokenisable:

       * **Compound-internal** boundaries (the treebank's own `Compound=Yes` feature on
         the left member) are joined with a **hyphen** (CSL prints a thin vertical line;
         we use `-`), and the whole compound stays one MWT range. The hyphen is baked
         into the non-final member's surface form, so concatenation reproduces the text
         and the tokeniser recovers the members by splitting on word-internal hyphens.

       * **External-sandhi** boundaries (between separate words written together in one
         UFAL MWT) are **re-segmented into separate space-joined tokens**, each in its
         **surface** form (`vahniḥ idraḥ` -> `vahnir idraḥ`). The MWT range is dropped
         across such a boundary.

       * **Vowel coalescence** at *either* kind of boundary is marked CSL-style: the left
         element loses its final vowel and takes `'` (elided short a/i/u) or `"` (elided
         long ā/ī/ū); the right element's initial vowel becomes the marked coalescence
         result -- circumflex when the original vowel was short (â ê î ô û / âi âu),
         macron when it was long (ā ē ī ō ū). Consonant/visarga sandhi keeps the surface
         form (already encoded in the combined form), with the divider inserted.

The Vedic treebank has no MWTs, so transform 2 is a no-op there too -- the script is the
shared front-end for both treebanks ("do the same processing ... to the extent it
contains MWTs").

Hard MWTs (stacked avagraha / multi-junction coalescence the heuristic aligner cannot
place confidently) are written with a best-effort rendering, tagged `CSLReview=Yes` on
the range/first row, and listed in the --review TSV for hand-correction. Supply the
corrected renderings back via --overrides (TSV: sent_id<TAB>range<TAB>display) to bake
them in.

Usage:
  .venv/bin/python scripts/sa_csl_prep.py IN.conllu OUT.conllu \
      [--review review.tsv] [--overrides overrides.tsv]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retokenize import read_conllu  # noqa: E402
from sa_tokenizer import normalise  # same Devanagari->IAST + accent handling as runtime

# CSL prints direct speech in double guillemets « » (so they don't clash with the sandhi
# apostrophes). Map the edition's typographic double quotes to them BEFORE normalise (which
# straightens curly quotes), preserving open/close.
_GUILLEMET = {0x201C: "«", 0x201D: "»"}


def to_iast(s):
    return normalise(s.translate(_GUILLEMET))

# --------------------------------------------------------------------------- #
# CSL vowel-coalescence model
# --------------------------------------------------------------------------- #
VOW = "aāiīuūṛṝḷeo"
APOS, DAPOS = "'", '"'               # elided short / long vowel marks
MUT_FINAL = set("ḥṃṁtdpkmnṅ")        # word-final consonants that change in sandhi


def last_vowel(s):
    if s.endswith(("ai", "au")):
        return s[-2:]
    return s[-1] if s and s[-1] in VOW else None


def first_vowel(s):
    if s[:2] in ("ai", "au"):
        return s[:2]
    return s[0] if s and s[0] in VOW else None


def coalesce(v1, v2):
    """(left final vowel, right initial vowel) -> (surface result vowel, CSL right mark)."""
    if v1 in ("a", "ā"):
        return {"a": ("ā", "â"), "ā": ("ā", "ā"), "i": ("e", "ê"), "ī": ("e", "ē"),
                "u": ("o", "ô"), "ū": ("o", "ō"), "e": ("ai", "âi"), "ai": ("ai", "ai"),
                "o": ("au", "âu"), "au": ("au", "āu")}.get(v2)
    if v1 in ("i", "ī") and v2 in ("i", "ī"):
        return ("ī", "î" if v2 == "i" else "ī")
    if v1 in ("u", "ū") and v2 in ("u", "ū"):
        return ("ū", "û" if v2 == "u" else "ū")
    return None


def carve(kids, S):
    """Split surface form S into one CSL display piece per analysed child.

    Returns (pieces, valid). `pieces` is None only when the surface cannot be aligned at
    all; otherwise it is the best-effort rendering and `valid` says whether it passed
    surface-reversion + per-word core validation.
    """
    n = len(kids)
    fired = [None] * (n - 1)               # per junction: (v1, v2, res, mark) if coalesced
    for j in range(n - 1):
        v1, v2 = last_vowel(kids[j]), first_vowel(kids[j + 1])
        if v1 and v2:
            c = coalesce(v1, v2)
            if c and (kids[j][:-len(v1)] + c[0]) in S:
                fired[j] = (v1, v2, c[0], c[1])

    cuts = [0]
    p = 0
    for i in range(n):
        if i > 0 and fired[i - 1]:                    # this child owns the coalesced vowel
            p += len(fired[i - 1][2])
        if i == n - 1:
            p = len(S)
        elif fired[i]:                                # right junction coalesces: consume stem
            v1 = fired[i][0]
            stem = kids[i][:-len(v1)]
            lead = len(first_vowel(kids[i])) if (i > 0 and fired[i - 1]) else 0
            body = stem[lead:]
            if S[p:p + len(body)] != body:
                return None, False
            p += len(body)
        else:                                         # consonant/concat: find next word's start
            nxt = kids[i + 1]
            consumed = len(first_vowel(kids[i])) if (i > 0 and fired[i - 1]) else 0
            est = p + len(kids[i]) - consumed         # consonant sandhi is ~length-preserving
            fv = first_vowel(nxt)

            def ok(pos):
                if pos <= p or pos > len(S):
                    return False
                if S.startswith(nxt[:2], pos) or S.startswith(nxt[:1], pos):
                    return True
                if fv and S.startswith(APOS + nxt[len(fv):][:2], pos):  # avagraha
                    return True
                return False

            cand = next((est + d for d in (0, -1, 1, -2, 2, -3, 3) if ok(est + d)), None)
            if cand is None:
                return None, False
            p = cand
        cuts.append(p)
    if cuts[-1] != len(S):
        return None, False

    pieces = []
    for i in range(n):
        seg = S[cuts[i]:cuts[i + 1]]
        if i > 0 and fired[i - 1]:                    # leading coalesced vowel -> CSL mark
            res, mark = fired[i - 1][2], fired[i - 1][3]
            seg = mark + seg[len(res):]
        if i < n - 1 and fired[i]:                    # trailing elision -> ' / "
            seg = seg + (APOS if fired[i][0] in ("a", "i", "u") else DAPOS)
        pieces.append(seg)

    # validation 1: reverting CSL marks to plain surface must reproduce S
    rev = ""
    for i, seg in enumerate(pieces):
        s2 = seg
        if i > 0 and fired[i - 1]:
            s2 = fired[i - 1][2] + s2[len(fired[i - 1][3]):]
        if i < n - 1 and fired[i]:
            s2 = s2[:-1]
        rev += s2
    if rev != S:
        return pieces, False

    # validation 2: every word's stable core must sit in its own surface slice
    valid = True
    for i in range(n):
        C = kids[i]
        lm = i > 0
        # how many trailing chars may mutate: a final vowel+visarga changes BOTH (aḥ -> o,
        # āḥ -> ā), a final stop / single vowel changes one, otherwise none.
        if len(C) >= 2 and C[-1] == "ḥ" and C[-2] in VOW:
            rstrip = 2
        elif i == n - 1 or bool(fired[i]) or C[-1] in MUT_FINAL or last_vowel(C) is not None:
            rstrip = 1
        else:
            rstrip = 0
        a = 1 if (lm and first_vowel(C)) else 0
        b = len(C) - rstrip
        core = C[a:b]
        sl = S[cuts[i]:cuts[i + 1]]
        ext = (fired[i - 1][3] + sl) if (i > 0 and fired[i - 1]) else sl
        if len(core) >= 2 and core not in sl and core not in ext:
            valid = False
            break
    return pieces, valid


# --------------------------------------------------------------------------- #
# CoNLL-U rewriting
# --------------------------------------------------------------------------- #
def set_misc(misc, drop=("Translit", "LTranslit"), space_after=None):
    """Return MISC with stale translit fields dropped and SpaceAfter forced if given."""
    keep = []
    for f in misc.split("|"):
        if f in ("", "_"):
            continue
        k = f.split("=", 1)[0]
        if k in drop or k == "SpaceAfter":
            continue
        keep.append(f)
    if space_after is False:
        keep.append("SpaceAfter=No")
    return "|".join(keep) if keep else "_"


def rewrite_sentence(comments, toks, mwts, combined, overrides, review, sid):
    by_id = {int(t.id): t for t in toks}
    # transliterate every token's FORM/LEMMA up front
    for t in toks:
        t.form = to_iast(t.form)
        if t.lemma not in ("_", ""):
            t.lemma = to_iast(t.lemma)

    # walk tokens, expanding MWTs into (range-rows + token-rows) groups
    out_ranges = []      # (start_id, end_id, combined_form, misc)
    order = sorted(int(i) for i in by_id)
    i = 0
    review_rows = []
    while i < len(order):
        tid = order[i]
        if tid in mwts:
            end, _form, rmisc = mwts[tid]
            ids = [j for j in range(tid, end + 1)]
            kids_surface = [normalise(by_id[j].form) for j in ids]  # already transliterated
            comp = ["Compound=Yes" in by_id[j].feats.split("|") for j in ids]
            key = (sid, f"{tid}-{end}")

            if key in overrides:
                disp = overrides[key]
                pieces, dividers = _parse_display(disp)   # corrected by hand
                flagged = False
            else:
                S = combined[tid]                      # transliterated surface combined form
                pieces, valid = carve(kids_surface, S)
                flagged = not valid or pieces is None
                if pieces is None:                     # could not align: plain analysed fallback
                    pieces = kids_surface
                dividers = ["-" if comp[k] else " " for k in range(len(ids) - 1)]
                disp = _join(pieces, dividers)
                if flagged:
                    review_rows.append((sid, f"{tid}-{end}", S, " ".join(kids_surface),
                                        "".join("C" if c else "." for c in comp), disp))

            _emit_groups(ids, pieces, dividers, by_id, rmisc, out_ranges, flagged)
            i += len(ids)
        else:
            t = by_id[tid]
            t.misc = set_misc(t.misc, space_after=(False if not t.space_after() else None))
            i += 1

    review.extend(review_rows)
    return out_ranges


def _parse_display(disp):
    """Turn a CSL display string 'a-b c' into (pieces, dividers)."""
    pieces, dividers, cur = [], [], ""
    i = 0
    while i < len(disp):
        ch = disp[i]
        if ch in "- ":
            pieces.append(cur)
            dividers.append(ch)
            cur = ""
        else:
            cur += ch
        i += 1
    pieces.append(cur)
    return pieces, dividers


def _join(pieces, dividers):
    out = pieces[0]
    for k in range(1, len(pieces)):
        out += dividers[k - 1] + pieces[k]
    return out


def _emit_groups(ids, pieces, dividers, by_id, rmisc, out_ranges, flagged):
    """Bake hyphens + SpaceAfter into the child rows and record compound MWT ranges.

    Children are split into groups at every space divider; each group of length >= 2 is a
    compound -> a hyphen-baked MWT range; length 1 -> a standalone token. The hyphen is
    appended to non-final compound members' FORM; SpaceAfter=No marks no-space junctions.
    """
    orig_last_space = "SpaceAfter=No" not in rmisc.split("|")  # space after the whole MWT?
    # group boundaries: new group starts after a space divider
    groups, cur = [], [0]
    for k in range(len(ids) - 1):
        if dividers[k] == " ":
            groups.append(cur)
            cur = [k + 1]
        else:
            cur.append(k + 1)
    groups.append(cur)

    for gi, g in enumerate(groups):
        last_group = gi == len(groups) - 1
        for pos, k in enumerate(g):
            tok = by_id[ids[k]]
            within = pos < len(g) - 1                  # not the last member of this group
            form = pieces[k] + ("-" if within else "")
            tok.form = form
            if within:
                tok.misc = set_misc(tok.misc, space_after=False)         # hyphen, no space
            elif not last_group:
                tok.misc = set_misc(tok.misc, space_after=None)          # external -> space
            else:                                                        # MWT-final member
                tok.misc = set_misc(tok.misc,
                                    space_after=(False if not orig_last_space else None))
            if flagged and gi == 0 and pos == 0:
                tok.misc = (tok.misc + "|CSLReview=Yes") if tok.misc != "_" else "CSLReview=Yes"
        if len(g) >= 2:
            combined = "".join(pieces[k] + ("-" if pos < len(g) - 1 else "")
                               for pos, k in enumerate(g))
            out_ranges.append((ids[g[0]], ids[g[-1]], combined,
                               "CSLReview=Yes" if (flagged and gi == 0) else "_"))


def rebuild_text(toks):
    s = []
    for t in toks:
        s.append(t.form)
        if t.space_after():
            s.append(" ")
    return "".join(s).rstrip()


def process(in_path, out_path, review_path, overrides):
    review = []
    n_sent = n_mwt = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for comments, toks, mwts in read_conllu(in_path):
            n_sent += 1
            n_mwt += len(mwts)
            sid = next((c.split("=", 1)[1].strip()
                        for c in comments if c.startswith("# sent_id")), "?")
            combined = {st: to_iast(f) for st, (e, f, m) in mwts.items()}
            ranges = rewrite_sentence(comments, toks, mwts, combined, overrides, review, sid)
            # write: comments (with rebuilt # text), then interleaved range + token rows
            range_at = {}
            for a, b, form, misc in ranges:
                range_at.setdefault(a, []).append((b, form, misc))
            for c in comments:
                if c.startswith("# text ="):
                    out.write("# text = " + rebuild_text(toks) + "\n")
                elif c.startswith("# translit ="):
                    continue                                   # now redundant (FORM is IAST)
                else:
                    out.write(c + "\n")
            for t in toks:
                tid = int(t.id)
                for b, form, misc in range_at.get(tid, []):
                    out.write("\t".join([f"{tid}-{b}", form, "_", "_", "_", "_",
                                         "_", "_", "_", misc]) + "\n")
                out.write("\t".join([t.id, t.form, t.lemma, t.upos, t.xpos, t.feats,
                                     t.head, t.deprel, t.deps, t.misc]) + "\n")
            out.write("\n")

    if review_path:                       # always (re)write, so it reflects current state
        with open(review_path, "w", encoding="utf-8") as r:
            r.write("sent_id\trange\tsurface\tanalysed\tcompound\tbest_effort\n")
            for row in review:
                r.write("\t".join(row) + "\n")
    print(f"{in_path}: {n_sent} sentences, {n_mwt} MWTs -> {out_path}"
          f"  ({len(review)} flagged for review)", file=sys.stderr)


def load_overrides(path):
    """Read hand-corrected renderings. Accepts either a 3-column file
    (sent_id, range, display) or the edited --review TSV itself (the corrected CSL goes
    in the last/best_effort column). The display must have exactly one hyphen/space-
    separated piece per child of the MWT."""
    ov = {}
    if not path or not os.path.exists(path):
        return ov
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("sent_id\t") or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[-1].strip():
            ov[(parts[0], parts[1])] = parts[-1].strip()
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--review", default=None)
    ap.add_argument("--overrides", default=None)
    a = ap.parse_args()
    process(a.inp, a.out, a.review, load_overrides(a.overrides))


if __name__ == "__main__":
    main()
