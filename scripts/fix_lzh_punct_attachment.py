#!/usr/bin/env python3
"""Re-attach punctuation so it does not create non-projectivity, per the UD convention.

WHY. In the lzh corpus **16.01 % of PUNCT arcs are non-projective against 1.41 % of content arcs**
— eleven times the rate — and it is concentrated in CLOSING marks: 」 83 %, ）85 %, ？48 %, 、37 %.
`align_kanripo_punct.py` attaches each mark to "the root of the material on its LEFT", and for a
closing quote that material is the last clause INSIDE the quote, whose own head lies outside it, so
the arc crosses. The merge is not the cause: the pre-merge corpus has the identical 16.01 %.

⚠ **THE COST IS NOT PUNCTUATION'S OWN ARCS, WHICH LAS DOES NOT SCORE.** spaCy pseudo-projectivises
the training data and deprojectivises on output, so every non-projective arc becomes a DECORATED
label — 13 743 of them here, inflating the action inventory the parser must learn over. The
hypothesis this script exists to test is that cleaning punctuation buys accuracy on CONTENT arcs.

THE RULE, which is UD's: punctuation attaches to the head of the phrase it sits in, as HIGH as
possible without creating non-projectivity, and never crosses another arc. Implemented as: among
all non-punct candidates whose arc from this mark crosses nothing, take the one closest to the
root; break ties toward the nearer token, and toward the LEFT for a closing mark and the RIGHT for
an opening one, so a quote's marks stay with the span they delimit.

⚠ The relation stays `punct` and the mark's own position never moves — only the head changes.
"""
import argparse
import collections
import pathlib


def read(path):
    out, cur, com = [], [], []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                out.append((com, cur))
            cur, com = [], []
            continue
        if line.startswith("#"):
            com.append(line)
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f)
    if cur:
        out.append((com, cur))
    return out


DELIM = set("，、；：。？！,;:.?!")


def normalise_runs(b, heads):
    """A RUN of adjacent punctuation marks must NEST, not cross.

    `擇不處仁，焉得知」？` has 」 on 得 (7->9) and ？ on 知 (8->10): two adjacent closing marks
    attached at incompatible depths, so the two arcs cross EACH OTHER. This is an annotation
    inconsistency, not a grammatical fact — the outer mark of a run cannot attach deeper than the
    inner one. Levelling a run onto its leftmost mark's head commits to nothing about the grammar:
    no content word changes head, and each mark still delimits the same material.
    """
    n, i, moved = len(b), 0, 0
    while i < n:
        if b[i][3] != "PUNCT":
            i += 1
            continue
        j = i
        while j + 1 < n and b[j + 1][3] == "PUNCT":
            j += 1
        if j > i:
            # the run's outermost (leftmost) mark sets the depth; anything attached OUTSIDE the run
            # and deeper is pulled up to it
            anchor = heads[i]
            if not (i <= anchor <= j):
                for m in range(i + 1, j + 1):
                    if heads[m] != m and not (i <= heads[m] <= j) and heads[m] != anchor:
                        heads[m] = anchor
                        moved += 1
        i = j + 1
    return moved


def delimiter_spans(forms):
    """Scope of each ONE-SIDED delimiter: the stretch it closes, back to the previous delimiter or
    to the edge of the innermost enclosing bracket span. Same principle as a bracket pair — the mark
    attaches to the head of the material it delimits — with only one side written."""
    inner = {}
    for o, c in matched_pairs(forms):
        for j in range(o + 1, c):
            inner[j] = (o + 1, c - 1)
    out, lo = [], 0
    for i, t in enumerate(forms):
        blo, bhi = inner.get(i, (0, len(forms) - 1))
        if i in inner and lo < blo:
            lo = blo
        if t in DELIM:
            if lo <= i - 1:
                out.append((lo, i - 1, (i,)))
            lo = i + 1
        elif t in PAIRS or t in CLOSERS:
            lo = i + 1
    return out


def edge_spans(forms):
    """Spans for UNMATCHED marks. Kyoto breaks a quotation across blocks, so most openers have no
    closer in their own block (4 292 of 11 291 bracket tokens are in a matched pair). An opener
    then encloses everything to the END of the block, a closer everything from its START — which is
    exactly the material the mark is bounding, just truncated by the block break."""
    paired = {i for o, c in matched_pairs(forms) for i in (o, c)}
    out = []
    for i, t in enumerate(forms):
        if i in paired:
            continue
        # ⚠ Only the MARK is movable. The other endpoint is the block edge — a synthetic boundary,
        # often a content word — and re-attaching it corrupted 2 151 arcs (今, 王, 臣, 以 among them)
        # before this was split out.
        if t in PAIRS:
            out.append((i, len(forms) - 1, (i,)))
        elif t in CLOSERS:
            out.append((0, i, (i,)))
    return out


def crossings(heads, i, h):
    """Would the arc (i -> h) cross any arc not involving i?"""
    a, b = sorted((i, h))
    for j, hj in enumerate(heads):
        if hj == j or j == i or hj == i:
            continue
        c, d = sorted((j, hj))
        if a < c < b < d or c < a < d < b:
            return True
    return False


def depth(heads, i, cap=200):
    d, seen = 0, set()
    while heads[i] != i and i not in seen and d < cap:
        seen.add(i)
        i = heads[i]
        d += 1
    return d


CLOSERS = set("」』）〕》】”’")
OPENERS = set("「『（〔《【“‘")
PAIRS = {"「": "」", "『": "』", "（": "）", "〔": "〕", "《": "》", "【": "】",
         "“": "”", "‘": "’", "(": ")", "[": "]"}


def matched_pairs(forms):
    """Outermost matched (open, close) index pairs. A repeated opener of the same kind is a
    CONTINUATION mark, not a nesting level — the convention `sent_join` also has to handle."""
    stack, out = [], []
    for i, t in enumerate(forms):
        if t in PAIRS:
            if any(x == t for x, _ in stack):
                continue
            stack.append((t, i))
        elif t in CLOSERS:
            for d in range(len(stack) - 1, -1, -1):
                if PAIRS.get(stack[d][0]) == t:
                    if d == 0:
                        out.append((stack[0][1], i))
                    del stack[d:]
                    break
    return out


DELIM = set("，、；：。？！,;:.?!")


def normalise_runs(b, heads):
    """A RUN of adjacent punctuation marks must NEST, not cross.

    `擇不處仁，焉得知」？` has 」 on 得 (7->9) and ？ on 知 (8->10): two adjacent closing marks
    attached at incompatible depths, so the two arcs cross EACH OTHER. This is an annotation
    inconsistency, not a grammatical fact — the outer mark of a run cannot attach deeper than the
    inner one. Levelling a run onto its leftmost mark's head commits to nothing about the grammar:
    no content word changes head, and each mark still delimits the same material.
    """
    n, i, moved = len(b), 0, 0
    while i < n:
        if b[i][3] != "PUNCT":
            i += 1
            continue
        j = i
        while j + 1 < n and b[j + 1][3] == "PUNCT":
            j += 1
        if j > i:
            # the run's outermost (leftmost) mark sets the depth; anything attached OUTSIDE the run
            # and deeper is pulled up to it
            anchor = heads[i]
            if not (i <= anchor <= j):
                for m in range(i + 1, j + 1):
                    if heads[m] != m and not (i <= heads[m] <= j) and heads[m] != anchor:
                        heads[m] = anchor
                        moved += 1
        i = j + 1
    return moved


def delimiter_spans(forms):
    """Scope of each ONE-SIDED delimiter: the stretch it closes, back to the previous delimiter or
    to the edge of the innermost enclosing bracket span. Same principle as a bracket pair — the mark
    attaches to the head of the material it delimits — with only one side written."""
    inner = {}
    for o, c in matched_pairs(forms):
        for j in range(o + 1, c):
            inner[j] = (o + 1, c - 1)
    out, lo = [], 0
    for i, t in enumerate(forms):
        blo, bhi = inner.get(i, (0, len(forms) - 1))
        if i in inner and lo < blo:
            lo = blo
        if t in DELIM:
            if lo <= i - 1:
                out.append((lo, i - 1, (i,)))
            lo = i + 1
        elif t in PAIRS or t in CLOSERS:
            lo = i + 1
    return out


def edge_spans(forms):
    """Spans for UNMATCHED marks. Kyoto breaks a quotation across blocks, so most openers have no
    closer in their own block (4 292 of 11 291 bracket tokens are in a matched pair). An opener
    then encloses everything to the END of the block, a closer everything from its START — which is
    exactly the material the mark is bounding, just truncated by the block break."""
    paired = {i for o, c in matched_pairs(forms) for i in (o, c)}
    out = []
    for i, t in enumerate(forms):
        if i in paired:
            continue
        # ⚠ Only the MARK is movable. The other endpoint is the block edge — a synthetic boundary,
        # often a content word — and re-attaching it corrupted 2 151 arcs (今, 王, 臣, 以 among them)
        # before this was split out.
        if t in PAIRS:
            out.append((i, len(forms) - 1, (i,)))
        elif t in CLOSERS:
            out.append((0, i, (i,)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--punct-runs", action="store_true",
                    help="level a RUN of adjacent marks onto the leftmost mark's head, so the "
                         "arcs of adjacent punctuation nest instead of crossing each other")
    ap.add_argument("--delimiters", action="store_true",
                    help="also re-attach one-sided delimiters (，、；。？) to the head of the "
                         "stretch of text they close")
    ap.add_argument("--to-edge", action="store_true",
                    help="also re-attach an UNMATCHED mark, treating it as enclosing the material "
                         "up to the block edge (Kyoto splits quotations across blocks)")
    ap.add_argument("--pairs-only", action="store_true",
                    help="re-attach ONLY matched bracket/quote pairs; leave every other "
                         "punctuation arc exactly as the treebank has it")
    ap.add_argument("--quote-aware", action="store_true",
                    help="attach a matched bracket/quote PAIR to the head of the material it "
                         "encloses, instead of to the highest projective host")
    a = ap.parse_args()

    blocks = read(a.inp)
    quote_aware = a.quote_aware or a.pairs_only
    pairs_only = a.pairs_only
    to_edge = a.to_edge
    delimiters = a.delimiters
    punct_runs = a.punct_runs
    moved = kept = stuck = 0
    before = after = 0
    lines = []
    for com, b in blocks:
        n = len(b)
        heads = [(i if int(f[6]) == 0 else int(f[6]) - 1) for i, f in enumerate(b)]
        pair_done = set()
        if punct_runs:
            moved += normalise_runs(b, heads)
        # ⚠ NEVER MOVE A PUNCT TOKEN THAT IS THE BLOCK ROOT. A handful of Kyoto blocks are wholly
        # punctuation, so their root IS a mark; re-attaching it leaves the block rootless, and if
        # the new head is inside its own subtree, cyclic. One of each reached the output before this
        # guard, and `spacy convert` accepts both without complaint.
        punct = [i for i, f in enumerate(b) if f[3] == "PUNCT" and heads[i] != i]

        # ⚠ A MATCHED PAIR GOES TO THE HEAD OF WHAT IT ENCLOSES. The generic "highest projective
        # host" rule below satisfies projectivity but DISCARDS the boundary: a 」 re-attached to
        # some nearby token no longer says where the quote ended, and retraining on that corpus
        # cost 2.03 SENTS_F. Anchoring both marks on the enclosed material's own head keeps the
        # boundary AND is projective by construction — the span is contiguous and its content is a
        # single subtree, so both arcs nest inside that head's outgoing arc rather than crossing it.
        if quote_aware:
            spans = [(o, c, (o, c)) for o, c in matched_pairs([f[1] for f in b])]
            if to_edge:
                spans = spans + edge_spans([f[1] for f in b])
            if delimiters:
                spans = spans + delimiter_spans([f[1] for f in b])
            for o, c, movable in spans:
                inner = [j for j in range(o + 1, c) if b[j][3] != "PUNCT"]
                if not inner:
                    continue
                host = next((j for j in inner if not (o < heads[j] < c)), inner[0])
                for m in movable:
                    if heads[m] != m and not crossings(heads, m, host) and host != m:
                        if heads[m] != host:
                            moved += 1
                        heads[m] = host
                        pair_done.add(m)
            punct = [i for i in punct if i not in pair_done]
        if pairs_only:
            punct = []
        before += sum(1 for i in punct if crossings(heads, i, heads[i]))
        for i in punct:
            cands = [j for j in range(n) if j != i and b[j][3] != "PUNCT"
                     and not crossings(heads, i, j)]
            if not cands:
                stuck += 1
                continue
            # highest (closest to the root) first; then nearest; then the side the mark delimits
            side = -1 if b[i][1] in CLOSERS else (1 if b[i][1] in OPENERS else 0)
            best = min(cands, key=lambda j: (depth(heads, j), abs(j - i),
                                             0 if side == 0 or (j - i) * side > 0 else 1))
            if best != heads[i]:
                moved += 1
                heads[i] = best
            else:
                kept += 1
        after += sum(1 for i in punct if crossings(heads, i, heads[i]))
        for i, f in enumerate(b):
            g = list(f)
            g[6] = "0" if heads[i] == i else str(heads[i] + 1)
            g[7] = "root" if heads[i] == i else g[7]
            lines.append("\t".join(g))
        lines.append("")
        for c in com:
            pass
    # rewrite with the original comments preserved
    out_lines, k = [], 0
    for com, b in blocks:
        out_lines.extend(com)
        for _ in b:
            out_lines.append(lines[k]); k += 1
        out_lines.append(lines[k]); k += 1
    pathlib.Path(a.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tot = moved + kept + stuck
    print(f"{a.out}: {tot} punct tokens  moved {moved} ({moved/max(tot,1):.1%})  "
          f"kept {kept}  no projective host {stuck}")
    print(f"   non-projective punct arcs {before} -> {after}")


if __name__ == "__main__":
    main()
