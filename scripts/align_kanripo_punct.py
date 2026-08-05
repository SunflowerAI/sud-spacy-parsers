#!/usr/bin/env python
"""Restore punctuation to SUD_Classical_Chinese-Kyoto from the Kanseki Repository editions.

Kyoto states in its README that it included no spaces or punctuation, because Classical Chinese had
none — so the parser has never seen a punctuation mark (5 tokens in 374 560) and `clause_parser` has
to strip every mark before parsing. But the treebank was built FROM kanripo (KRxxxxxxx sent_ids;
Christian Wittern contributes to both), and the kanripo editions ARE punctuated. Both are
CC BY-SA 4.0, so the marks can be put back.

**Alignment is by content, not by identifier.** The sent_id's second field looks like a kanripo file
number and is one for 論語 and 禮記, but 戰國策 numbers 046–501 against kanripo's 000–010 and 楚辭
has no section numbering at all. So this aligns the two CHARACTER STREAMS of a work: Kyoto's units
in text order against the kanripo text with markup and punctuation removed. They are the same text,
so a two-pointer walk with resync on divergence maps almost every mark onto a Kyoto character
position; marks over a divergence (variant characters, paratext, passages one edition lacks) are
DROPPED rather than guessed, and counted in the report.

**What this does and does not decide.** Each mark is inserted as a `punct` token, attached the way
`clause_parser` attaches one at inference: a closing or pause mark to the head of the material on
its LEFT, an opening mark to the root of the material on its RIGHT. Sentences stay one 句讀 unit
each — this script does NOT merge units into punctuation-delimited sentences, because that needs a
relation between unit roots, which Kyoto never annotated. It records the grouping it would imply
(`# sent_group`, `# sent_final`) so that work can proceed from here.

**The WITNESS matters, and kanripo keeps witnesses on git BRANCHES.** 戰國策 aligns at 0.3 % on
master and on the SBCK witness, 75.5 % on WYG and **93.6 % on `tls`** — and the tls branch has 502
files against Kyoto's file numbers 046–501, which is the tell. Clone per work:

    git clone --depth 1 https://github.com/kanripo/KR1h0004.git DIR/KR1h0004
    git clone --depth 1 --branch tls https://github.com/kanripo/KR2e0003.git DIR/KR2e0003

Two works have no usable kanripo source: KR4h0169 has no repository at all, and 十八史略 KR2b0041
aligns at only 49.8 % on its one branch — the Kyoto README sources it from a `18shilue` path of its
own rather than from kanripo, so the transcriptions genuinely differ. Their sentences pass through
untouched.

Usage:
    align_kanripo_punct.py --kanripo DIR --conllu F [F ...] [--out-suffix .punct] [--dry-run]

DIR holds the cloned kanripo repos (one directory per work, e.g. DIR/KR1h0004/KR1h0004_004.txt).
"""
import argparse
import collections
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clause_parser import punct_tag  # noqa: E402  (single source of truth for the 記号 tagset)

# org-mode metadata, page-break anchors, the paragraph mark, section numbers, any residual tag
MARKUP = re.compile(r"<pb:[^>]*>|<[^>]*>|¶|\d+\.\d+")
# Opening marks attach RIGHT (they belong to what follows); everything else attaches LEFT.
OPENERS = set("（「『【〔《〈［｛(<[{“‘«‹")
# Marks that would end a sentence, recorded for the follow-on work (see `# sent_final`).
SENT_FINAL = set("。．！？!?…")
SENT_ID = re.compile(r"^(KR\w+?)_(\d+)_(?:par(\d+)_(\d+)|(title))")


# --------------------------------------------------------------------------- CoNLL-U


class Sent:
    __slots__ = ("comments", "sid", "toks")

    def __init__(self, comments, sid, toks):
        self.comments, self.sid, self.toks = comments, sid, toks

    @property
    def text(self):
        return "".join(t[1] for t in self.toks)


def read_conllu(path):
    sents, comments, toks, sid = [], [], [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if toks:
                sents.append(Sent(comments, sid, toks))
            comments, toks, sid = [], [], None
        elif line.startswith("#"):
            comments.append(line)
            if line.startswith("# sent_id"):
                sid = line.split("= ", 1)[1]
        else:
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]:
                toks.append(f)
    if toks:
        sents.append(Sent(comments, sid, toks))
    return sents


def write_conllu(path, sents):
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            for c in s.comments:
                fh.write(c + "\n")
            for t in s.toks:
                fh.write("\t".join(t) + "\n")
            fh.write("\n")


def sort_key(sid):
    """Text order within a work: (file, paragraph, character offset). `_title` sorts first."""
    m = SENT_ID.match(sid or "")
    if not m:
        return (10 ** 9, 10 ** 9, 10 ** 9)
    return (int(m.group(2)), 0 if m.group(5) else int(m.group(3)),
            0 if m.group(5) else int(m.group(4)))


def work_of(sid):
    m = SENT_ID.match(sid or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------- kanripo


def is_mark(c):
    return unicodedata.category(c).startswith("P")


def kanripo_stream(kdir, work):
    """(bare_text, [(index_into_bare, mark), ...]) for a whole work, files in numeric order.

    A mark's index is the position in the BARE stream it precedes — i.e. the number of content
    characters before it — which is the coordinate system the treebank text uses.
    """
    wdir = os.path.join(kdir, work)
    if not os.path.isdir(wdir):
        return None
    files = sorted(f for f in os.listdir(wdir) if f.startswith(work) and f.endswith(".txt"))
    bare, marks = [], []
    for fn in files:
        for line in open(os.path.join(wdir, fn), encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith(("#", "*")):
                continue                       # org headers, `# src:` notes, outline headings
            for c in MARKUP.sub("", line):
                if c.isspace():
                    continue
                if is_mark(c):
                    marks.append((len(bare), c))
                else:
                    bare.append(c)
    return "".join(bare), marks


# --------------------------------------------------------------------------- alignment


def align(gold, other, anchor=12):
    """Map indices of `other` (kanripo bare) onto `gold` (Kyoto). Returns a list the length of
    `other`, holding the gold index each character matched, or None where it did not.

    The two streams are the same text, so this walks them in lockstep and only pays for divergence.
    Resync is by GLOBAL anchor lookup, not a local window: kanripo files carry front matter and
    paratext the treebank never took, and 戰國策 opens with ~thousands of characters of it, so a
    bounded local search finds nothing and (in the first version of this script) abandoned the whole
    work. An `anchor`-character index of `gold` finds where the current position of `other` belongs
    however far away it is, and taking the earliest candidate at or after the current gold position
    keeps the walk monotonic, so a formulaic phrase cannot pull the alignment backwards. A stretch
    with no anchor at all is skipped one character at a time and its marks are dropped.
    """
    out = [None] * len(other)
    index = collections.defaultdict(list)
    for p in range(len(gold) - anchor + 1):
        index[gold[p:p + anchor]].append(p)
    i = j = 0
    while i < len(gold) and j < len(other):
        if gold[i] == other[j]:
            out[j] = i
            i += 1
            j += 1
            continue
        cands = index.get(other[j:j + anchor]) if j + anchor <= len(other) else None
        nxt = next((p for p in cands if p >= i), None) if cands else None
        if nxt is None:
            j += 1                              # unbridgeable here; drop this character's marks
            continue
        i = nxt
    return out


# --------------------------------------------------------------------------- insertion


def unit_head(idxs, toks):
    """Head of a contiguous run of tokens: the one whose own head lies outside the run (its root,
    or the token linking it to the rest of the tree). Leftmost on a tie. Same rule as
    `clause_parser._unit_head`, so training data and inference agree."""
    ids = {toks[k][0] for k in idxs}
    for k in idxs:
        if toks[k][6] == "0" or toks[k][6] not in ids:
            return k
    return idxs[0]


def insert_marks(sent, placements):
    """Insert marks into one sentence. `placements` is [(offset, mark)], offset = number of the
    sentence's own characters before the mark. Returns the number inserted."""
    toks = sent.toks
    # character offset -> token index boundary (all Kyoto tokens here are single characters, but
    # 852 of 27 566 are longer, so walk the widths rather than assuming)
    bound, run = {0: 0}, 0
    for k, t in enumerate(toks):
        run += len(t[1])
        bound[run] = k + 1
    new, old_to_new, by_pos = [], {}, collections.defaultdict(list)
    for off, mark in placements:
        if off in bound:
            by_pos[bound[off]].append(mark)
    inserted = 0
    for k in range(len(toks) + 1):
        for mark in by_pos.get(k, []):
            new.append(["", mark, mark, "PUNCT", punct_tag(mark), "_", "", "punct", "_",
                        "SpaceAfter=No"])
            inserted += 1
        if k < len(toks):
            old_to_new[toks[k][0]] = len(new)
            new.append(list(toks[k]))
    # renumber, then repoint: content heads through the map, marks onto their anchor
    for n, t in enumerate(new, 1):
        t[0] = str(n)
    content = [n for n, t in enumerate(new) if t[7] != "punct" or t[3] != "PUNCT"]
    for n, t in enumerate(new):
        if t[3] == "PUNCT" and t[7] == "punct":
            left = [c for c in content if c < n]
            right = [c for c in content if c > n]
            # a run stops at the previous / next mark, so a mark anchors to its OWN unit
            if t[1] in OPENERS and right:
                run_ = []
                for c in right:
                    if c > n and all(new[x][3] != "PUNCT" for x in range(n + 1, c + 1)):
                        run_.append(c)
                anchor = unit_head(run_ or [right[0]], new)
            elif left:
                run_ = [c for c in left
                        if all(new[x][3] != "PUNCT" for x in range(c, n))]
                anchor = unit_head(run_ or [left[-1]], new)
            elif right:
                anchor = unit_head([right[0]], new)
            else:
                continue
            t[6] = new[anchor][0]
        else:
            t[6] = "0" if t[6] == "0" else old_to_new_id(old_to_new, t[6], new)
    sent.toks = new
    return inserted


def old_to_new_id(mapping, old, new):
    k = mapping.get(old)
    return new[k][0] if k is not None else "0"


# --------------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kanripo", required=True, help="directory of cloned KRxxxxxxx repos")
    ap.add_argument("--conllu", nargs="+", required=True)
    ap.add_argument("--out-suffix", default=".punct")
    # kanripo keeps each witness on its own BRANCH (KR2e0003: SBCK / WYG / tls / master), and the
    # treebank followed one of them — so a work that aligns poorly is worth retrying per witness.
    ap.add_argument("--work", action="append", help="restrict to these works (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args()

    # every split of a work has to be aligned together — the splits are disjoint by chapter, but a
    # work's character stream runs through all three
    per_file = {p: read_conllu(p) for p in args.conllu}
    works = collections.defaultdict(list)
    for p, sents in per_file.items():
        for s in sents:
            w = work_of(s.sid)
            if w and (not args.work or w in args.work):
                works[w].append((p, s))

    print(f"{'work':<12} {'units':>7} {'chars':>8} {'aligned':>8} {'marks':>7} {'placed':>7} "
          f"{'at bdry':>8}")
    totals = collections.Counter()
    boundary = collections.Counter()
    for work in sorted(works):
        units = sorted(works[work], key=lambda ps: sort_key(ps[1].sid))
        gold = "".join(s.text for _, s in units)
        stream = kanripo_stream(args.kanripo, work)
        if stream is None:
            print(f"{work:<12} {len(units):>7} {len(gold):>8}  (no kanripo source)")
            continue
        bare, marks = stream
        amap = align(gold, bare)
        nalign = sum(1 for x in amap if x is not None)

        # gold character offset -> (unit index, offset within unit)
        starts, run = [], 0
        for _, s in units:
            starts.append(run)
            run += len(s.text)
        placements = collections.defaultdict(list)
        placed = 0
        wbound = collections.Counter()          # this work only — the global one is for the table
        for bidx, mark in marks:
            # the mark precedes bare[bidx]; place it before the gold character that matched
            g = amap[bidx] if bidx < len(amap) else None
            if g is None:                       # mark at the very end, or over a divergence
                g = amap[bidx - 1] + 1 if bidx and bidx - 1 < len(amap) and amap[bidx - 1] is not None else None
            if g is None:
                continue
            u = max(0, min(len(starts) - 1, _bisect(starts, g)))
            # A mark falling EXACTLY on a unit boundary belongs to the unit it CLOSES, not the one
            # it precedes. `_bisect` alone puts it at offset 0 of the following unit, which made
            # every 。？！ the FIRST token of the next unit — 2 780 of them, none trailing — and so
            # pushed `sent_final`/`sent_group` one unit late, mis-segmenting every merged sentence.
            # The round-trip check could not catch it: the text and the character offsets were
            # right, only the OWNERSHIP was wrong. An OPENING mark is the exception, since 「 does
            # belong to what follows it.
            if g == starts[u] and u > 0 and mark not in OPENERS:
                u -= 1
            placements[u].append((g - starts[u], mark))
            placed += 1
            if g == starts[u] or g - starts[u] == len(units[u][1].text):
                wbound[mark] += 1
                boundary[mark] += 1

        n_ins = 0
        for u, (_, s) in enumerate(units):
            if placements.get(u):
                n_ins += insert_marks(s, sorted(placements[u]))
        # Record the grouping the punctuation implies, WITHOUT acting on it: units that a
        # sentence-final mark does not separate belong to one sentence, and merging them needs a
        # relation between their roots that Kyoto never annotated. `# sent_group` numbers the runs
        # so that work can start from here; `# sent_final` marks the unit that closes each run.
        # INVARIANT: a sentence-final mark must never open a unit — it closes the unit before it.
        # This is checked rather than assumed because the round-trip test cannot see it (the text
        # is identical either way) and everything downstream of `sent_final` silently depends on it.
        lead = sum(1 for _, s in units if s.toks and s.toks[0][3] == "PUNCT"
                   and any(c in SENT_FINAL for c in s.toks[0][1]))
        if lead:
            print(f"   *** {work}: {lead} units OPEN with a sentence-final mark — ownership is "
                  f"off by one and sent_group will be wrong ***")

        group = 0
        for _, s in units:
            final = any(t[1] and t[1][-1] in SENT_FINAL for t in s.toks if t[3] == "PUNCT")
            s.comments.append(f"# sent_group = {work}_{group}")
            s.comments.append(f"# sent_final = {'yes' if final else 'no'}")
            if final:
                group += 1
        nb = sum(wbound.values())
        print(f"{work:<12} {len(units):>7} {len(gold):>8} {100*nalign/max(1,len(bare)):>7.1f}% "
              f"{len(marks):>7} {n_ins:>7} {100*nb/max(1,placed):>7.1f}%")
        totals["units"] += len(units); totals["marks"] += len(marks); totals["placed"] += n_ins

    print(f"\n{'TOTAL':<12} {totals['units']:>7} units, {totals['marks']:>7} marks in source, "
          f"{totals['placed']:>7} inserted "
          f"({100*totals['placed']/max(1,totals['marks']):.1f} %)")
    print("\nmarks landing on a 句讀 unit boundary, by type:")
    for c, n in boundary.most_common(12):
        kind = "sentence-final" if c in SENT_FINAL else ("opening" if c in OPENERS else "pause")
        print(f"   {c}  {n:>7}   {kind}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    for p, sents in per_file.items():
        out = p.replace(".conllu", args.out_suffix + ".conllu")
        for s in sents:
            s.comments = [c for c in s.comments if not c.startswith("# text =")]
            s.comments.append("# text = " + s.text)
        write_conllu(out, sents)
        print(f"wrote {out}  ({len(sents)} sentences)")


def _bisect(starts, g):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= g:
            lo = mid
        else:
            hi = mid - 1
    return lo


if __name__ == "__main__":
    main()
