#!/usr/bin/env python3
"""Recombine Kyoto's 句讀 units into sentences BY THE SAME RULE the arm applies at inference.

WHY THIS IS THE HIGHEST-VALUE CHANGE AVAILABLE. The lzh parser's error decomposition says a THIRD
of its errors are root decisions — 988 tokens gold attaches that the parser makes a ROOT, 679 the
reverse, 33.6 % of all errors between them — while non-projectivity costs 3.4 % and adjacent arcs
carry the mass that extra context cannot help. The parser is not confused about which word governs
which; it is confused about where a sentence ends. And it is confused because THE TRAINING DATA
NEVER SETTLED IT: Kyoto annotates one 句讀 unit per block with no cross-unit arcs at all, and
`cross_unit_rules.py` fills only the 37.1 % of boundaries it can derive at >= 90 % dominance,
leaving the rest as sentence breaks.

So instead of patching the convention on at inference with `sent_join`, apply the SAME rule to the
training data and let the parser learn it. The rule is not reimplemented here — this builds a Doc
per kanripo PARAGRAPH with the gold trees in it, each block its own sentence, and runs the actual
`SentJoin` pipe over it. One implementation, two uses.

⚠ **PARAGRAPH IS THE GROUPING, and it is load-bearing.** `sent_id` is
`KR1h0004_001_par1_12-17` — work, section, paragraph, character range. `pause_join` never breaks at
a comma, so feeding it a whole work would chain every comma-final unit in the file into one
sentence. Blocks are only ever joined within one paragraph.

⚠ **THIS REWRITES THE TEST GOLD TOO**, which is the standing trap of every relabelling in this repo
(docs/udep-relabel.md: "each relabel rewrites the test gold too, so comp:obl F has a moving
denominator"). An arm trained on the merged corpus and scored on the merged test is NOT comparable
to 76.46 LAS. Score both arms on both corpora and say which is which.

Usage:
    merge_lzh_clauses.py --in  assets_lzh/.../lzh_kyoto-sud-train.<...>.punct.conllu \
                         --out assets_lzh/.../lzh_kyoto-sud-train.<...>.punct.sjmerged.conllu
"""
import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def read_blocks(path):
    """(sent_id, comment lines, rows) per block; rows are the 10 CoNLL-U columns."""
    out, cur, com, sid = [], [], [], None
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                out.append((sid, com, cur))
            cur, com, sid = [], [], None
            continue
        if line.startswith("#"):
            com.append(line)
            m = re.match(r"#\s*sent_id\s*=\s*(\S+)", line)
            if m:
                sid = m.group(1)
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f)
    if cur:
        out.append((sid, com, cur))
    return out


def paragraph(sid):
    """`KR1h0004_001_par1_12-17` -> `KR1h0004_001_par1`; anything unparsable is its own group."""
    if not sid:
        return None
    return re.sub(r"_\d+(-\d+)?$", "", sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joins", default=None)
    ap.add_argument("--no-pause-join", action="store_true")
    ap.add_argument("--max-sent", type=int, default=100,
                    help="stop joining rather than emit a sentence longer than this (0 = no limit)")
    a = ap.parse_args()

    import seg_code  # noqa: F401  (registers the lzh language)
    import spacy
    from spacy.tokens import Doc
    from sent_join import SentJoin

    nlp = spacy.blank("lzh")
    pipe = SentJoin(pause_join=not a.no_pause_join, max_sent=a.max_sent)
    if a.joins:
        pipe.load_joins(a.joins)

    blocks = read_blocks(a.inp)
    groups = collections.OrderedDict()
    for k, (sid, com, rows) in enumerate(blocks):
        groups.setdefault(paragraph(sid) or f"__{k}", []).append((sid, com, rows))
    print(f"{len(blocks)} blocks in {len(groups)} paragraphs", flush=True)

    out_lines, n_out, joined = [], 0, 0
    branch = collections.Counter()
    for gid, members in groups.items():
        words, tags, poss, heads, deps, extra = [], [], [], [], [], []
        off = 0
        for sid, com, rows in members:
            for i, f in enumerate(rows):
                words.append(f[1])
                tags.append(f[4] if f[4] != "_" else "")
                poss.append(f[3] if f[3] != "_" else "")
                h = int(f[6])
                heads.append(off + (i if h == 0 else h - 1))
                deps.append(f[7] if f[7] != "_" else "dep")
                extra.append((f[2], f[5], f[9]))          # LEMMA, FEATS, MISC carried verbatim
            off += len(rows)
        doc = Doc(nlp.vocab, words=words, spaces=[False] * len(words),
                  heads=heads, deps=deps, tags=tags, pos=poss)
        pipe.debug = []
        doc = pipe(doc)
        for r in pipe.debug:
            branch[r["branch"]] += 1
        # ⚠ EMIT-TIME INVARIANT. A CoNLL-U block must have exactly one root. `doc.sents` is spaCy's
        # view and this code has already been bitten once by it disagreeing with the tree, so the
        # spans are re-split at every additional root rather than trusted.
        sents = []
        for sent in doc.sents:
            roots = [t.i for t in sent if t.head.i == t.i]
            if len(roots) <= 1:
                sents.append(sent)
                continue
            bounds = [sent.start] + roots[1:] + [sent.end]
            for x, y in zip(bounds, bounds[1:]):
                if y > x:
                    sents.append(doc[x:y])
        joined += len(members) - len(sents)
        for si, sent in enumerate(sents):
            n_out += 1
            out_lines.append(f"# sent_id = {gid}_sj{si+1}")
            out_lines.append(f"# text = {''.join(t.text for t in sent)}")
            base = sent.start
            for t in sent:
                lem, feats, misc = extra[t.i]
                head = 0 if t.head.i == t.i else (t.head.i - base + 1)
                dep = "root" if head == 0 else t.dep_
                out_lines.append("\t".join([
                    str(t.i - base + 1), t.text, lem, t.pos_ or "_", t.tag_ or "_", feats,
                    str(head), dep, "_", misc]))
            out_lines.append("")
    pipe.debug = None
    pathlib.Path(a.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {n_out} sentences ({joined} block joins)")
    print("  branches used:", ' '.join(f'{k}:{v}' for k, v in branch.most_common()))


if __name__ == "__main__":
    main()
