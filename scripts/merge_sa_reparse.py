#!/usr/bin/env python3
"""Merge SUD_Sanskrit-Vedic clause UNITS into sentences, taking the inter-unit link from a RE-PARSE.

WHY MERGE AT ALL. The treebank segments text into short punctuation-free clause units and carries no
in-text sentence boundaries, so a parser trained on it never sees more than one clause and cannot
segment running text — which is the entire reason `clause_parser` exists, reconstructing at
inference something training removed. The grouping is recoverable: a Vedic `sent_id` is
`<DCS sentence id>_<clause index>`, so units of one sentence share a base id. Of 17 100 sentences
over 21 477 units, 83.6 % are single-unit, 11.7 % two, 2.6 % three, 2.1 % four or more — so 16.4 %
merge, averaging 14.2 tokens against 8.5.

WHY THE LINK COMES FROM A RE-PARSE AND NOT FROM A CONSTANT. The treebank annotates each unit's tree
and says NOTHING about how one unit relates to the next, so merging must supply that arc. An earlier
version of this script supplied a constant `parataxis`, on the grounds that it is UD/SUD's relation
for juxtaposed clauses without a coordinator and that it already chains here (a parataxis dependent's
head is itself parataxis 516 times). Re-parsing the concatenated units says that was wrong in the
majority of cases. Measured over 1 200 multi-unit sentences, 1 774 boundaries:

    stayed inside its own unit   705   39.7 %      conj:coord   436   24.6 %
    parataxis                    354   20.0 %      comp:obj      75    4.2 %
    subj                          50    2.8 %      mod@relcl     32    1.8 %

and where it did link, it attached to the PREVIOUS UNIT'S ROOT 861 of 1 069 times (80.5 %). So the
commonest choice is `conj:coord`, not `parataxis` — defensible, since parallel elliptical sūtra
clauses are arguably coordinate, and `conj:coord` chains here too (3 854 instances).

⚠ 40 % OF BOUNDARIES ARE LEFT UNLINKED, DELIBERATELY. Where the parser declines to attach across the
boundary, this script leaves the later unit self-headed, so that sentence keeps MORE THAN ONE ROOT.
That is not a failure to merge: a multi-root doc is a multi-sentence example, which is exactly what
`sud.GoldTokCorpus.v1` is built to train on, and asserting a link the model declines to draw would
be fabricating structure in the 40 % of cases with the least evidence for it.

⚠ CIRCULARITY, STATED PLAINLY. The parser never saw these boundaries in training, so it is
extrapolating from the clause relations it learned WITHIN units. Training on its output makes its
current preferences self-confirming, and a `conj:coord` or `parataxis` score on the result partly
measures this bootstrapping rather than the language. Gold intra-unit trees are never overwritten —
only the boundary arc comes from the model — but any claim resting on those two labels should say
where they came from.

PUNCTUATION (--punct). The treebank realises no punctuation at all — 0 PUNCT tokens, 0 `punct` arcs
— but records it: `Punctuation=fullStop` on 24 650 tokens and `Punctuation=comma` on 8 394, marking
where a mark FOLLOWS that token (fullStop is unit-final 19 370 times and mid-unit 5 280; comma is
mid-unit 8 129 times). `clause_parser` exists partly because of that absence: a stray daṇḍa gets
tagged as a noun or verb and can become the clause root, so it strips marks before parsing.

WHICH MARK IS WHICH. Sanskrit writes a single daṇḍa `।` at the half-verse and a double daṇḍa `॥` at
the end of the verse, and that is exactly the distinction DCS records, so the mapping is EDITORIAL,
not syntactic:

    ॥  every `Punctuation=fullStop`   (written `‖` U+2016 — see SINGLE/DOUBLE below)
    ।  every `Punctuation=comma`      (written `|`)

A syntactic mapping was written first and discarded: it demoted a fullStop internal to a merged
sentence to `।`, so a `॥` would never once have appeared mid-sentence in training — while a user
pastes verses whose `॥` falls wherever the metre puts it, merged sentence or not. That is hazard 10
(ask the model what regime its input is in) arriving through the training data instead. The
editorial mapping keeps the two distributions the same, and lets the parser learn the one thing that
matters here, which is that a `॥` can be crossed.

Two consequences of being faithful: 28 dev sentences end in a single `।` (their last recorded mark
is a comma — a half-verse boundary that the sentence genuinely ends on), and 229 end with no mark.

ATTACHMENT follows `clause_parser`'s own convention rather than inventing another: a mark is a
`punct` child of the root of the unit it closes. A mark recorded on a token INSIDE an MWT is emitted
after the last member of that MWT — it follows the whole surface word, and a range row may not span
it.

    merge_sa_reparse.py MODEL IN.conllu OUT.conllu [--punct] [--max-len 80]
"""
import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conllu_misc import misc_dict  # noqa: E402
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402


def read(path):
    sent, com = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if sent:
                yield com, sent
            sent, com = [], []
        elif line.startswith("#"):
            com.append(line)
        else:
            sent.append(line.split("\t"))
    if sent:
        yield com, sent


# The mark FORMS are the ones the sa tokeniser actually emits, not the Devanagari characters: it
# transliterates to CSL and normalises every double daṇḍa to U+2016, so ।/॥, |/||, and their
# Devanagari originals all arrive at the parser as `|` / `‖`. Writing ।/॥ into the corpus would
# train the parser on two tokens it can never be shown at inference (hazard 10, via the data).
SINGLE, DOUBLE = "|", "\u2016"


def _insert_danda(rows, roots, spans, linked):
    """Realise the marks recorded in MISC as daṇḍa tokens and renumber. See PUNCTUATION above.

    Row order (including the ``n-m`` MWT range rows, which precede their members) is preserved. A
    mark goes after the row of the token it follows — or, when that token is inside an MWT, after
    the last member of that MWT, since the mark follows the whole surface word.
    """
    real = [t for t in rows if "-" not in t[0] and "." not in t[0]]
    oldid = {t[0]: i for i, t in enumerate(real)}
    mwt_end = {}                    # 0-based index of an MWT member -> index of its last member
    for t in rows:
        if "-" in t[0]:
            lo, hi = t[0].split("-")
            for i in range(oldid[lo], oldid[hi] + 1):
                mwt_end[i] = oldid[hi]
    unit_of = {}
    for ui, (lo, hi) in enumerate(spans):
        for i in range(lo, hi):
            unit_of[i] = ui
    marks = {}                      # 0-based index of the row a mark FOLLOWS -> (form, root idx)
    for i, t in enumerate(real):
        misc = misc_dict(t[9])
        kind = misc.get("Punctuation")
        if not kind:
            continue
        at = mwt_end.get(i, i)
        marks[at] = (DOUBLE if kind == "fullStop" else SINGLE, roots[unit_of.get(at, 0)])
    if not marks:
        return rows
    # new id for each real token: its rank in the sequence once the marks are interleaved
    newid, k = {}, 0
    for i in range(len(real)):
        k += 1
        newid[i] = k
        if i in marks:
            k += 1
    out, seen = [], 0
    for t in rows:
        t = list(t)
        if "-" in t[0]:                                     # MWT range row: remap both endpoints
            lo, hi = t[0].split("-")
            t[0] = f"{newid[oldid[lo]]}-{newid[oldid[hi]]}"
            out.append(t)
            continue
        if "." in t[0]:                                     # empty node: left as-is
            out.append(t)
            continue
        i = seen
        seen += 1
        t[0] = str(newid[i])
        if t[6] != "0":
            t[6] = str(newid[oldid[t[6]]])
        out.append(t)
        if i in marks:
            form, root_i = marks[i]
            out.append([str(newid[i] + 1), form, form, "PUNCT", "_", "_",
                        str(newid[root_i]), "punct", "_", "_"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--punct", action="store_true",
                    help="realise the recorded marks as daṇḍa tokens (see PUNCTUATION below)")
    ap.add_argument("--max-len", type=int, default=80,
                    help="skip merging a sentence longer than this; the parser degrades on very "
                         "long input and a bad boundary arc is worse than none")
    a = ap.parse_args()
    nlp = spacy.load(a.model)

    groups = collections.OrderedDict()
    for com, sent in read(a.inp):
        s = next((c.split("=", 1)[1].strip() for c in com if c.startswith("# sent_id")), "")
        m = re.match(r"(.*)_(\d+)$", s)
        base, idx = (m.group(1), int(m.group(2))) if m else (s, 1)
        groups.setdefault(base, []).append((idx, com, sent))

    stats = collections.Counter()
    rels = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for base, units in groups.items():
            units.sort(key=lambda u: u[0])
            if len(units) == 1:
                stats["single unit"] += 1
                com, sent = units[0][1], units[0][2]
                if a.punct:
                    real = [t for t in sent if "-" not in t[0] and "." not in t[0]]
                    rts = [i for i, t in enumerate(real) if t[6] == "0"]
                    sent = _insert_danda(sent, [rts[0] if rts else 0],
                                         [(0, len(real))], [False])
                for c in com:
                    f.write(c + "\n")
                for t in sent:
                    f.write("\t".join(t) + "\n")
                f.write("\n")
                continue

            # --- renumber into one token sequence, gold trees untouched -------------------------
            out, offset, texts = [], 0, []
            roots, spans = [], []
            for idx, com, sent in units:
                unit_roots = []
                texts += [c[9:].strip() for c in com if c.startswith("# text = ")]
                real = [t for t in sent if "-" not in t[0] and "." not in t[0]]
                nid = {t[0]: str(offset + k + 1) for k, t in enumerate(real)}
                start = offset
                for t in sent:
                    t = list(t)
                    if "-" in t[0]:
                        lo, hi = t[0].split("-")
                        t[0] = f"{nid[lo]}-{nid[hi]}"
                    else:
                        if t[6] == "0":
                            unit_roots.append(int(nid[t[0]]) - 1)
                        else:
                            t[6] = nid[t[6]]
                        t[0] = nid[t[0]]
                    out.append(t)
                offset += len(real)
                spans.append((start, offset))
                roots.append(unit_roots[0] if unit_roots else start)

            linked = [False] * len(units)
            real_out = [t for t in out if "-" not in t[0] and "." not in t[0]]
            if len(real_out) <= a.max_len:
                doc = Doc(nlp.vocab, words=[t[1] for t in real_out],
                          spaces=[True] * (len(real_out) - 1) + [False])
                for tok, t in zip(doc, real_out):
                    misc = misc_dict(t[9])
                    tok.norm_ = misc.get("Unsandhied") or t[1]
                    if misc.get("Compound") or "Compound=Yes" in t[5]:
                        tok.set_morph("Compound=Yes")
                pred = nlp(doc)
                by_id = {int(t[0]) - 1: t for t in real_out}
                for ui in range(1, len(units)):
                    r = roots[ui]
                    h = pred[r].head.i
                    if h < spans[ui][0]:                    # attached into an EARLIER unit
                        by_id[r][6] = str(h + 1)
                        by_id[r][7] = pred[r].dep_
                        rels[pred[r].dep_] += 1
                        linked[ui] = True
                        stats["linked by re-parse"] += 1
                    else:
                        by_id[r][6], by_id[r][7] = "0", "root"
                        stats["left unlinked (own root kept)"] += 1
            else:
                stats["too long, left unlinked"] += 1
                for ui in range(1, len(units)):
                    pass
            # every root that was not re-attached keeps head 0 / root
            for t in out:
                if "-" not in t[0] and "." not in t[0] and t[6] not in ("0",) and not t[6].isdigit():
                    t[6], t[7] = "0", "root"
            if a.punct:
                out = _insert_danda(out, roots, spans, linked)
            stats["merged"] += 1
            f.write(f"# sent_id = {base}\n")
            f.write("# text = " + " ".join(texts) + "\n")
            f.write(f"# merged_units = {len(units)}\n")
            for t in out:
                f.write("\t".join(t) + "\n")
            f.write("\n")

    for k, v in stats.most_common():
        print(f"  {k:<32}{v}")
    print("  relations chosen by the re-parse: " +
          ", ".join(f"{k} {v}" for k, v in rels.most_common(6)))


if __name__ == "__main__":
    main()
