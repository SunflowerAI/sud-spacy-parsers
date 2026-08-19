#!/usr/bin/env python3
"""Merge SUD_Sanskrit-Vedic clause UNITS back into the sentences they came from.

WHY. The treebank segments text into short punctuation-free clause units and carries no in-text
sentence boundaries, so a parser trained on it never sees more than one clause and cannot segment
running text — which is the entire reason `clause_parser` exists, reconstructing at inference
something training deliberately removed. But the grouping is not lost: a Vedic `sent_id` is
`<DCS sentence id>_<clause index>`, so units of one sentence share a base id.

HOW MUCH IS THERE. 17 100 DCS sentences over 21 477 units: 83.6 % are a single unit, 11.7 % two,
2.6 % three, 2.1 % four or more. So 16.4 % of sentences merge, and they average 14.2 tokens against
8.5 for the single-unit ones. The gain is bounded by that 16.4 %.

⚠ THE LINK BETWEEN UNITS IS INFERRED, NOT GOLD. The treebank annotates each unit's tree and says
NOTHING about how one unit relates to the next — merging therefore has to supply a relation, and
supplying one is fabricating annotation. `parataxis` is the least-bad choice and the choice the
treebank's own usage supports: it is UD/SUD's relation for juxtaposed clauses with no coordinator,
which is exactly what these are (sūtra lists like `dakṣiṇataḥ kapardā vasiṣṭhānām / ubhayato
'tri-bhārgava-kāśyapānām / ...`), and it is already chained here — a parataxis dependent's head is
itself parataxis 516 times, more than any other relation, mirroring `conj:coord`'s 3 854.
So each unit's root attaches to the PREVIOUS unit's root as `parataxis`, leaving exactly one root.

NB this is a different decision from the one recorded for `clause_parser`, which declines to
fabricate parataxis at INFERENCE and re-parses instead. Providing a gold link at training time and
inventing one at run time are not the same act — but this is still an inferred link, and any result
resting on it should say so.

    merge_sa_clauses.py IN.conllu OUT.conllu [--report]
"""
import argparse
import collections
import re


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


def sid(com):
    for c in com:
        if c.startswith("# sent_id"):
            return c.split("=", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    groups = collections.OrderedDict()
    for com, sent in read(a.inp):
        s = sid(com) or ""
        m = re.match(r"(.*)_(\d+)$", s)
        base, idx = (m.group(1), int(m.group(2))) if m else (s, 1)
        groups.setdefault(base, []).append((idx, com, sent))

    merged = kept = links = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for base, units in groups.items():
            units.sort(key=lambda u: u[0])
            if len(units) == 1:
                kept += 1
                com, sent = units[0][1], units[0][2]
                for c in com:
                    f.write(c + "\n")
                for t in sent:
                    f.write("\t".join(t) + "\n")
                f.write("\n")
                continue
            merged += 1
            out, offset, prev_root, texts = [], 0, None, []
            for idx, com, sent in units:
                for c in com:
                    if c.startswith("# text = "):
                        texts.append(c[9:].strip())
                # renumber: only real tokens carry ids we re-index; MWT ranges are rewritten too
                nid = {}
                real = [t for t in sent if "-" not in t[0] and "." not in t[0]]
                for k, t in enumerate(real):
                    nid[t[0]] = str(offset + k + 1)
                root_new = None
                for t in sent:
                    t = list(t)
                    if "-" in t[0]:
                        lo, hi = t[0].split("-")
                        t[0] = f"{nid[lo]}-{nid[hi]}"
                    else:
                        t[0] = nid[t[0]]
                        if t[6] == "0":
                            root_new = t[0]
                            if prev_root is None:
                                t[6], t[7] = "0", "root"
                            else:
                                t[6], t[7] = prev_root, "parataxis"
                                links += 1
                        else:
                            t[6] = nid[t[6]]
                    out.append(t)
                prev_root = root_new
                offset += len(real)
            f.write(f"# sent_id = {base}\n")
            f.write("# text = " + " ".join(texts) + "\n")
            f.write(f"# merged_units = {len(units)}\n")
            for t in out:
                f.write("\t".join(t) + "\n")
            f.write("\n")
    print(f"  single-unit sentences kept {kept}; merged {merged}; parataxis links added {links}")


if __name__ == "__main__":
    main()
