#!/usr/bin/env python
"""Re-render PROIEL's and Perseus's XPOS in ITTB's tagset, in place, DEPREL-style.

The Latin parser trains on a plain concatenation of three treebanks whose XPOS tagsets have
nothing to do with one another (see build_la_xpos_map.py).  Until now the mismatch was handled
by DELETION: blank_perseus_xpos.py blanked field 5 on the Perseus tail because its sparse
9-position tagset dragged the combined TAG metric down, and PROIEL's 23-value tagset was simply
left to sit beside ITTB's 1 914 composite codes.  That leaves the tagger predicting two
conventions at once over one third of the data and nothing at all over another twentieth.

This normalises instead: ITTB is the largest treebank (390 787 train tokens against PROIEL's
177 558 and Perseus's 18 259), so its conventions win, its own rows are left BYTE-IDENTICAL,
and the other two are re-rendered from their own (form, lemma, UPOS, FEATS) through the map
harvested from ITTB.  Held out on ITTB itself that rendering reproduces the treebank's own gold
XPOS 93.7-94.0 % exactly, which is the honest ceiling on the tags manufactured here.

Which treebank a sentence belongs to is read off its sent_id -- ITTB writes `train-s1`, PROIEL
a bare number, Perseus `phi0975.phi001.perseus-lat1.tb.xml@226` -- rather than off a sentence
COUNT, as blank_perseus_xpos.py did.  A count is silently wrong the moment a split is rebuilt
in a different order or a treebank changes size; an unrecognised sent_id here is an error.

Field 5 is the only column written.  The script re-reads its own output and refuses to keep it
unless every other column, every comment line and every range/empty-node line is unchanged --
the same discipline the DEPREL rewriters hold, and worth having because the FORM column is what
the macron and augmentation work varies and must not be touched.  Idempotent: the rendering
reads only columns this script never writes.

    normalise_la_xpos.py <file.conllu> [more.conllu ...] [--map assets_la/la_xpos_map.json]
                                       [--dry-run]
"""
import argparse, collections, json, re, sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from build_la_xpos_map import Renderer, rows  # noqa: E402

ITTB_SENT = re.compile(r"^(train|dev|test)-s\d+$")


def treebank_of(sent_id):
    if "perseus" in sent_id:
        return "perseus"
    if sent_id.isdigit():
        return "proiel"
    if ITTB_SENT.match(sent_id):
        return "ittb"
    return None


def sent_id_of(block):
    for line in block.split("\n"):
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return None


def normalise(path, renderer, dry_run=False):
    raw = open(path, encoding="utf-8").read()
    blocks = raw.split("\n\n")
    stats = collections.Counter()
    tags = collections.Counter()
    out_blocks = []
    for block in blocks:
        if not block.strip():
            out_blocks.append(block)
            continue
        sid = sent_id_of(block)
        tb = treebank_of(sid) if sid is not None else None
        if tb is None:
            raise SystemExit(f"{path}: unrecognised sent_id {sid!r} -- refusing to guess "
                             f"which treebank it belongs to")
        stats[f"sents_{tb}"] += 1
        if tb == "ittb":
            out_blocks.append(block)
            stats["tokens_ittb"] += sum(1 for ln in block.split("\n")
                                        if "\t" in ln and ln.split("\t")[0].isdigit())
            continue
        lines = []
        for line in block.split("\n"):
            cols = line.split("\t")
            if len(cols) != 10 or not cols[0].isdigit():
                lines.append(line)
                continue
            stats[f"tokens_{tb}"] += 1
            new, tail_src, letter_src = renderer.render(cols[1], cols[2], cols[3], cols[5])
            stats[f"letter_{letter_src}"] += 1
            stats[f"tail_{tail_src}"] += 1
            stats["changed"] += new != cols[4]
            tags[new] += 1
            cols[4] = new
            lines.append("\t".join(cols))
        out_blocks.append("\n".join(lines))

    new_raw = "\n\n".join(out_blocks)
    verify(raw, new_raw, path)
    if not dry_run:
        open(path, "w", encoding="utf-8").write(new_raw)
    return stats, tags


def verify(old, new, path):
    """Nothing but field 5 on token rows may differ."""
    o, n = old.split("\n"), new.split("\n")
    if len(o) != len(n):
        raise SystemExit(f"{path}: line count changed {len(o)} -> {len(n)}")
    for i, (a, b) in enumerate(zip(o, n), 1):
        if a == b:
            continue
        ca, cb = a.split("\t"), b.split("\t")
        if len(ca) != 10 or len(cb) != 10 or not ca[0].isdigit():
            raise SystemExit(f"{path}:{i}: non-token line changed:\n  {a!r}\n  {b!r}")
        if [x for j, x in enumerate(ca) if j != 4] != [x for j, x in enumerate(cb) if j != 4]:
            raise SystemExit(f"{path}:{i}: a column other than XPOS changed:\n  {a!r}\n  {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--map", default="assets_la/la_xpos_map.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    renderer = Renderer(json.load(open(a.map, encoding="utf-8")))
    total = collections.Counter()
    all_tags = collections.Counter()
    for path in a.files:
        stats, tags = normalise(path, renderer, a.dry_run)
        total.update(stats)
        all_tags.update(tags)
        print(f"{path}: ittb {stats['tokens_ittb']} kept, "
              f"proiel {stats['tokens_proiel']} + perseus {stats['tokens_perseus']} rendered "
              f"({stats['changed']} cells changed)")
    print(f"\ntotal: rendered {total['tokens_proiel'] + total['tokens_perseus']} tokens into "
          f"{len(all_tags)} distinct ITTB-form tags")
    print("  letter source:", {k[7:]: f"{100*v/max(sum(x for j,x in total.items() if j.startswith('letter_')),1):.1f}%"
                               for k, v in total.items() if k.startswith("letter_")})
    print("  tail source:  ", {k[5:]: f"{100*v/max(sum(x for j,x in total.items() if j.startswith('tail_')),1):.1f}%"
                               for k, v in total.items() if k.startswith("tail_")})
    if a.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
