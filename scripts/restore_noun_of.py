#!/usr/bin/env python3
"""Give every noun-level `of` back the DEPREL the SUD treebank itself assigns.

WHY. The pristine SUD English treebanks leave an `of`-PP under a noun underspecified: `udep` on
2,908 of 2,927 noun-level `of`s in EWT train and 4,075 of 4,149 in GUM. The extended relabel
(`relabel_ext.py`, bucket `adp_NOUN`/`adp_PROPN`) then asked qwen3:8b, token by token, whether each one
was a complement or a modifier -- with a prompt and twelve shots written for VERBS, and with the whole
head-noun subtree in the prompt, so the same noun got a different question in every sentence. The
result split the same head noun almost evenly (part of 56 comp:obl / 43 mod, member of 28/21, end
of 29/21, copy of 13/13 in EWT+GUM train), and the decision tracked irrelevant context (PP length,
the head's own relation). The clausal ones (a VERB in the PP) were never asked at all and were later
defaulted to `mod` by `resolve_udep.py`'s (NOUN, ADP) majority. By user decision (2026-10-25) the
relabel no longer touches these: they carry whatever the treebank says.

WHAT. A DEPREL-only rewrite. A token qualifies if it is ADP with lemma `of`, its head is NOUN or
PROPN, and the pristine treebank has the same token (sent_id + id + form) with the same head and
the DEPREL `udep`; its DEPREL becomes `udep` again. (The treebank's handful of non-`udep` noun-level
`of`s were never relabelled and are left as they are.) Every other byte is untouched. Idempotent. `relabel_ext.py` and `resolve_udep.py` now skip the same tokens,
so a full rebuild reproduces what this script does to an existing file; this script exists so the
derived files on disk can be brought into line without re-running the chain.

    restore_noun_of.py FILE... [--inplace | --suffix .X]  [--pristine P.conllu ...]
"""
import argparse
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PRISTINE = sorted(str(p) for p in (HERE / "assets/SUD_English-EWT").glob("en_ewt-sud-*.conllu")) + \
                   sorted(str(p) for p in (HERE / "assets/SUD_English-GUM").glob("en_gum-sud-*.conllu"))
NOUNS = ("NOUN", "PROPN")


def is_noun_of(f, rows):
    """f: the 10 CoNLL-U columns; rows: id -> columns of the same sentence."""
    if f[3] != "ADP" or f[2].lower() != "of":
        return False
    h = rows.get(f[6])
    return h is not None and h[3] in NOUNS


def load_pristine(paths):
    gold = {}
    for p in paths:
        sid = None
        for line in open(p, encoding="utf-8"):
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line[:1].isdigit():
                f = line.rstrip("\n").split("\t")
                if len(f) == 10 and f[0].isdigit():
                    gold[(sid, f[0])] = (f[1], f[6], f[7])
    return gold


def restore(text, gold, stats):
    out = []
    for blk in text.split("\n\n"):
        lines = blk.split("\n")
        sid = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("# sent_id")), None)
        rows = {}
        for l in lines:
            if l and not l.startswith("#"):
                f = l.split("\t")
                if len(f) == 10 and f[0].isdigit():
                    rows[f[0]] = f
        for i, l in enumerate(lines):
            if not l or l.startswith("#"):
                continue
            f = l.split("\t")
            if len(f) != 10 or not f[0].isdigit() or not is_noun_of(f, rows):
                continue
            g = gold.get((sid, f[0]))
            if g is None or g[0] != f[1]:
                stats["no pristine token"] += 1
                continue
            if g[1] != f[6]:
                stats["head differs from pristine (left alone)"] += 1
                continue
            if g[2] != "udep":
                # Only `udep` was ever relabelled. The treebank's few other labels here are already
                # in place, bar normalise_reparandum's deliberate reparandum -> conj:dicto.
                stats[f"pristine {g[2]} (left alone)"] += 1
                continue
            if f[7] != g[2]:
                stats[f"{f[7]} -> {g[2]}"] += 1
                f[7] = g[2]
                lines[i] = "\t".join(f)
            else:
                stats["already pristine"] += 1
        out.append("\n".join(lines))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pristine", nargs="*", default=DEFAULT_PRISTINE)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inplace", action="store_true")
    g.add_argument("--suffix", help="write FILE with .conllu replaced by SUFFIX.conllu")
    a = ap.parse_args()
    gold = load_pristine(a.pristine)
    if not gold:
        sys.exit("no pristine tokens loaded")
    for p in a.files:
        stats = collections.Counter()
        text = open(p, encoding="utf-8").read()
        new = restore(text, gold, stats)
        dst = p if a.inplace else p.replace(".conllu", f"{a.suffix}.conllu")
        if new != text or not a.inplace:
            open(dst, "w", encoding="utf-8").write(new)
        print(f"{dst}: " + ", ".join(f"{k} {v}" for k, v in stats.most_common()))


if __name__ == "__main__":
    main()
