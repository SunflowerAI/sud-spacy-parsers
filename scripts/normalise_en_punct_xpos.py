#!/usr/bin/env python
"""Normalise the EWT half of the merged English corpus onto GUM's punctuation XPOS convention.

EWT and GUM share one PTB tagset -- 49 tags against 46, GUM's a strict subset -- and agree on
every open and closed word class: of 688 (form, UPOS) types frequent in both, only 17 disagree
on their majority tag, and all but a handful of those are genuine ambiguity rather than
convention (`know` VB/VBP, `her` PRP/PRP$, `got` VBD/VBN -- web text and edited prose really do
differ in how often each reading occurs).

The exception is punctuation, and there it is a flat convention conflict.  PTB reserves `,` for
the comma and gives dashes, semicolons, colons and ellipses the tag `:`.  GUM follows that
without exception; EWT tags `;` as `,` 101 times out of 101, `--` as `,` 123 times, `...` 159,
`/` 142.  So the same character in the same context carries different gold depending on which
treebank the sentence came from, and the tagger cannot win either way.

Direction is GUM's, by user decision (2026-08-10), and it is worth recording that this is NOT
the "normalise towards the largest treebank" rule: in the arm that ships -- the merge with
GUM's five NonCommercial genres dropped -- EWT is the LARGER half, 204 578 tokens against
135 746.  The rule is overridden here because the disagreement is not a house style but a
standard: `,` is defined as the comma tag, GUM is consistent with that definition, and EWT is
the outlier.  Making `,` mean comma in both is what the merged tagset needs.

The mapping is DERIVED from the GUM half of the same file, not hardcoded: a form is rewritten
only where GUM's own usage commits an answer (>= --min-count attestations, dominant).  Two
ellipsis forms GUM never uses (`....`, and `..` where EWT already writes `,`) fall back to the
PTB class rule, declared explicitly below rather than guessed -- a form written only from `.`
or only from dashes is the `:` class by definition.  Anything else is left alone and reported,
so an undecidable case shows up rather than being silently committed.

EWT-only material is NOT touched: `!!`, `*`, `<<`, `:)`, `@` and the rest are web-text tokens
GUM has no opinion about, so there is no conflict to resolve.  Only the `,` conflation is.

Field 5 on EWT rows is the only thing written, and the file is re-read to prove it.  The
EWT-only treebank files are deliberately out of scope: `en_sud_ewt` trains on EWT alone, where
its own convention is internally consistent and its published metrics stand.

    normalise_en_punct_xpos.py <merged.conllu> [more.conllu ...] [--min-count 20] [--dry-run]
"""
import argparse, collections, re, sys

DASHES = set("-‐‑‒–—―")
COLON_CLASS = re.compile(r"^(\.{2,}|…+)$")     # ellipsis written with periods, or U+2026


def spacing(rows, i):
    """A dash is `glued` when it has no space on either side -- i.e. it is inside a compound.

    This is the discriminator the bare form cannot give.  PTB tags `-` HYPH inside `well-known`
    and `:` when it is a dash between clauses, so a rule keyed on the form alone answers with
    GUM's compound-internal majority (HYPH 886) for tokens that are sentence punctuation.  Split
    by spacing, GUM is decisive where it matters: glued `-` is HYPH 593 of 608 (97.5 %), and a
    SPACED dash is `:` -- 266 of 267 for the em dash, 169 of 203 for the en dash.
    """
    glued_after = "SpaceAfter=No" in rows[i][9]
    glued_before = i > 0 and "SpaceAfter=No" in rows[i - 1][9]
    return "glued" if (glued_after and glued_before) else "spaced"


def is_gum(sent_id):
    return sent_id.startswith("GUM")


def sent_id_of(block):
    for line in block.split("\n"):
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return ""


def ptb_class(form, space):
    """The PTB tag a form belongs to by definition, or None if the definition says nothing.

    A SPACED dash of any length is the `:` class by definition, which is what carries the bare
    hyphen: GUM uses `:` on a spaced `-` only by plurality (54 of 91), too thin to commit on its
    own, but the definition and GUM point the same way -- and so does EWT itself, which already
    tags the spaced hyphen `:` 256 times beside the 318 it tags `,`.  So this makes EWT
    self-consistent as much as it aligns it with GUM.
    """
    if COLON_CLASS.match(form):
        return ":"
    if form and all(c in DASHES for c in form) and (len(form) > 1 or space == "spaced"):
        return ":"
    return None


def harvest(blocks, min_count):
    """form -> the tag the GUM half uses for it, where GUM commits one."""
    seen = collections.defaultdict(collections.Counter)
    for b in blocks:
        if not is_gum(sent_id_of(b)):
            continue
        rows = [ln.split("\t") for ln in b.split("\n")
                if len(ln.split("\t")) == 10 and ln.split("\t")[0].isdigit()]
        for i, c in enumerate(rows):
            if c[3] in ("PUNCT", "SYM"):
                seen[(c[1], spacing(rows, i))][c[4]] += 1
                seen[(c[1], None)][c[4]] += 1          # the backed-off, spacing-blind key
    out = {}
    for key, ctr in seen.items():
        n = sum(ctr.values())
        tag, k = ctr.most_common(1)[0]
        if n >= min_count and k / n >= 0.9:
            out[key] = tag
    return out


def normalise(path, table, dry_run):
    raw = open(path, encoding="utf-8").read()
    blocks = raw.split("\n\n")
    changed = collections.Counter()
    undecided = collections.Counter()
    out = []
    for b in blocks:
        if not b.strip() or is_gum(sent_id_of(b)):
            out.append(b)
            continue
        lines = list(b.split("\n"))
        idx = [j for j, ln in enumerate(lines)
               if len(ln.split("\t")) == 10 and ln.split("\t")[0].isdigit()]
        rows = [lines[j].split("\t") for j in idx]
        for i, c in enumerate(rows):
            if not (c[3] in ("PUNCT", "SYM") and c[4] == "," and c[1] != ","):
                continue
            space = spacing(rows, i)
            # (form, spacing) first, then the spacing-blind key, then the class definition.
            # The backoff earns its keep where splitting by spacing fragments unanimous evidence
            # below the count bar rather than discriminating anything: GUM writes `/` SYM 49 of
            # 49 and `?` `.` 593 of 595 whatever the spacing, but the glued halves are only 14
            # and 15 examples.  It does NOT fire for the dash, whose spacing-blind key is HYPH at
            # 86 % -- under the bar precisely because there the distinction is real.
            new = (table.get((c[1], space)) or table.get((c[1], None))
                   or ptb_class(c[1], space))
            if new and new != ",":
                changed[f"{c[1]} ({space}) , -> {new}"] += 1
                c[4] = new
                lines[idx[i]] = "\t".join(c)
            else:
                undecided[f"{c[1]} ({space})"] += 1
        out.append("\n".join(lines))

    new_raw = "\n\n".join(out)
    verify(raw, new_raw, path)
    if not dry_run:
        open(path, "w", encoding="utf-8").write(new_raw)
    return changed, undecided


def verify(old, new, path):
    o, n = old.split("\n"), new.split("\n")
    if len(o) != len(n):
        raise SystemExit(f"{path}: line count changed")
    for i, (a, b) in enumerate(zip(o, n), 1):
        if a == b:
            continue
        ca, cb = a.split("\t"), b.split("\t")
        if len(ca) != 10 or not ca[0].isdigit() or \
                [x for j, x in enumerate(ca) if j != 4] != [x for j, x in enumerate(cb) if j != 4]:
            raise SystemExit(f"{path}:{i}: something other than XPOS changed:\n  {a!r}\n  {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--harvest-from", default=None,
                    help="file to harvest GUM's table from; defaults to the largest input")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ONE table for every split.  Harvesting per file looked reasonable and was a bug: dev and
    # test carry a fraction of GUM's evidence, so they fell under --min-count on forms train
    # committed (`/` -> SYM in train, left `,` in dev/test) -- which would have left the gold
    # inconsistent between train and test, the exact defect this script exists to remove.
    src = a.harvest_from or max(a.files, key=lambda f: __import__("os").path.getsize(f))
    table = harvest(open(src, encoding="utf-8").read().split("\n\n"), a.min_count)
    print(f"table harvested from {src}: {len(table)} committed keys\n")

    total, undec = collections.Counter(), collections.Counter()
    for path in a.files:
        ch, un = normalise(path, table, a.dry_run)
        total.update(ch)
        undec.update(un)
        print(f"{path}: {sum(ch.values())} EWT cells rewritten")
    print("\nrewrites:", ", ".join(f"{k} ({v})" for k, v in total.most_common()))
    if undec:
        print("left alone (GUM commits nothing and no PTB class applies):",
              dict(undec.most_common(10)))
    if a.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
