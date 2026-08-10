#!/usr/bin/env python
"""Split a merged Latin test file into its three treebank spans, by sent_id.

The XPOS normalisation leaves ITTB's rows byte-identical and rewrites only PROIEL's and
Perseus's, so the ITTB slice is the ONE span whose gold is the same before and after -- which
makes it the only apples-to-apples measurement of whether normalising the other two treebanks
helped or hurt tagging on the largest one.  PROIEL's gold moved (23 codes -> composite ones)
and Perseus had no gold at all (blanked), so a TAG figure on those spans is a new number, not
a comparison.

    slice_la_test_by_treebank.py <merged-test.conllu> <outdir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalise_la_xpos import sent_id_of, treebank_of  # noqa: E402

src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
blocks = [b for b in src.read_text(encoding="utf-8").split("\n\n") if b.strip()]
groups = {}
for b in blocks:
    groups.setdefault(treebank_of(sent_id_of(b) or ""), []).append(b)
stem = src.name.replace(".conllu", "")
for tb, bs in groups.items():
    out = outdir / f"{tb}-{stem}.conllu"
    out.write_text("\n\n".join(bs) + "\n\n", encoding="utf-8")
    n = sum(1 for b in bs for ln in b.split("\n")
            if "\t" in ln and ln.split("\t")[0].isdigit())
    print(f"{out}: {len(bs)} sentences, {n} tokens")
