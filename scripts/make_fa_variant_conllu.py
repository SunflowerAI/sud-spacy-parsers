#!/usr/bin/env python3
"""Render the Persian treebank at chosen orthographic variants, trees held constant.

The fa counterpart of `make_ar_variant_conllu.py`. Arabic can do this from gold, because PADT ships
`Vform`; Persian has no vocalised gold, so the marks come from the SAME reconstructed table and the
SAME syntactically-derived ezāfe rules the augmenter trains on. That makes this an honest test of
robustness -- can the arm read text pointed the way this table points it -- and NOT an independent
test of whether the pointing is correct. Nothing here can measure the latter; no Persian gold
exists in this project.

Variants: bare / voc_full / voc_sparse / ezafe / arabic (ی->ي, ک->ك) / nozwnj / all.

FORM only; `--check` asserts every other column is byte-identical to the source.
"""
import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa_orth   # noqa: E402

VARIANTS = ["bare", "voc_full", "voc_sparse", "ezafe", "arabic", "nozwnj", "all"]


def read(path):
    sents, cur = [], []
    raw = []
    for line in open(path, encoding="utf-8"):
        raw.append(line)
        if line.startswith("#") or not line.strip():
            if not line.strip() and cur:
                sents.append(cur)
                cur = []
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10 or "-" in f[0] or "." in f[0]:
            continue
        cur.append((len(raw) - 1, int(f[0]), f[1], f[3], int(f[6]), f[7]))
    if cur:
        sents.append(cur)
    return raw, sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--lut", default="scripts/fa_vocalise_lut.json.gz")
    ap.add_argument("--ezafe-rules", default="scripts/fa_ezafe_rules.json")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--prefix", default="fa")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    forms = dict(json.loads(gzip.open(a.lut, "rb").read().decode("utf-8")).get("F", []))
    rules = json.loads(Path(a.ezafe_rules).read_text(encoding="utf-8"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for v in a.variants:
        rng = random.Random(a.seed)
        raw, sents = read(a.src)
        lines = list(raw)
        for sent in sents:
            byid = {tid: (li, form, upos, head, rel) for li, tid, form, upos, head, rel in sent}
            for li, tid, form, upos, head, rel in sent:
                voc = forms.get(fa_orth.strip_diac(form))
                nxt = byid.get(tid + 1)
                ez = bool(nxt and nxt[3] == tid
                          and "|".join((upos, nxt[4], nxt[2])) in rules)
                style = fa_orth.Style(
                    rate=1.0 if v in ("voc_full", "voc_sparse", "all") else 0.0,
                    mode="sparse" if v == "voc_sparse" else "full",
                    ezafe=v in ("ezafe", "all"),
                    arabic=v in ("arabic", "all"),
                    zwnj=v in ("nozwnj", "all"))
                new = fa_orth.vary_word(form, voc, ez, style, rng)
                f = lines[li].rstrip("\n").split("\t")
                f[1] = new
                lines[li] = "\t".join(f) + "\n"
        text = "".join(lines)
        p = out / f"{a.prefix}_{v}.conllu"
        p.write_text(text, encoding="utf-8")
        base = [l.rstrip("\n").split("\t") for l in raw if l.strip() and not l.startswith("#")]
        got = [l.rstrip("\n").split("\t") for l in text.splitlines()
               if l.strip() and not l.startswith("#")]
        for g, b in zip(got, base):
            assert g[0] == b[0] and g[2:] == b[2:], (v, g, b)
        n = sum(1 for g, b in zip(got, base) if g[1] != b[1])
        print(f"  {v:11} {p}  ({n} FORMs differ)")


if __name__ == "__main__":
    main()
