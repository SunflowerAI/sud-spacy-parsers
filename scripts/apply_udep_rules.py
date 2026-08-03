#!/usr/bin/env python3
"""Commit the residual `udep` tokens whose label the treebank already decides for us.

`relabel_ext.py` asks one question (comp:obl or mod) and leaves everything else `udep`. The audit in
`udep_residue_audit.py` shows 23 % of that residue is not ambiguous at all: the same treebank
commits an overwhelming majority label for the identical (head UPOS, dep UPOS, dep lemma) signature
elsewhere. Persian's relativiser `که` under a noun is `mod` in 98 % of 375 committed cases and sits
`udep` 5 060 times; Japanese adnominal `た`/`だ` likewise; Classical Chinese temporal `今`/`後` are
`comp:obj`. None of these is a comp/mod decision, so no model can be scored on them — but they have
a correct SUD label and it is recoverable by rule.

Rules are DERIVED, never hardcoded: this runs the same audit and applies whatever clears the
evidence bar, so it cannot drift from the data. Defaults are deliberately strict (>= 20 committed
examples, >= 90 % dominated).

What it will NOT touch, and why that matters:
  * Arabic ADJ <- NOUN, the largest single ar class. Split by `Case` the committed evidence is
    Gen -> mod 93 % (a rule, but only ~12 residual tokens) while Acc -> subj is just 59 % against
    mod 15 % / comp:obj 14 %. The residue is 79 % accusative, so it is genuinely ambiguous — an
    adjectival predicate with a nominal subject, not an adverbial accusative.
  * en `'s` (568) and `to` (382) under a noun, zh `的` (248), fa `در` under AUX (1 007): all
    below threshold, all left alone.

Writes `*.udep_ruled.conllu` beside the input; the DEPREL column is the only thing that changes.

    apply_udep_rules.py --all --dry-run
    apply_udep_rules.py --lang fa
"""
import argparse
import collections
import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dpp", _HERE / "disambiguate_pp.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

_aspec = importlib.util.spec_from_file_location("aud", _HERE / "udep_residue_audit.py")
aud = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(aud)


def derive_rules(path, min_committed, threshold):
    rules, _ = aud.audit(pathlib.Path(path), min_committed, threshold)
    return {sig: label for sig, _n, label, _s, _t in rules}


def rewrite(path, rules, out_path, dry_run):
    """Block-based rewrite: only the DEPREL cell of a ruled token changes."""
    lines = pathlib.Path(path).read_text(encoding="utf-8").split("\n")
    # first pass: map (sentence index, token id) -> head token, so the signature can be built
    sent, by, blocks = [], {}, []
    for line in lines:
        if line.startswith("#") or not line.strip():
            if not line.strip() and sent:
                blocks.append((sent, by)); sent, by = [], {}
            continue
        c = line.split("\t")
        if len(c) < 10 or "-" in c[0] or "." in c[0]:
            continue
        tok = {"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3],
               "feats": c[5], "head": int(c[6]), "deprel": c[7]}
        sent.append(tok); by[tok["id"]] = tok
    if sent:
        blocks.append((sent, by))

    decided = {}
    applied = collections.Counter()
    si = 0
    for toks, bymap in blocks:
        for t in toks:
            head = bymap.get(t["head"])
            if head is None or not aud.is_udep(t["deprel"]):
                continue
            label = rules.get(aud.signature(head, t))
            if label:
                decided[(si, t["id"])] = label
                applied[f"{head['upos']}->{t['upos']} {(t['lemma'] or t['form']).lower()}"] += 1
        si += 1

    if dry_run:
        return applied, None

    out, si, seen_ids = [], 0, set()
    for line in lines:
        if line.startswith("#") or not line.strip():
            if not line.strip() and seen_ids:
                si += 1; seen_ids = set()
            out.append(line); continue
        c = line.split("\t")
        if len(c) < 10 or "-" in c[0] or "." in c[0]:
            out.append(line); continue
        tid = int(c[0]); seen_ids.add(tid)
        label = decided.get((si, tid))
        if label:
            c[7] = label
            out.append("\t".join(c))
        else:
            out.append(line)
    pathlib.Path(out_path).write_text("\n".join(out), encoding="utf-8")
    return applied, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-committed", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    langs = list(aud.TREEBANKS) if a.all else [a.lang]
    grand = 0
    for lang in langs:
        train = pathlib.Path(aud.TREEBANKS[lang])
        if not train.exists():
            print(f"  {lang}: {train} not found"); continue
        rules = derive_rules(train, a.min_committed, a.threshold)
        total = 0
        # rules are derived from train, then applied to every split of the same treebank
        for split in ("train", "dev", "test"):
            p = pathlib.Path(str(train).replace("-train.", f"-{split}."))
            if not p.exists():
                continue
            applied, written = rewrite(p, rules, str(p).replace(".conllu", ".udep_ruled.conllu"),
                                       a.dry_run)
            n = sum(applied.values()); total += n
            if split == "train":
                top = ", ".join(f"{k}={v}" for k, v in applied.most_common(3))
        grand += total
        print(f"  {lang:<5} {len(rules):3d} rules -> {total:6d} tokens committed   {top}")
    print(f"\n  TOTAL {grand} udep tokens given a label by rule"
          + ("  (dry run, nothing written)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
