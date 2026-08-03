#!/usr/bin/env python3
"""What SHOULD each remaining `udep` be? Answered from the treebank's own committed decisions.

`relabel_ext.py` only ever asks one question — comp:obl or mod — so anything that is not an
adpositional/case-marked oblique falls out of scope and stays `udep`. The survey behind this script
found 14 k such tokens across nine languages, dominated by material where no oblique/modifier choice
is being deferred at all: Persian's relativiser `که` (5 060), English `'s` and infinitival `to`
(950), Japanese adnominal/copular auxiliaries `た`/`な` (355). Those still have a CORRECT SUD label;
it just isn't comp:obl or mod.

The method is the one this project already uses for gold-building: for each residual `udep`, find
every token in the SAME treebank sharing its (head UPOS, dependent UPOS, dependent lemma) signature
that the annotators DID commit, and report the distribution. Where one label takes essentially all
of it, that is a rule and needs no model; where the split is real, it is a genuine ambiguity and
belongs to an LLM pass or to nobody.

Deliberately conservative: it reports, it does not rewrite. A pattern only counts as decisive if it
has enough committed evidence to be worth trusting (`--min-committed`, default 20) and is dominated
past `--threshold` (default 0.90). Everything else is printed as unresolved so the residue after any
rule is visible rather than implied.

    udep_residue_audit.py --lang fa
    udep_residue_audit.py --all --threshold 0.95
"""
import argparse
import collections
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dpp", _HERE / "disambiguate_pp.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

TREEBANKS = {
    "en": "assets/en_ewt-sud-train.relabeled_ext.conllu",
    "zh": "assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-train.relabeled_ext.conllu",
    "id": "assets_id/SUD_Indonesian-GSD/id_gsd-sud-train.relabeled_ext.conllu",
    "ko": "assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.relabeled_ext.conllu",
    "fa": "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-train.relabeled_ext.conllu",
    "ar": "assets_ar/SUD_Arabic-PADT/ar_padt-sud-train.relabeled_ext.conllu",
    "ja": "assets_ja/SUD_Japanese-GSD/ja_gsd-sud-train.relabeled_ext.conllu",
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-train.relabeled_ext.conllu",
    "yue": "assets_yue/SUD_Cantonese-HK/yue_hk-sud-train.relabeled_ext.conllu",
}


def is_udep(rel):
    return rel == "udep" or rel.startswith("udep@")


def signature(head, t):
    """The key a rule would fire on: head POS, dependent POS, dependent lemma."""
    return (head["upos"], t["upos"], (t["lemma"] or t["form"]).lower())


def audit(path, min_committed, threshold):
    committed = collections.defaultdict(collections.Counter)
    residue = collections.Counter()
    for _sid, toks in d.parse_conllu(path):
        by = {t["id"]: t for t in toks}
        for t in toks:
            head = by.get(t["head"])
            if head is None:
                continue
            sig = signature(head, t)
            if is_udep(t["deprel"]):
                residue[sig] += 1
            else:
                committed[sig][t["deprel"]] += 1

    rules, unresolved = [], []
    for sig, n in residue.most_common():
        dist = committed.get(sig)
        if not dist:
            unresolved.append((sig, n, None, 0.0, 0))
            continue
        label, top = dist.most_common(1)[0]
        total = sum(dist.values())
        share = top / total
        if total >= min_committed and share >= threshold:
            rules.append((sig, n, label, share, total))
        else:
            unresolved.append((sig, n, label, share, total))
    return rules, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-committed", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    langs = list(TREEBANKS) if a.all else [a.lang]
    grand_rule = grand_unres = 0
    for lang in langs:
        p = pathlib.Path(TREEBANKS[lang])
        if not p.exists():
            print(f"\n=== {lang}: {p} not found"); continue
        rules, unresolved = audit(p, a.min_committed, a.threshold)
        n_rule = sum(n for _, n, *_ in rules)
        n_unres = sum(n for _, n, *_ in unresolved)
        grand_rule += n_rule; grand_unres += n_unres
        print(f"\n=== {lang}: {n_rule + n_unres} residual udep — "
              f"{n_rule} rule-decidable ({n_rule/max(n_rule+n_unres,1):.0%}), {n_unres} not")
        if rules:
            print(f"  {'head':<6} {'dep':<6} {'lemma':<14} {'n':>6}  -> {'label':<16} {'evidence'}")
            for (h, dp, lm), n, label, share, total in rules[:a.top]:
                print(f"  {h:<6} {dp:<6} {lm[:14]:<14} {n:6d}  -> {label:<16} "
                      f"{share:.0%} of {total}")
        if unresolved[:a.top]:
            print("  unresolved:")
            for (h, dp, lm), n, label, share, total in unresolved[:a.top]:
                why = "no committed evidence" if label is None else \
                      f"best {label} {share:.0%} of {total}"
                print(f"    {h:<6} {dp:<6} {lm[:14]:<14} {n:6d}  ({why})")
    if len(langs) > 1:
        tot = grand_rule + grand_unres
        print(f"\n=== TOTAL {tot}: {grand_rule} rule-decidable ({grand_rule/max(tot,1):.0%}), "
              f"{grand_unres} not")


if __name__ == "__main__":
    main()
