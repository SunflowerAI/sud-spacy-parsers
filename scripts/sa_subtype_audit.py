#!/usr/bin/env python3
"""Audit Sanskrit's `udep@<subtype>` semantic-role tags to see whether they carry usable
comp:obl/mod signal that `scripts/relabel_ext.py`'s Case-only `sa_case` bucket currently misses
entirely (it only fires on bare `udep`, never `udep@instr`/`udep@goal`/etc).

For each subtype: (1) whether SUD ever *commits* the SAME subtype as comp:obl@X/mod@X elsewhere
in the corpus (the `_harvest_committed` signal `lang_gold.py` uses for lzh/fa/ar/la), (2) the
Case-feature distribution of the still-`udep` instances, (3) the top governing-verb lemmas (to
spot a motion/placement/selecting class, parallel to lzh's `LZH_LOC_COMP_VCLASS`), (4) a few
sample lines for eyeballing. Read-only / analysis-only -- writes nothing.

Findings (2026-07-30, train split, see CLAUDE.md for the classification this fed into):
  - @manner is the ONLY subtype with in-treebank commit evidence: 626 mod@manner / 0
    comp:obl@manner -> confident MOD default.
  - @instr/@lmod/@tmod/@source/@benef/@grad have no commit evidence, but their Case distribution
    is dominated by cases already established as circumstantial for Sanskrit (SA_MOD_CASES =
    Loc/Abl/Voc/Nom) or by canonical adjunct semantics (Ins-of-means, Dat/Gen-of-purpose,
    Abl-of-comparison) -> MOD default.
  - @goal/@path are dominated (>85%) by motion/placement/ritual-offering verb heads (i, gam,
    āgam, praviś, āruh, āviś, dhā, nidhā, hu, nī, gamay, anugam, kram, car, yā...) taking
    Acc/Loc goal-of-motion or path-traversed arguments -- the paradigm SELECTED oblique ->
    COMP:OBL default (no verb-class gating needed; the annotators' own subtype IS the signal).
  - @soc is a genuine mix on sampling (ingredient-mixing instrumentals vs. true accompaniment
    under joining/union verbs) -> left for the LLM, same as Ins/Acc/Gen elsewhere in Sanskrit.
"""
import importlib.util
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)

TRAIN = "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-train.conllu"
SUBTYPES = ["instr", "goal", "lmod", "tmod", "source", "manner", "soc", "benef", "grad", "path"]


def _feat(feats, key):
    import re
    m = re.search(rf"{key}=([^|]+)", feats or "")
    return m.group(1) if m else "_"


def audit():
    committed = Counter()      # (subtype, base_deprel) -> count, for base in {comp:obl, mod}
    case_dist = {s: Counter() for s in SUBTYPES}
    verb_dist = {s: Counter() for s in SUBTYPES}
    samples = {s: [] for s in SUBTYPES}

    for sid, toks in d.parse_conllu(TRAIN):
        by = {t["id"]: t for t in toks}
        for t in toks:
            base, _, sub = t["deprel"].partition("@")
            if not sub or sub not in SUBTYPES:
                continue
            if base in ("comp:obl", "mod"):
                committed[(sub, base)] += 1
            if base != "udep":
                continue
            case = _feat(t.get("feats"), "Case")
            case_dist[sub][case] += 1
            head = by.get(t["head"])
            if head:
                verb_dist[sub][head["lemma"]] += 1
            if len(samples[sub]) < 5:
                samples[sub].append((t["form"], case, head["lemma"] if head else "?",
                                      d.render(toks, d.descendants(toks, t["head"]))[:60]))

    print("=== already-committed comp:obl@X / mod@X for these subtypes (harvest signal) ===")
    for sub in SUBTYPES:
        c_comp, c_mod = committed[(sub, "comp:obl")], committed[(sub, "mod")]
        if c_comp or c_mod:
            print(f"  @{sub:8} comp:obl={c_comp:5} mod={c_mod:5}")
    print()

    for sub in SUBTYPES:
        print(f"=== udep@{sub} (n={sum(case_dist[sub].values())}) ===")
        print(f"  Case: {dict(case_dist[sub].most_common())}")
        print(f"  top heads: {verb_dist[sub].most_common(10)}")
        for form, case, vlemma, vphrase in samples[sub]:
            print(f"    {form} (Case={case}) <- {vlemma}: {vphrase}")
        print()


if __name__ == "__main__":
    audit()
