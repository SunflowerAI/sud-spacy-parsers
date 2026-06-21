#!/usr/bin/env python3
"""Audit ALL `udep` relations across the four SUD treebanks (train+dev+test).

Goal: find udep cases beyond "ADP dependent of VERB" that could be EASILY relabelled
comp vs mod. For each udep token we record head-UPOS, dep-UPOS, whether the dep subtree
is clausal (contains a VERB), subtree size, and whether it is currently in relabel scope.
"""
import importlib.util
from collections import Counter, defaultdict

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)

# original (non-relabeled) gold files, train+dev+test, per language
FILES = {
    "en": ["assets/en_sud-train.conllu", "assets/en_sud-dev.conllu", "assets/en_sud-test.conllu"],
    "zh": ["assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-train.conllu",
           "assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-dev.conllu",
           "assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-test.conllu"],
    "ko": ["assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.conllu",
           "assets_ko/SUD_Korean-GSD/ko_gsd-sud-dev.conllu",
           "assets_ko/SUD_Korean-GSD/ko_gsd-sud-test.conllu"],
    "id": ["assets_id/SUD_Indonesian-GSD/id_gsd-sud-train.conllu",
           "assets_id/SUD_Indonesian-GSD/id_gsd-sud-dev.conllu",
           "assets_id/SUD_Indonesian-GSD/id_gsd-sud-test.conllu"],
}


def is_clausal(toks, by, t):
    sub = d.descendants(toks, t["id"])
    return any(by[i]["upos"] in ("VERB",) for i in sub if i != t["id"])


def current_scope(toks, by, t):
    """The scope the relabel scripts actually act on."""
    if t["upos"] != "ADP" or t["deprel"] != "udep" or t["head"] == 0:
        return False
    if by.get(t["head"], {}).get("upos") != "VERB":
        return False
    sub = d.descendants(toks, t["id"])
    if any(by[i]["upos"] == "VERB" for i in sub if i != t["id"]):
        return False
    return len(sub) <= 8


def main():
    for lang, paths in FILES.items():
        total_udep = 0
        in_scope = 0
        by_headpos = Counter()
        by_deppos = Counter()
        head_dep = Counter()          # (headUPOS, depUPOS)
        clausal = Counter()           # clausal vs nominal among out-of-scope
        # out-of-scope nominal (non-clausal) udep, grouped for "easy win" inspection
        oos_nominal = Counter()       # (headUPOS, depUPOS)
        # specific frequent dep lemmas for non-ADP nominal udep on VERB heads
        examples = defaultdict(list)
        for path in paths:
            for sid, toks in d.parse_conllu(path):
                by = {t["id"]: t for t in toks}
                for t in toks:
                    if t["deprel"] != "udep":
                        continue
                    total_udep += 1
                    h = by.get(t["head"], {}).get("upos", "ROOT" if t["head"] == 0 else "?")
                    by_headpos[h] += 1
                    by_deppos[t["upos"]] += 1
                    head_dep[(h, t["upos"])] += 1
                    cl = is_clausal(toks, by, t)
                    if current_scope(toks, by, t):
                        in_scope += 1
                    else:
                        clausal["clausal" if cl else "nominal"] += 1
                        if not cl:
                            oos_nominal[(h, t["upos"])] += 1
                            grp = (h, t["upos"])
                            if len(examples[grp]) < 4:
                                pp = d.render(toks, d.descendants(toks, t["id"]))
                                hd = by.get(t["head"], {}).get("form", "ROOT")
                                examples[grp].append(f"[{hd}] <- {pp}")
        print("=" * 70)
        print(f"{lang}:  total udep = {total_udep}   currently relabelled (in scope) = {in_scope}")
        print(f"  out-of-scope split: {dict(clausal)}")
        print(f"  udep by HEAD upos: {by_headpos.most_common()}")
        print(f"  udep by DEP  upos: {by_deppos.most_common()}")
        print(f"  -- out-of-scope NOMINAL (non-clausal) udep, by (head,dep) UPOS, top 15 --")
        for (hp, dp), n in oos_nominal.most_common(15):
            ex = examples[(hp, dp)][:2]
            print(f"      head={hp:5} dep={dp:5}  n={n:5}   e.g. {ex}")
    print("=" * 70)


if __name__ == "__main__":
    main()
