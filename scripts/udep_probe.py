#!/usr/bin/env python3
"""Probe the ALREADY-COMMITTED (non-udep) dependents to calibrate the relabel rules.

Default mode: how SUD labels committed ADP dependents by head POS — confirms a comp/mod
contrast exists for NOUN/ADJ heads (motivates the extended scope in relabel_ext.py).

--ko-case mode: the Korean case-particle -> comp/mod calibration that produced
relabel_ext.KO_MOD_CASES / KO_COMP_CASES. For committed VERB<-{NOUN,ADV} dependents, read the
case particle off the rightmost (head-final) eojeol's lemma suffix and tabulate comp vs mod;
also count the particle frequencies in the udep residue that relabel_ext.ko_case_label rules.
"""
import argparse, importlib.util
from collections import Counter, defaultdict

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)

FILES = {
    "en": ["assets/en_sud-train.conllu"],
    "zh": ["assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-train.conllu"],
    "ko": ["assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.conllu"],
    "id": ["assets_id/SUD_Indonesian-GSD/id_gsd-sud-train.conllu"],
}
KO = ["assets_ko/SUD_Korean-GSD/ko_gsd-sud-%s.conllu" % s for s in ("train", "dev", "test")]


def base(deprel):
    return deprel.split(":")[0].split("@")[0]


def ko_case_of(toks, by, tid):
    """Case particle = suffix after last '+' of the rightmost non-punct eojeol (head-final)."""
    sub = [i for i in d.descendants(toks, tid) if by[i]["upos"] != "PUNCT"]
    return by[max(sub)]["lemma"].split("+")[-1]


def ko_case_calib():
    tab = defaultdict(Counter)   # particle -> {comp, mod} among committed VERB<-{NOUN,ADV}
    udep_cnt = Counter()         # particle -> count among the udep residue
    for f in KO:
        for sid, toks in d.parse_conllu(f):
            by = {t["id"]: t for t in toks}
            for t in toks:
                h = by.get(t["head"])
                if not h or h["upos"] != "VERB" or t["upos"] not in ("NOUN", "ADV"):
                    continue
                c = ko_case_of(toks, by, t["id"])
                b = base(t["deprel"])
                if b in ("comp", "mod"):
                    tab[c][b] += 1
                if t["deprel"] == "udep":
                    udep_cnt[c] += 1
    print("Korean VERB<-{NOUN,ADV}: committed case particle -> comp/mod (compShare = comp/total)")
    for c, cnt in sorted(tab.items(), key=lambda kv: -(kv[1]["comp"] + kv[1]["mod"])):
        tot = cnt["comp"] + cnt["mod"]
        if tot >= 5:
            print(f"  {c:8} comp={cnt['comp']:4} mod={cnt['mod']:4}  compShare={cnt['comp']/tot:.2f}  (n={tot})")
    print("\nudep residue particle frequency (what ko_case_label rules), top 20:")
    for c, n in udep_cnt.most_common(20):
        t = tab.get(c, Counter()); tot = t["comp"] + t["mod"]
        sh = f"{t['comp']/tot:.2f}" if tot else "  - "
        print(f"  {c:8} udep_n={n:4}   committed_compShare={sh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko-case", action="store_true", help="Korean case-particle calibration")
    if ap.parse_args().ko_case:
        ko_case_calib(); return
    for lang, paths in FILES.items():
        # committed comp/mod breakdown for ADP dependents, by head UPOS
        committed = defaultdict(Counter)   # headUPOS -> {comp, mod}
        for path in paths:
            for sid, toks in d.parse_conllu(path):
                by = {t["id"]: t for t in toks}
                for t in toks:
                    if t["upos"] != "ADP" or t["head"] == 0:
                        continue
                    b = base(t["deprel"])
                    if b not in ("comp", "mod"):
                        continue
                    h = by.get(t["head"], {}).get("upos", "?")
                    committed[h][b] += 1
        print("=" * 60)
        print(f"{lang}: committed ADP-dependents comp/mod by HEAD upos (train):")
        for h in ("VERB", "NOUN", "PROPN", "ADJ", "AUX", "ADV", "NUM", "PRON"):
            c = committed[h]
            tot = c["comp"] + c["mod"]
            if tot:
                print(f"   head={h:5}  comp={c['comp']:5}  mod={c['mod']:5}  "
                      f"(comp share {c['comp']/tot:.2f})")


if __name__ == "__main__":
    main()
