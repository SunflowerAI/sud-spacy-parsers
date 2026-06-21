#!/usr/bin/env python3
"""Sample the `udep` 'hard residue' buckets with full sentence context, for human judging."""
import importlib.util, random
_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)

EN = "assets/en_sud-train.conllu"
KO = "assets_ko/SUD_Korean-GSD/ko_gsd-sud-train.conllu"


def sent_text(toks):
    out = ""
    for t in toks:
        out += t["form"] + (" " if t["space_after"] else "")
    return out.strip()


def collect(path, pred, n, seed=1):
    hits = []
    for sid, toks in d.parse_conllu(path):
        by = {t["id"]: t for t in toks}
        for t in toks:
            if t["deprel"] != "udep" or t["head"] == 0:
                continue
            head = by.get(t["head"])
            if head and pred(toks, by, t, head):
                dep = d.render(toks, d.descendants(toks, t["id"]))
                hits.append((sent_text(toks), head["form"], head["upos"], t["form"], t["upos"], dep))
    random.Random(seed).shuffle(hits)
    return hits[:n]


def has_verb_sub(toks, by, t):
    sub = d.descendants(toks, t["id"])
    return any(by[i]["upos"] == "VERB" for i in sub if i != t["id"])


def show(title, rows):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    for s, hf, hp, df, dp, dep in rows:
        print(f"\n  sentence : {s[:160]}")
        print(f"  head     : {hf} ({hp})")
        print(f"  udep dep : '{dep[:90]}'   [dep token: {df} / {dp}]")


# A. clausal ADP-of-VERB: a PP whose subtree contains a verb (gerund/relative inside)
show("A. English — clausal PP on a verb (ADP subtree contains a VERB)",
     collect(EN, lambda tk, by, t, h: t["upos"] == "ADP" and h["upos"] == "VERB" and has_verb_sub(tk, by, t), 8))

# B. dep = SCONJ on a verb (subordinator: complement clause vs adverbial clause)
show("B. English — SCONJ dependent of a verb (clausal)",
     collect(EN, lambda tk, by, t, h: t["upos"] == "SCONJ" and h["upos"] == "VERB", 8))

# C. AUX-headed ADP (copula: predicate locative vs modifier)
show("C. English — ADP dependent of an AUX (copula)",
     collect(EN, lambda tk, by, t, h: t["upos"] == "ADP" and h["upos"] == "AUX", 8))

# D. dep = VERB on a verb head (e.g. 'including', 'according to')
show("D. English — VERB dependent of a verb (participial)",
     collect(EN, lambda tk, by, t, h: t["upos"] == "VERB" and h["upos"] == "VERB", 6))

# E. Korean bare NP dependent of a verb (the mixed comp/subj/mod residue)
show("E. Korean — bare NOUN dependent of a verb",
     collect(KO, lambda tk, by, t, h: t["upos"] == "NOUN" and h["upos"] == "VERB" and not has_verb_sub(tk, by, t), 8))
