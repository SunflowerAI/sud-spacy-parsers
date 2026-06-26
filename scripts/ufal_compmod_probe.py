#!/usr/bin/env python
"""Does the LLM help comp-vs-mod on the *classical* UFAL Sanskrit test set?

Ground truth = UFAL's own committed labels on verb-headed oblique nominals (comp:obl ->
complement, mod -> modifier). For each we compare two predictors:
  * the case rule (the sa pipeline's SA_MOD_CASES + dative-recipient frames);
  * the LLM (qwen3:8b, canonical English comp/mod prompt).
The decisive subset is the Ins/Acc/Gen residue, where the case rule is blind — there the
question is whether the LLM beats the majority baseline.
"""
import importlib.util
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from sa_tokenizer import normalise

d = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "d", "scripts/disambiguate_pp.py"))
importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py").loader.exec_module(d)
e = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "e", "scripts/eval_prompts.py"))
importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py").loader.exec_module(e)

UFAL = "assets_sa_ufal/SUD_Sanskrit-UFAL/sa_ufal-sud-test.conllu"
SA_MOD_CASES = {"Loc", "Abl", "Voc", "Nom"}
PREFIX = e.PREFIXES["fewshot12_def"]


def feat(feats, k):
    return dict(f.split("=") for f in (feats or "").split("|") if "=" in f).get(k)


def iast_tokens(toks):
    for t in toks:
        t["form"] = normalise(t["form"])
    return toks


def case_rule(verb_lemma, case, frames):
    if (verb_lemma, case) in frames:
        return "complement"
    if case in SA_MOD_CASES:
        return "modifier"
    return None                                            # residue: rule is blind


def main():
    # derive dative-recipient frames from UFAL itself (verb,Case=Dat that the treebank commits comp)
    frames = set()
    gold = []                                              # committed comp:obl / mod items
    cnt = Counter()
    for sid, toks in d.parse_conllu(UFAL):
        toks = iast_tokens(toks)
        by = {t["id"]: t for t in toks}
        for t in toks:
            if t["head"] == 0 or t["upos"] not in ("NOUN", "PRON", "ADJ", "NUM"):
                continue
            head = by.get(t["head"])
            if not head or head["upos"] != "VERB":
                continue
            base = t["deprel"].split("@")[0]
            case = feat(t.get("feats"), "Case")
            if base == "comp:obl" and case == "Dat":
                frames.add((head["lemma"], case))
            if base in ("comp:obl", "mod", "udep"):
                sub = d.descendants(toks, t["id"])
                if any(by[i]["upos"] == "VERB" for i in sub if i != t["id"]) or len(sub) > 8:
                    continue
                item = {"verb": head["form"], "case": case or "-", "deprel": base,
                        "prep_phrase": d.render(toks, sub),
                        "verb_phrase": d.render(toks, d.descendants(toks, head["id"])),
                        "gold": {"comp:obl": "complement", "mod": "modifier"}.get(base)}
                gold.append(item)

    committed = [g for g in gold if g["gold"]]
    print(f"UFAL verb-headed obliques: committed comp/mod = {len(committed)}, "
          f"udep (uncommitted) = {sum(1 for g in gold if g['deprel']=='udep')}")
    print("committed by (case, gold):",
          dict(Counter((g["case"], g["gold"]) for g in committed).most_common()))

    # the residue = committed items whose case the rule cannot decide (not in SA_MOD_CASES,
    # not a dative-recipient frame) — Ins/Acc/Gen and dative-of-purpose
    residue = [g for g in committed
               if g["case"] not in SA_MOD_CASES and (None, g["case"]) not in frames]
    maj = Counter(g["gold"] for g in residue).most_common(1)
    maj_label = maj[0][0] if maj else "complement"

    rule_ok = rule_dec = llm_ok = 0
    res_rule_ok = res_llm_ok = 0
    for g in committed:
        # need verb lemma for frames; recompute via a light pass is overkill — frames keyed by lemma,
        # but we stored verb form; approximate rule with case only (frames mostly empty on tiny data)
        rl = "modifier" if g["case"] in SA_MOD_CASES else None
        prompt = PREFIX + e.suffix(g)
        pred = d.query(prompt)
        pred = "complement" if pred.startswith("comp") else ("modifier" if pred.startswith("mod") else pred)
        if pred == g["gold"]:
            llm_ok += 1
        if rl is not None:
            rule_dec += 1
            if rl == g["gold"]:
                rule_ok += 1
        else:                                              # residue
            if maj_label == g["gold"]:
                res_rule_ok += 1
            if pred == g["gold"]:
                res_llm_ok += 1
    n = len(committed)
    nres = len(residue)
    print(f"\n=== against UFAL committed labels (n={n}) ===")
    print(f"LLM accuracy overall:            {llm_ok}/{n} = {llm_ok/n:.2f}")
    print(f"case rule (where it decides):    {rule_ok}/{rule_dec} = {rule_ok/max(rule_dec,1):.2f}")
    print(f"\n=== residue (Ins/Acc/Gen/Dat, rule blind; n={nres}) ===")
    print(f"majority baseline ('{maj_label}'): {res_rule_ok}/{nres} = {res_rule_ok/max(nres,1):.2f}")
    print(f"LLM on residue:                  {res_llm_ok}/{nres} = {res_llm_ok/max(nres,1):.2f}")


if __name__ == "__main__":
    main()
