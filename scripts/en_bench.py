#!/usr/bin/env python3
"""Push English comp/mod accuracy with a comprehensive same-preposition contrastive set.
Each confusable preposition (on/for/at/in/to/of/with/from/by/over/after) appears as both a
verb-selected COMPLEMENT and a free/temporal MODIFIER. Benchmarked on gold_udep.jsonl against
the current best (fewshot12_def, 0.912). Test split = balanced 200/class, seed 0.
"""
import importlib.util, json
from collections import Counter

_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)
_sb = importlib.util.spec_from_file_location("b", "scripts/lang_bench.py")
b = importlib.util.module_from_spec(_sb); _sb.loader.exec_module(b)
d = e.d

def S(vp, pp, v, g): return {"verb_phrase": vp, "prep_phrase": pp, "verb": v, "gold": g}

CONTRAST = [
    # COMPLEMENT: verb lexically selects the preposition
    S("The outcome depends on the weather", "on the weather", "depends", "complement"),
    S("She relies on her team", "on her team", "relies", "complement"),
    S("The study focuses on safety", "on safety", "focuses", "complement"),
    S("It is based on evidence", "on evidence", "based", "complement"),
    S("He waited for the bus", "for the bus", "waited", "complement"),
    S("They searched for the keys", "for the keys", "searched", "complement"),
    S("I looked at the painting", "at the painting", "looked", "complement"),
    S("He aimed at the target", "at the target", "aimed", "complement"),
    S("She believes in ghosts", "in ghosts", "believes", "complement"),
    S("The delay resulted in chaos", "in chaos", "resulted", "complement"),
    S("We referred to the manual", "to the manual", "referred", "complement"),
    S("It belongs to the museum", "to the museum", "belongs", "complement"),
    S("The book consists of ten parts", "of ten parts", "consists", "complement"),
    S("They accused him of fraud", "of fraud", "accused", "complement"),
    S("She deals with complaints", "with complaints", "deals", "complement"),
    S("He suffers from asthma", "from asthma", "suffers", "complement"),
    # MODIFIER: temporal / free circumstantial use of the SAME prepositions
    S("He arrived on Friday", "on Friday", "arrived", "modifier"),
    S("She left early for safety", "for safety", "left", "modifier"),
    S("He stayed for three years", "for three years", "stayed", "modifier"),
    S("They met at noon", "at noon", "met", "modifier"),
    S("He read it in the morning", "in the morning", "read", "modifier"),
    S("It happened in 1999", "in 1999", "happened", "modifier"),
    S("She was born in April", "in April", "born", "modifier"),
    S("He sang in the kitchen", "in the kitchen", "sang", "modifier"),
    S("They worked during the war", "during the war", "worked", "modifier"),
    S("We traveled by train", "by train", "traveled", "modifier"),
    S("Finish the report by Friday", "by Friday", "finish", "modifier"),
    S("She spoke with a smile", "with a smile", "spoke", "modifier"),
    S("It changed over the weekend", "over the weekend", "changed", "modifier"),
    S("They walked home after lunch", "after lunch", "walked", "modifier"),
]


# targeted complement frames the model misclassified as modifier (from en_errors.py)
TARGETED = [
    S("He engages in risky sports", "in risky sports", "engages", "complement"),
    S("They participate in the program", "in the program", "participate", "complement"),
    S("She specializes in tax law", "in tax law", "specializes", "complement"),
    S("The model is based on data", "on data", "based", "complement"),
    S("The risk is associated with smoking", "with smoking", "associated", "complement"),
    S("The event coincided with the holiday", "with the holiday", "coincided", "complement"),
    S("Beware of scams", "of scams", "Beware", "complement"),
    S("He apologized for the delay", "for the delay", "apologized", "complement"),
    S("They strive for excellence", "for excellence", "strive", "complement"),
    S("She belongs to the club", "to the club", "belongs", "complement"),
    S("He was robbed of his savings", "of his savings", "robbed", "complement"),
    S("They looked at the data", "at the data", "looked", "complement"),
    S("It clashes with the theme", "with the theme", "clashes", "complement"),
    S("The plan relies on funding", "on funding", "relies", "complement"),
]


def run(name, prefix, test):
    preds = [d.query(prefix + e.suffix(c)) for c in test]
    preds = [p if p in ("complement", "modifier") else "?" for p in preds]
    acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
    rc = {c: sum(preds[i] == c for i, t in enumerate(test) if t["gold"] == c)
          / max(1, sum(t["gold"] == c for t in test)) for c in ("complement", "modifier")}
    print(f"  {name:22} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
          f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}")


def main():
    gold = e.load_gold("gold_udep.jsonl")
    test = e.balanced_sample(gold, 200, 0)
    print(f"en test {len(test)} (200/class); contrast shots {len(CONTRAST)}\n")
    variants = {
        "fewshot12_def (0.912)": e.PREFIXES["fewshot12_def"],
        "shots12+targeted":      e.PREFIXES["fewshot12_def"] + b.shots_block(TARGETED),
        "shots12+targeted+contrast": e.PREFIXES["fewshot12_def"] + b.shots_block(TARGETED + CONTRAST),
    }
    for name, pre in variants.items():
        run(name, pre, test)


if __name__ == "__main__":
    main()
