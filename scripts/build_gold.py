#!/usr/bin/env python3
"""Build a high-confidence comp/mod benchmark from SUD `udep` cases.

English SUD dumps almost all prepositional verb dependents into `udep`. Many of those are
nonetheless unambiguous, so we self-label the confident ones:

  COMPLEMENT: the verb lexically selects the preposition (prepositional-verb frames such as
              depend ON, accuse OF, refer TO, deal WITH, ...).
  MODIFIER:   a temporal / circumstantial PP that attaches freely (during X, in <year>,
              on <weekday>, at <time-noun>, ...), with the verb NOT selecting the preposition.

Both sides are drawn from `udep` cases, giving a clean, on-target test set. Writes JSONL.
"""
import importlib.util, json, re
from collections import Counter

_spec = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(d)

FILES = ["assets/en_sud-train.conllu", "assets/en_sud-dev.conllu", "assets/en_sud-test.conllu"]

# prepositional-verb frames: prep lemma -> verbs that lexically select it (high precision)
FRAMES = {
    "on": {"depend", "rely", "focus", "insist", "concentrate", "embark", "dwell",
           "elaborate", "comment", "decide", "rest", "prey", "feed", "dote", "base",
           "pride", "thrive", "count", "bank", "hinge", "capitalize", "frown"},
    "of": {"consist", "accuse", "approve", "conceive", "remind", "convince", "deprive",
           "rob", "inform", "warn", "rid", "boast", "beware", "despair", "repent",
           "smell", "taste", "dream", "complain", "die", "partake", "dispose"},
    "to": {"refer", "relate", "react", "object", "contribute", "resort", "succumb",
           "subscribe", "allude", "adhere", "conform", "cater", "amount", "testify",
           "belong", "cling", "appeal", "attend", "confess", "defer", "accede",
           "aspire", "consent", "listen", "reply", "respond", "yield", "adapt", "revert"},
    "about": {"talk", "think", "care", "worry", "complain", "dream", "boast", "joke",
              "inquire", "fret", "quibble", "fantasize", "reminisce", "grumble", "wonder"},
    "at": {"look", "glance", "stare", "gaze", "aim", "hint", "peek", "peer", "marvel",
           "scoff", "sneer", "glare", "gawk", "balk", "excel", "jeer"},
    "with": {"deal", "cope", "comply", "interfere", "tamper", "collide", "associate",
             "cooperate", "correspond", "coincide", "clash", "flirt", "sympathize",
             "empathize", "contend", "grapple", "dispense", "meddle", "tinker",
             "converse", "consort", "reconcile", "side"},
    "in": {"result", "persist", "engage", "participate", "invest", "specialize",
           "indulge", "believe", "enroll", "dabble", "revel", "wallow", "confide",
           "culminate", "reside"},
    "for": {"account", "search", "long", "yearn", "hope", "wait", "wish", "strive",
            "opt", "vouch", "atone", "compensate", "cater", "qualify", "apologize",
            "advocate", "campaign", "clamor", "hanker", "pine", "fend", "hunger"},
    "from": {"suffer", "benefit", "recover", "differ", "derive", "refrain", "abstain",
             "stem", "descend", "deviate", "detract", "desist", "profit", "recoil",
             "graduate", "emanate", "shrink"},
    "into": {"delve", "tap", "plunge", "morph", "segue", "merge", "evolve", "transform"},
    "against": {"protest", "rebel", "discriminate", "guard", "conspire", "retaliate",
                "militate", "campaign", "caution", "plead", "sin"},
    "upon": {"rely", "depend", "embark", "prey", "stumble", "impinge", "encroach"},
}

# temporal-modifier signals
TEMP_PREPS_ALWAYS = {"during", "throughout", "amid", "amidst"}
TEMP_OBJ_PREPS = {"in", "on", "at", "by", "over", "within", "since", "until", "before", "after"}
TEMP_NOUNS = {
    "morning", "afternoon", "evening", "night", "midnight", "noon", "day", "week",
    "weekend", "fortnight", "month", "year", "decade", "century", "hour", "minute",
    "moment", "dawn", "dusk", "today", "tomorrow", "yesterday", "spring", "summer",
    "autumn", "fall", "winter", "christmas", "easter", "time", "era", "age", "period",
    "future", "past", "outset", "beginning", "end", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
}


def prep_object(toks, prep_id):
    """Return the object token a preposition governs (its comp:obj child), or None."""
    children = [t for t in toks if t["head"] == prep_id]
    for t in children:
        if t["deprel"].startswith("comp:obj"):
            return t
    for t in children:
        if t["upos"] in ("NOUN", "PROPN", "NUM", "PRON"):
            return t
    return None


def is_year(form):
    return bool(re.fullmatch(r"\d{4}", form)) and 1000 <= int(form) <= 2100


def classify(toks, prep, head):
    verb = head["lemma"].lower()
    prep_l = prep["lemma"].lower()
    # confident complement: verb lexically selects this preposition
    if prep_l in FRAMES and verb in FRAMES[prep_l]:
        # ...but a temporal object makes it a temporal modifier, e.g. "believe in 1999"
        obj = prep_object(toks, prep["id"])
        if obj and (obj["lemma"].lower() in TEMP_NOUNS or is_year(obj["form"])):
            return "modifier"
        return "complement"
    # confident modifier: temporal/circumstantial, verb does NOT select the prep
    if verb in FRAMES.get(prep_l, set()):
        return None  # selected -> not a free modifier; skip
    if prep_l in TEMP_PREPS_ALWAYS:
        return "modifier"
    if prep_l in TEMP_OBJ_PREPS:
        obj = prep_object(toks, prep["id"])
        if obj and (obj["lemma"].lower() in TEMP_NOUNS or is_year(obj["form"])):
            return "modifier"
    return None


def main():
    items, dropped = [], 0
    for f in FILES:
        for sid, toks in d.parse_conllu(f):
            by = {t["id"]: t for t in toks}
            for t in toks:
                if t["upos"] != "ADP" or t["deprel"] != "udep" or t["head"] == 0:
                    continue
                head = by[t["head"]]
                if head["upos"] != "VERB":
                    continue
                pp = d.render(toks, d.descendants(toks, t["id"]))
                if len(pp.split()) < 2:
                    continue
                label = classify(toks, t, head)
                if label is None:
                    dropped += 1
                    continue
                items.append({"sent_id": sid, "verb": head["form"],
                              "prep": t["lemma"].lower(), "prep_phrase": pp,
                              "verb_phrase": d.render(toks, d.descendants(toks, head["id"])),
                              "gold": label})

    dist = Counter(i["gold"] for i in items)
    print(f"Confident udep-derived gold: {dict(dist)}  (skipped {dropped} unclear udep cases)\n")
    for cls in ("complement", "modifier"):
        print(f"--- sample {cls} ---")
        for i in [x for x in items if x["gold"] == cls][:10]:
            print(f"   [{i['verb']} … {i['prep']}]  {i['prep_phrase'][:70]}")
        print()

    with open("gold_udep.jsonl", "w") as fh:
        for i in items:
            fh.write(json.dumps(i) + "\n")
    print(f"Wrote {len(items)} items to gold_udep.jsonl")


if __name__ == "__main__":
    main()
