#!/usr/bin/env python3
"""Rebuild assets/en_vnrulings_lexicon.json from files that exist on this machine (nothing here is hand-edited).

Sources: the EWT+GUM targets (assets/en_ewtgum-sud-<split>.relabeled_ext.conllu: ADP under VERB, gold `udep`) and the
kaikki.org Wiktionary dumps in ../lexicons (by-pos-verb, by-pos-prep_phrase). All four tables are derived:

  idioms_vp    verb [particle] + preposition headwords of the verb dump whose (verb key, preposition) occurs among the
               targets. Strict: a sense that is not a form-of / alt-of and carries no historical / obsolete / archaic /
               dated / rare / regional / dialectal / nonstandard / reflexive tag; at least one of its examples (parsed by
               en_core_web_lg) shows V + P + object with P a PREPOSITION, and no example uses P as a particle.
  idioms_vnp   verb + (det) + noun + preposition headwords of the verb dump (take care of), stored as `verb noun prep`,
               whose (verb, preposition) occurs among the targets; same sense standard, no example check.
  g2_pairs     verb key|preposition pairs, counted over the TRAIN targets: G2 >= 10.83 and PMI > 0, with count >= 5, or
               count 3-4 for `for` / `with` / `as` (the prepositions that are least often a complement).
  fixed_pps    multiword lower-case entries of the prep_phrase dump that start with a preposition and have a
               non-form-of, unrestricted sense; entries whose rest is only a pronoun (for it, with that) are left out.

    build_en_vnrulings_lexicon.py [--verbs PATH] [--phrases PATH] [--out PATH]
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_verbnet as R                                            # noqa: E402
import relabel_vnobj as V                                              # noqa: E402
from restore_noun_of import DEFAULT_PRISTINE, load_pristine           # noqa: E402

ROOT = HERE.parent
VERBS = ROOT.parent / "lexicons/wiktionary/kaikki.org-dictionary-English-by-pos-verb.jsonl"
PHRASES = ROOT.parent / "lexicons/wiktionary-phrases/kaikki.org-dictionary-English-by-pos-prep_phrase.jsonl"
OUT = ROOT / "assets/en_vnrulings_lexicon.json"
BAD_TAGS = {"historical", "obsolete", "archaic", "dated", "rare", "regional", "dialectal", "nonstandard", "reflexive"}
DETS = {"a", "an", "the", "one's", "someone's", "somebody's", "his", "her", "my", "your", "their", "its", "our", "some", "any"}
PRONOUNS = {"it", "that", "this", "them", "him", "her", "me", "us", "you", "one", "these", "those", "there", "here"}
START_P = (R.PREP_CLASSES["spatial"] | {"of", "for", "with", "as", "by", "on", "in", "to", "at", "from", "about", "after",
           "before", "during", "without", "within", "upon", "into", "onto", "per", "via", "against", "beyond", "like",
           "since", "until", "till", "under", "over", "off", "out", "among", "despite", "except", "toward", "towards"})
G2_MIN, PMI_MIN = 10.83, 0.0
# Hand-curated exclusions from the G2 table (the only entries in any table that are not derived). `use for` names a
# purpose (use it for X): 36 train tokens, every one judged a purpose adjunct in two independent spot-checks, and it
# is not separable from `work for` (also VerbNet use-105.1, half employer) by class or by P(prep | verb).
G2_EXCLUDE = {"use|for"}


def sense_ok(sense):
    return not (sense.get("form_of") or sense.get("alt_of") or BAD_TAGS & set(sense.get("tags", [])))


def jsonl(path, want):
    """Yield (word, entry) for the entries whose headword passes `want` (the key order differs between lines)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            if want(j["word"]):
                yield j["word"], j


def targets(splits=("train", "dev", "test")):
    gold = load_pristine(DEFAULT_PRISTINE)
    out = {}
    for sp in splits:
        rows = []
        for blk in open(ROOT / f"assets/en_ewtgum-sud-{sp}.relabeled_ext.conllu", encoding="utf-8").read().split("\n\n"):
            rows += [(t["key"], t["prep"], t["lemma"]) for _l, _i, _f, t in V.targets(blk, gold)]
        out[sp] = rows
    return out


def g2_pairs(train):
    pair = collections.Counter((k, p) for k, p, _ in train)
    kc, pc, n = collections.Counter(k for k, _, _ in train), collections.Counter(p for _, p, _ in train), len(train)
    keep = []
    for (k, p), a in sorted(pair.items()):
        if not (a >= 5 or (3 <= a <= 4 and p in ("for", "with", "as"))):
            continue
        b, c = kc[k] - a, pc[p] - a
        d = n - a - b - c
        g = 2 * sum(o * math.log(o / (r * cc / n)) for o, r, cc in
                    ((a, a + b, a + c), (b, a + b, b + d), (c, c + d, a + c), (d, c + d, b + d)) if o > 0)
        if g >= G2_MIN and math.log(a * n / (kc[k] * pc[p])) > PMI_MIN and f"{k}|{p}" not in G2_EXCLUDE:
            keep.append(f"{k}|{p}")
    return keep


def idioms(verbs_path, pairs, preps):
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    cand = {}                                   # headword -> (verb, particle, prep)
    vnp = set()
    for word, j in jsonl(verbs_path, lambda w: w == w.lower() and 2 <= len(w.split()) <= 4 and w.split()[-1] in preps):
        w = word.split()
        if not any(sense_ok(s) for s in j.get("senses", [])):
            continue
        if len(w) == 2 and (w[0], w[1]) in pairs:
            cand[word] = (w[0], "", w[1], j)
        elif len(w) == 3 and (f"{w[0]}_{w[1]}", w[2]) in pairs:
            cand[word] = (w[0], w[1], w[2], j)
        if len(w) in (3, 4) and (w[0], w[-1]) in pairs:
            mid = [x for x in w[1:-1] if x not in DETS]
            if len(mid) == 1:
                vnp.add(f"{w[0]} {mid[0]} {w[-1]}")
    vp = []
    for word, (v, prt, p, j) in sorted(cand.items()):
        texts = [e["text"] for s in j["senses"] if sense_ok(s) for e in s.get("examples", []) if e.get("type") != "quotation"]
        shows, particle = False, False
        for doc in nlp.pipe(texts):
            for t in doc:
                if t.lemma_.lower() != v:
                    continue
                kids = list(t.children)
                if prt and not any(k.dep_ == "prt" and k.text.lower() == prt for k in kids):
                    continue
                for k in kids:
                    if k.text.lower() == p:
                        if k.dep_ == "prt":
                            particle = True
                        elif k.dep_ == "prep" and any(c.dep_ == "pobj" for c in k.children):
                            shows = True
        if shows and not particle:
            vp.append(word)
    return vp, sorted(vnp)


def fixed_pps(path):
    out = set()
    for word, j in jsonl(path, lambda w: w == w.lower() and len(w.split()) >= 2 and w.split()[0] in START_P):
        rest = word.split()[1:]
        if all(x in PRONOUNS for x in rest) or not any(sense_ok(s) for s in j.get("senses", [])):
            continue
        out.add(word)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbs", default=str(VERBS))
    ap.add_argument("--phrases", default=str(PHRASES))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tg = targets()
    allt = [r for rows in tg.values() for r in rows]
    pairs = {(k, p) for k, p, _ in allt}
    preps = {p for _, p, _ in allt}
    vp, vnp = idioms(a.verbs, pairs, preps)
    doc = {
        "_provenance": {
            "idioms_vp": f"{len(vp)} headwords of {Path(a.verbs).name}: verb [particle] + preposition whose (verb key, preposition) occurs among the EWT+GUM targets (all splits); strict: no form-of / alt-of sense, no historical / obsolete / archaic / dated / rare / regional / dialectal / nonstandard / reflexive tag, at least one non-quotation example shows V + P + object with P a preposition (en_core_web_lg parse) and none uses P as a particle. Built by scripts/build_en_vnrulings_lexicon.py.",
            "idioms_vnp": "verb + (det) + noun + preposition headwords of the same dump, stored as `verb noun prep`, whose (verb, preposition) occurs among the targets; matched at relabel time when the noun is the verb's object.",
            "g2_pairs": "verb key|prep pairs counted over the ADP-under-VERB targets of assets/en_ewtgum-sud-train.relabeled_ext.conllu: G2 >= 10.83 and PMI > 0, with count >= 5 (any preposition) or 3-4 (for / with / as only), minus the hand-curated exclusion " + ", ".join(sorted(G2_EXCLUDE)) + " (a purpose adjunct; see G2_EXCLUDE in scripts/build_en_vnrulings_lexicon.py).",
            "fixed_pps": f"multiword lower-case entries of {Path(a.phrases).name} starting with a preposition, with a non-form-of unrestricted sense; entries whose rest is only a pronoun (for it, with that) are left out, because they cannot be told from a literal PP.",
        },
        "idioms_vp": vp, "idioms_vnp": vnp, "g2_pairs": g2_pairs(tg["train"]), "fixed_pps": fixed_pps(a.phrases)}
    out = ["{"]
    for n, (k, v) in enumerate(doc.items()):
        c = "," if n < len(doc) - 1 else ""
        if isinstance(v, list):
            out += [json.dumps(k) + ": ["] + [json.dumps(x, ensure_ascii=False) + ("," if i < len(v) - 1 else "") for i, x in enumerate(v)] + ["]" + c]
        else:
            it = list(v.items())
            out += [json.dumps(k) + ": {"] + [json.dumps(x) + ": " + json.dumps(y, ensure_ascii=False) + ("," if i < len(it) - 1 else "") for i, (x, y) in enumerate(it)] + ["}" + c]
    out.append("}")
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in doc.items() if isinstance(v, list)}, "->", a.out)


if __name__ == "__main__":
    main()
