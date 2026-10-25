#!/usr/bin/env python3
"""Decide English verb-level `comp:obl` vs `mod` by VerbNet + the semantic class of the PP's object.

DEFINITION (user decision, 2026-10-25). A complement is a preposition the verb selects that does NOT
denote a circumstant. Per token, for every ADP the SUD treebank marks `udep` under a VERB (noun-level
`of` is not touched; it keeps `udep`), currently labelled mod or comp:obl:

  1. DIRECTIONAL -> comp:obl (standing ruling: directional PPs are complements). The preposition is in
     DIRECTIONAL and the object is not a time expression.
  2. CIRCUMSTANT -> mod, first match of:
       fronted    a sentence-initial PP set off by a comma (not a relative `in which` ...)
       discourse  a fixed adverbial PP (relabel_verbnet.DISCOURSE_PPS; `in fact`/`in case` only bare)
       time       relabel_verbnet's temporal rule (with the repairs below), or a time-class object under
                  a TEMPORAL_P preposition
       place      a place-class object under a LOCATIVE preposition
       occasion   an event-class object under at/during/after/before/following/upon
       manner     in/with + way/manner/fashion/style
       reason     for/on/out of + reason/sake/purpose/grounds; because (of), due (to), owing, despite
  3. VERBNET -> comp:obl iff VerbNet 3.3 licenses the preposition for the verb (relabel_verbnet.licensed,
     `--polysemy majority` by default: licensed in at least half of the lemma's classes; particle verbs
     looked up as lemma_particle), else mod. Existential `be` (`there is X in Y`) is mod: VerbNet lists
     `be` only in copular/representation senses.
  4. NOT IN VERBNET (no class for the verb key): the current label is KEPT. No fallback is invented.
  No smoothing: the definition is per token.

OBJECT CLASS (place / time / event / other), first match of: repairs for non-temporal `time` words
(`for your time`, `more time`, `opening times`, `the beginning of this letter`, `promotional ends`);
closed lists; years / decades / clock times / numeric dates / `January 30`; spaCy en_core_web_lg NER on
the sentence for proper nouns (GPE/LOC/FAC place, DATE/TIME time, EVENT event); WordNet 3.0 first noun
sense (noun.time; noun.location or a building/structure/room/facility/region hypernym; noun.event);
otherwise a lemma table `assets/en_objclass_vectors.json` predicted from the cxSUD parser's nucleus
vectors (~/vec/nucvec256.npz on the server: fastText wiki-news-300d-1M, PCA 256) by a logistic
regression trained on the WordNet labels. Hand-checked: 90/100 and 84/100 objects on two samples
before the repairs.

DATA: VerbNet as in relabel_verbnet.py (../lexicons/verbnet3/verbnet3, or --verbnet / $VERBNET_DIR);
WordNet 3.0 from NLTK data (~/nltk_data/corpora/wordnet); spaCy en_core_web_lg installed in .venv.

    relabel_vnobj.py FILE...                 # dry run: counts only
    relabel_vnobj.py --inplace FILE...       # rewrite DEPREL of the targets
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_verbnet as R                      # noqa: E402
from restore_noun_of import DEFAULT_PRISTINE, load_pristine   # noqa: E402

VEC_TABLE = HERE.parent / "assets/en_objclass_vectors.json"
DIRECTIONAL = {"to", "into", "onto", "toward", "towards", "from", "out", "off", "through", "across", "along",
               "past", "around", "round", "over", "up", "down", "via", "away", "throughout"}
LOCATIVE = {"in", "at", "on", "near", "inside", "outside", "within", "beside", "under", "underneath", "beneath",
            "above", "below", "behind", "among", "amongst", "amid", "amidst", "between", "upon", "atop", "opposite"}
TEMPORAL_P = {"in", "at", "on", "during", "for", "before", "after", "since", "until", "till", "by", "over",
              "within", "around", "throughout", "from", "through", "between", "past", "upon"}
EVENT_P = {"at", "during", "after", "before", "following", "upon"}
MANNER_O = {"way", "manner", "fashion", "style"}
REASON_O = {"reason", "sake", "purpose", "grounds"}
REASON_P = {"because", "due", "owing", "despite"}
REL = {"which", "whom", "what", "that", "who", "where", "whose"}

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"}
PLACE_W = {"street", "road", "world", "earth", "planet", "sea", "ocean", "river", "lake", "field", "ground", "floor",
           "stage", "campus", "here", "there", "where", "somewhere", "everywhere", "anywhere", "nowhere", "abroad",
           "home", "downtown", "overseas", "upstairs", "downstairs", "inside", "outside"}
TIME_W = {"now", "then", "when", "today", "tonight", "tomorrow", "yesterday", "once", "recently", "ago", "later",
          "soon", "while"}
EVENT_W = {"vacation", "holiday", "trip", "leave", "break", "honeymoon", "visit"}
QUANT = {"more", "much", "enough", "extra", "some", "any", "no", "less", "plenty", "little", "lot", "all"}
TEMP_HEADS = {"year", "day", "week", "month", "century", "decade", "season", "summer", "winter", "spring", "autumn",
              "fall", "war", "era", "period", "game", "match", "meeting", "film", "movie", "show", "trip", "semester",
              "term", "life"}
NUM_TIME = re.compile(r"'?\d0'?s|\d{3}0'?s|\d{1,2}:\d\d|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", re.I)


class ObjectClassifier:
    def __init__(self, vec_table=VEC_TABLE):
        from nltk.corpus import wordnet as wn
        self.wn = wn
        self.place_hyp = {wn.synset(s) for s in (
            "location.n.01", "region.n.01", "structure.n.01", "room.n.01", "facility.n.01", "geographical_area.n.01",
            "area.n.05", "establishment.n.04", "building.n.01", "point.n.02", "land.n.04")}
        self.vec = {k: v for k, v in json.load(open(vec_table, encoding="utf-8")).items() if not k.startswith("_")}
        self._nlp = None
        self._ents = {}
        self._wn = {}

    def wn_class(self, w):
        if w not in self._wn:
            ss = self.wn.synsets(w, "n")
            if not ss:
                self._wn[w] = None
            else:
                s = ss[0]
                hyps = set(s.closure(lambda t: t.hypernyms() + t.instance_hypernyms())) | {s}
                self._wn[w] = ("time" if s.lexname() == "noun.time" else
                               "place" if s.lexname() == "noun.location" or hyps & self.place_hyp else
                               "event" if s.lexname() == "noun.event" else "other")
        return self._wn[w]

    def ner_class(self, sent, form):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_lg")
        if sent not in self._ents:
            self._ents[sent] = [(e.text.split(), e.label_) for e in self._nlp(sent).ents]
        for words, lab in self._ents[sent]:
            if form in words:
                return {"GPE": "place", "LOC": "place", "FAC": "place", "DATE": "time", "TIME": "time",
                        "EVENT": "event"}.get(lab, "other")
        return None

    def classify(self, obj, sent):
        """obj: dict(lemma, form, upos, kids=[(lemma, deprel)], ofobj) or None -> class"""
        if obj is None:
            return "other"
        o, f, kids = obj["lemma"], obj["form"], obj["kids"]
        deps = {d.split("@")[0] for _, d in kids}
        kl = {lem for lem, _ in kids}
        if o == "time" and (any("poss" in d for _, d in kids) or kl & QUANT
                            or (f.lower() == "times" and "compound" in deps)):
            return "other"
        if o in ("beginning", "end", "start") and obj["ofobj"] and obj["ofobj"] not in TEMP_HEADS \
                and self.wn_class(obj["ofobj"]) not in ("time", "event"):
            return "other"
        if o == "end" and f.lower() == "ends":
            return "other"
        if o in EVENT_W:
            return "event"
        if o in ("time", "age"):
            return "time"
        if o in ("calendar", "schedule", "agenda"):
            return "other"
        if f.isdigit() and int(f) <= 31 and kl & MONTHS:
            return "time"
        if o in PLACE_W:
            return "place"
        if o in TIME_W:
            return "time"
        if re.fullmatch(r"(1[0-9]|20)\d\d", f) or NUM_TIME.fullmatch(f):
            return "time"
        if obj["upos"] == "PROPN":
            c = self.ner_class(sent, f)
            if c:
                return c
        else:
            c = self.wn_class(o)
            if c:
                return c
        return self.vec.get(o, "other")


def _temporal_repaired(temporal, obj, prep, ocls):
    if not temporal:
        return False
    if obj is None:
        return True
    kids = obj["kids"]
    poss = any("poss" in d for _, d in kids)
    det = any(d.startswith("det") or "poss" in d for _, d in kids)
    if obj["lemma"] == "time" and poss:
        return False
    if obj["lemma"] in ("fact", "case") and det:
        return False
    if prep == "throughout" and obj["lemma"] not in ("time", "beginning", "end", "fact", "case", "way") \
            and not obj["form"].isdigit():
        return False
    if obj["form"].lower() == "springs":
        return False
    if ocls == "other" and obj["lemma"] in ("time", "beginning", "end", "start"):
        return False                     # a repaired non-temporal `time` / `beginning of [a text]`
    return True


def decide(t, members, oc, polysemy):
    """-> (source, label); label None = keep the current one."""
    prep, obj = t["prep"], t["obj"]
    ocls = oc.classify(obj, t["sent"])
    if prep in DIRECTIONAL and ocls != "time":
        return "dir", "comp:obl"
    olem = obj["lemma"] if obj else None
    bare = obj is not None and not any(d.startswith("det") or "poss" in d for _, d in obj["kids"])
    if t["left"] and obj is not None and olem not in REL and t["pp"].rstrip().endswith(","):
        return "circ:fronted", "mod"
    if (prep, olem) in R.DISCOURSE_PPS and (olem not in ("fact", "case") or bare):
        return "circ:discourse", "mod"
    if _temporal_repaired(t["temporal"], obj, prep, ocls) or (ocls == "time" and prep in TEMPORAL_P):
        return "circ:time", "mod"
    if ocls == "place" and prep in LOCATIVE:
        return "circ:place", "mod"
    if ocls == "event" and prep in EVENT_P:
        return "circ:occasion", "mod"
    if olem in MANNER_O and prep in ("in", "with"):
        return "circ:manner", "mod"
    if (olem in REASON_O and prep in ("for", "on", "out")) or prep in REASON_P:
        return "circ:reason", "mod"
    key = t["key"]
    if not members.get(key):
        return "novn", None
    if t["lemma"] == "be" and t["expl"]:
        return "vn:existential", "mod"
    return ("vn", "comp:obl") if R.licensed(members, key, prep, polysemy) else ("vn-unlic", "mod")


def targets(blk, gold):
    """relabel_verbnet.targets plus what the definition needs about the object and the PP."""
    for lines, i, f, lemma, prt, prep, temporal in R.targets(blk, gold):
        rows = {}
        for ln in lines:
            if ln and not ln.startswith("#"):
                c = ln.split("\t")
                if len(c) == 10 and c[0].isdigit():
                    rows[c[0]] = c
        kids = collections.defaultdict(list)
        for c in rows.values():
            kids[c[6]].append(c)
        o = next((k for k in kids[f[0]] if k[7].startswith("comp:obj")), None) or \
            next((k for k in kids[f[0]] if k[3] in ("NOUN", "PROPN", "NUM", "PRON")), None)
        obj = None
        if o is not None:
            ofobj = next((g[2].lower() for k in kids[o[0]] if k[2].lower() == "of"
                          for g in kids[k[0]] if g[7].startswith("comp:obj")), None)
            obj = dict(lemma=o[2].lower(), form=o[1], upos=o[3], kids=[(k[2].lower(), k[7]) for k in kids[o[0]]],
                       ofobj=ofobj)
        sub, ch = {f[0]}, True
        while ch:
            ch = False
            for k, c in rows.items():
                if c[6] in sub and k not in sub:
                    sub.add(k)
                    ch = True
        h = rows[f[6]]
        yield lines, i, f, dict(lemma=lemma, prt=prt, key=f"{lemma}_{prt}" if prt else lemma, prep=prep, obj=obj,
                                temporal=bool(temporal), left=int(f[0]) < int(h[0]),
                                pp=" ".join(rows[k][1] for k in sorted(sub, key=int)),
                                expl=any("expl" in k[7] for k in kids[h[0]]),
                                sent=" ".join(c[1] for c in rows.values()))


def relabel(text, members, gold, oc, polysemy, stats, log=None):
    out = []
    for blk in text.split("\n\n"):
        lines = None
        for lines_, i, f, t in targets(blk, gold):
            lines = lines_
            src, lab = decide(t, members, oc, polysemy)
            stats[src] += 1
            if log is not None:
                log.append((t["key"], t["prep"], f[7], lab or f[7], src))
            if lab is not None and lab != f[7]:
                stats[f"{f[7]} -> {lab}"] += 1
                f[7] = lab
                lines[i] = "\t".join(f)
        out.append(blk if lines is None else "\n".join(lines))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--verbnet", default=R.DEFAULT_VN)
    ap.add_argument("--polysemy", choices=("majority", "union"), default="majority")
    ap.add_argument("--inplace", action="store_true")
    a = ap.parse_args()
    members = R.load_verbnet(a.verbnet)
    gold = load_pristine(DEFAULT_PRISTINE)
    oc = ObjectClassifier()
    print(f"VerbNet: {len(members)} members; vector class table {len(oc.vec)} lemmas; polysemy={a.polysemy}")
    for p in a.files:
        text = open(p, encoding="utf-8").read()
        stats = collections.Counter()
        new = relabel(text, members, gold, oc, a.polysemy, stats)
        if a.inplace and new != text:
            open(p, "w", encoding="utf-8").write(new)
        print(f"{p}{'' if a.inplace else ' (dry run)'}: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
