#!/usr/bin/env python3
"""Decide English verb-level `comp:obl` vs `mod` (successor of relabel_vnobj.py, user rulings of 2026-10-26).

Same targets as relabel_vnobj.py (every ADP the SUD treebank marks `udep` under a VERB, currently mod or
comp:obl). Object classes as relabel_vnobj.py, with two repairs (`objclass`): WordNet and the vector table
are consulted only for a nominal object (NOUN / PROPN / NUM), and a WordNet first sense in noun.time counts
only if at least half of the noun senses of the lemma are noun.time (grave, history, cycle are not times).
Per token, first match of:

  1. ORIGIN `from` -> mod. `from` with a place-class object, whatever VerbNet says about the verb and
     whether or not the verb is in VerbNet (ruling 2: origin `from` is a modifier, all of it).
  2. NARROW DIRECTIONAL -> comp:obl (`narrowdir`; the old blanket directional rule is gone, `nodir`): a
     goal / source preposition (GOAL_SOURCE_P) or path preposition (PATH_P) under a verb that is a member of
     a motion, caused-motion, transfer / change-of-possession or removal class (MOTION_CLASSES). Comes before
     the circumstant rules; a time object stops it except under `through` (walk me through June;
     `throughtime`). The light verb `take place` has no class membership (`takeplace`): its PPs are
     circumstants or mod. Any other directional
     PP goes through 3-7 like any other PP.
  3. CIRCUMSTANT -> mod: fronted, discourse, FIXED PP (a Wiktionary prepositional-phrase entry that is the
     whole PP: `in the end`, `on a whim`), time, place, occasion, manner, reason (as relabel_vnobj.py).
     Exception, LOCATION SELECTION: a place PP (a place object under a locative preposition, or under a
     directional one, `dirlocsel`) is comp:obl when a frame of one of the verb's VerbNet classes takes it
     in a Location / Destination / Goal / Source / Path ... slot (frame matched to the token's shape, see
     6). Occurrence and existence classes do NOT select locations (ruling 1): OCC_EXIST_CLASSES below,
     so `happen in`, `exist in`, `reside in` fall back to the place rule. A time-class object is not
     overridden by a licence unless --with timesel / timesel-any is given (measured to over-fire).
  4. NOT IN VERBNET: a strict Wiktionary verb+preposition idiom -> comp:obl, else the current label is kept.
  5. Existential `be` -> mod.
  6. VERBNET LICENCE, MAJORITY: comp:obl iff at least half of the verb's VerbNet classes have a frame that
     matches the token's shape and gives the PP a licensing role (not Instrument / Beneficiary / Time /
     Manner / Extent / Context). FRAME SHAPE: the number of NP objects (a passive counts its subject),
     the verb's PP and locative-adverb dependents filled into the frame's PP slots one per slot, left to right.
  7. GAP FILLING for a PP the majority does not license: comp:obl if the pair is a strict Wiktionary idiom
     (verb+P, or verb + its object noun + P), or the verb key and P are associated in a corpus (G2 >= 10.83,
     count >= 5, PMI > 0); otherwise mod.

The lexicons for 3, 4 and 7 are in assets/en_vnrulings_lexicon.json (provenance inside the file).

    relabel_vnrulings.py FILE...                 # dry run: counts only
    relabel_vnrulings.py --inplace FILE...       # rewrite DEPREL of the targets (never MISC)
    relabel_vnrulings.py --log OUT.tsv FILE...   # per-token decisions
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_verbnet as R                      # noqa: E402
import relabel_vnobj as V                        # noqa: E402
from restore_noun_of import DEFAULT_PRISTINE, load_pristine   # noqa: E402

LEXICON = HERE.parent / "assets/en_vnrulings_lexicon.json"
NONLIC = {"Instrument", "Beneficiary", "Time", "Manner", "Extent", "Context"}
LOCROLES = {"Location", "Destination", "Goal", "Initial_Location", "Source", "Trajectory", "Path"}
LOCADV = {"there", "here", "home", "away", "back", "abroad", "upstairs", "downstairs", "inside", "outside", "somewhere",
          "anywhere", "everywhere", "nowhere", "elsewhere", "where", "downtown", "overseas", "out", "in", "up", "down"}
G2_MIN = 10.83
# Ruling 1: occurrence / existence classes do not select locations. Levin 47 (existence) and 48 (appearance,
# disappearance, occurrence) with their subclasses, except posture (spatial_configuration-47.6), plus lodge-46
# (residence: reside, dwell, live, stay, settle). Placement, posture and motion classes keep selecting.
OCC_EXIST_CLASSES = ("lodge-46", "exist-47.1", "entity_specific_modes_being-47.2", "modes_of_being_with_motion-47.3",
                     "sound_existence-47.4", "swarm-47.5.1", "herd-47.5.2", "bulge-47.5.3", "meander-47.7",
                     "contiguous_location-47.8", "terminus-47.9", "appear-48.1.1", "reflexive_appearance-48.1.2", "disappearance-48.2",
                     "occur-48.3")


def occ_exist(cid):
    """True for an OCC_EXIST_CLASSES class or one of its subclasses (exist-47.1-1 ...)."""
    return any(cid == c or cid.startswith(c + "-") for c in OCC_EXIST_CLASSES)


def load_frames(vn_dir):
    """member -> [(class id, [frame dict(nobj, pps=[(preps, role)], advs=[role])])], in load_verbnet order."""
    mem = collections.defaultdict(list)

    def walk(node, inherited):
        frames = list(inherited)
        for fr in node.findall("./FRAMES/FRAME"):
            syn = list(fr.find("SYNTAX"))
            seen_v, nps, pps, advs, i = False, 0, [], [], 0
            while i < len(syn):
                e = syn[i]
                if e.tag == "VERB":
                    seen_v = True
                elif seen_v and e.tag == "PREP":
                    role = syn[i + 1].get("value", "").lstrip("?") if i + 1 < len(syn) and syn[i + 1].tag == "NP" else ""
                    pps.append((frozenset(R._prep_set(e)), role))
                    i += 1
                elif seen_v and e.tag == "NP":
                    role = e.get("value", "").lstrip("?")
                    if role in LOCROLES:
                        advs.append(role)
                    else:
                        nps += 1
                i += 1
            frames.append(dict(nobj=nps, pps=pps, advs=advs))
        for m in node.findall("./MEMBERS/MEMBER"):
            mem[m.get("name").lower()].append((node.get("ID"), frames))
        for sub in node.findall("./SUBCLASSES/VNSUBCLASS"):
            walk(sub, frames)

    for f in sorted(glob.glob(os.path.join(vn_dir, "*.xml"))):
        walk(ET.parse(f).getroot(), [])
    return mem


def match_frame(fr, ctx, tid):
    """Role the frame gives token `tid` once the verb's dependents fill its slots, or None."""
    if min(fr["nobj"], 2) != min(ctx["nobj"], 2):
        return None
    used, used_adv = [False] * len(fr["pps"]), [False] * len(fr["advs"])
    for pos, kind, prep in sorted(ctx["deps"]):
        slot = None
        if kind == "pp":
            for j, (ps, role) in enumerate(fr["pps"]):
                if not used[j] and prep in ps:
                    slot = ("pp", j, role)
                    break
        else:
            for j, role in enumerate(fr["advs"]):
                if not used_adv[j]:
                    slot = ("adv", j, role)
                    break
            if slot is None:
                for j, (ps, role) in enumerate(fr["pps"]):
                    if not used[j] and role in LOCROLES:
                        slot = ("pp", j, role)
                        break
        if slot is None:
            continue
        (used if slot[0] == "pp" else used_adv)[slot[1]] = True
        if pos == tid:
            return slot[2]
    return None


def class_roles(classes, ctx, tid):
    """Per class: the best role (a licensing role preferred), None if no frame matches."""
    out = []
    for cid, frames in classes:
        roles = [r for r in (match_frame(fr, ctx, tid) for fr in frames) if r is not None]
        lic = [r for r in roles if r not in NONLIC]
        out.append((cid, lic[0] if lic else (roles[0] if roles else None)))
    return out


def shape_context(lines, f):
    """Frame-matching context of the verb heading target row f: objects, passive, PP/adverb dependents."""
    rows = {}
    for ln in lines:
        if ln and not ln.startswith("#"):
            c = ln.split("\t")
            if len(c) == 10 and c[0].isdigit():
                rows[c[0]] = c
    kids = collections.defaultdict(list)
    for c in rows.values():
        kids[c[6]].append(c)
    v = rows[f[6]]
    nobj = sum(1 for k in kids[v[0]] if k[7].split("@")[0] == "comp:obj")
    passive = "Voice=Pass" in v[5] or v[7].endswith("@pass") or (
        v[4] == "VBN" and v[6] in rows and rows[v[6]][3] in ("NOUN", "PROPN", "PRON") and v[7].split("@")[0] in ("mod", "udep"))
    deps = []
    for k in kids[v[0]]:
        if k[3] == "ADP" and not k[7].startswith("compound") and "agent" not in k[7]:
            deps.append((int(k[0]), "pp", k[2].lower()))
        elif k[3] == "ADV" and k[2].lower() in LOCADV and not k[7].startswith("compound"):
            deps.append((int(k[0]), "adv", k[2].lower()))
    vobjs = [k[2].lower() for k in kids[v[0]] if k[7].split("@")[0] == "comp:obj"]
    if passive and v[7].split("@")[0] in ("mod", "udep") and v[6] in rows:
        vobjs.append(rows[v[6]][2].lower())
    if passive:
        vobjs += [k[2].lower() for k in rows.values() if k[7] == "subj@pass" and k[6] in (v[6], v[0])]
    takeplace = v[2].lower() == "take" and any(
        k[2].lower() == "place" and k[7].split("@")[0] in ("comp:obj", "compound") for k in kids[v[0]])
    return dict(nobj=nobj + int(passive), deps=deps, vobjs=vobjs, tid=int(f[0]), takeplace=takeplace)


class Lexicon:
    def __init__(self, path=LEXICON):
        d = json.load(open(path, encoding="utf-8"))
        self.vp = set(d["idioms_vp"])
        self.vnp = {tuple(s.split()) for s in d["idioms_vnp"]}
        self.g2 = set(d["g2_pairs"])
        self.fixed = set(d["fixed_pps"])


def ikey(t):
    return f"{t['lemma']} {t['prt']} {t['prep']}" if t["prt"] else f"{t['lemma']} {t['prep']}"


def circumstant(t, ocls, lex):
    prep, obj = t["prep"], t["obj"]
    olem = obj["lemma"] if obj else None
    bare = obj is not None and not any(d.startswith("det") or "poss" in d for _, d in obj["kids"])
    if t["left"] and obj is not None and olem not in V.REL and t["pp"].rstrip().endswith(","):
        return "fronted"
    if (prep, olem) in R.DISCOURSE_PPS and (olem not in ("fact", "case") or bare):
        return "discourse"
    if lex is not None and re.sub(r"\s+", " ", t["pp"].strip().lower()) in lex.fixed:
        return "fixedpp"
    if V._temporal_repaired(t["temporal"], obj, prep, ocls) or (ocls == "time" and prep in V.TEMPORAL_P):
        return "time"
    if ocls == "place" and prep in V.LOCATIVE:
        return "place"
    if ocls == "event" and prep in V.EVENT_P:
        return "occasion"
    if olem in V.MANNER_O and prep in ("in", "with"):
        return "manner"
    if (olem in V.REASON_O and prep in ("for", "on", "out")) or prep in V.REASON_P:
        return "reason"
    return None


RULINGS = ("occexist", "origin", "fixedpp", "objclass", "nodir", "dirlocsel", "narrowdir", "throughtime", "takeplace")
# Narrow directional rule (ruling of 2026-10-26): a goal / source / path preposition is comp:obl when the verb
# is a member of a motion, caused-motion, transfer / change-of-possession or removal class (or a subclass).
GOAL_SOURCE_P = {"to", "into", "onto", "out", "off", "from", "toward", "towards"}
PATH_P = {"through", "across", "along", "past", "via", "up", "down", "over"}   # not around/round: around noon
MOTION_CLASSES = ("put-9.1", "put_spatial-9.2", "funnel-9.3", "put_direction-9.4", "pour-9.5", "pocket-9.10",
                  "remove-10.1", "banish-10.2", "clear-10.3", "wipe_manner-10.4.1", "wipe_instr-10.4.2", "steal-10.5",
                  "cheat-10.6.1", "deprive-10.6.2", "free-10.6.3", "rob-10.6.4", "pit-10.7", "debone-10.8", "mine-10.9",
                  "fire-10.10", "resign-10.11",
                  "send-11.1", "slide-11.2", "bring-11.3", "carry-11.4", "drive-11.5", "push-12",
                  "give-13.1", "contribute-13.2", "future_having-13.3", "fulfilling-13.4.1", "equip-13.4.2",
                  "get-13.5.1", "obtain-13.5.2", "hire-13.5.3", "invest-13.5.4", "exchange-13.6.1", "substitute-13.6.2",
                  "berry-13.7", "throw-17.1",
                  "escape-51.1", "leave-51.2", "roll-51.3.1", "run-51.3.2", "vehicle-51.4.1", "nonvehicle-51.4.2",
                  "vehicle_path-51.4.3", "waltz-51.5", "chase-51.6", "accompany-51.7", "reach-51.8")


def motion_class(cid):
    return any(cid == c or cid.startswith(c + "-") for c in MOTION_CLASSES)
OPTIONAL = ("timesel", "timesel-any")            # off by default; switched on with --with


class ObjectClassifier(V.ObjectClassifier):
    """relabel_vnobj's classifier with two repairs to WordNet-based classes (ruling of 2026-10-26, `objclass`):
    WordNet and the vector table are consulted only for a common-noun object (a PP whose object slot holds an
    adverb or preposition, `from all over X`, was classed by the cricket sense of `over`), and a first sense in
    noun.time counts only when at least half of the noun senses of the lemma are noun.time (grave, history, cycle)."""

    _skip = None

    def wn_class(self, w):
        if w == self._skip:
            return None
        if w not in self._wn:
            ss = self.wn.synsets(w, "n")
            c = V.ObjectClassifier.wn_class(self, w)
            if c == "time" and 2 * sum(1 for x in ss if x.lexname() == "noun.time") < len(ss):
                s = ss[0]
                hyps = set(s.closure(lambda t: t.hypernyms() + t.instance_hypernyms())) | {s}
                c = "place" if hyps & self.place_hyp else "other"
            self._wn[w] = c
        return self._wn[w]

    def classify(self, obj, sent):
        if obj is None or obj["upos"] in ("NOUN", "PROPN", "NUM"):
            return V.ObjectClassifier.classify(self, obj, sent)
        saved, self.vec, self._skip = self.vec, {}, obj["lemma"]
        try:
            return V.ObjectClassifier.classify(self, obj, sent)
        finally:
            self.vec, self._skip = saved, None


def licence(t, ctx, classes, roles, nlic, lex):
    """The non-circumstant tail: VerbNet majority over matched frames, then the idiom / G2 gap filling."""
    prep, key = t["prep"], t["key"]
    if not classes:
        return ("idiom", "comp:obl") if ikey(t) in lex.vp else ("novn", None)
    if t["lemma"] == "be" and t["expl"]:
        return "vn:existential", "mod"
    n = sum(1 for _, r in roles if r and r not in NONLIC)
    if 2 * n >= len(classes):
        return "vn", "comp:obl"
    if ikey(t) in lex.vp or any((t["lemma"], o, prep) in lex.vnp for o in ctx["vobjs"]):
        return "gap:wiktionary", "comp:obl"
    if f"{key}|{prep}" in lex.g2:
        return "gap:g2", "comp:obl"
    return ("vn-unlic-shape" if nlic > 0 else "vn-unlic"), "mod"


def decide(t, ctx, members, frames, oc, lex, rulings=RULINGS):
    """-> (source, label); label None = keep the current one."""
    prep, key = t["prep"], t["key"]
    ocls = oc.classify(t["obj"], t["sent"])
    classes = members.get(key) or []
    nlic = sum(1 for _, lic in classes if prep in lic)
    if prep == "from" and ocls == "place" and ("origin" in rulings or (classes and nlic == 0)):
        return "circ:origin", "mod"
    if "nodir" not in rulings and prep in V.DIRECTIONAL and ocls != "time":
        return "dir", "comp:obl"
    takeplace = "takeplace" in rulings and ctx["takeplace"]
    if takeplace:
        classes = []          # light verb `take place`: no class membership licenses anything (not removal)
    if "narrowdir" in rulings and (prep in GOAL_SOURCE_P or prep in PATH_P) \
            and any(motion_class(cid) for cid, _ in classes) \
            and (ocls != "time" or prep == "through" or "throughtime" not in rulings):
        return "dir:motion", "comp:obl"            # before the time rule: walk me through June
    roles = [] if takeplace else class_roles(frames.get(key) or [], ctx, ctx["tid"])
    ct = circumstant(t, ocls, lex if "fixedpp" in rulings else None)
    placepp = ct == "place" or ("dirlocsel" in rulings and ct is None and ocls == "place" and prep in V.DIRECTIONAL)
    if placepp and any(r in LOCROLES for cid, r in roles if not ("occexist" in rulings and occ_exist(cid))):
        return "vn:locsel", "comp:obl"
    lic = ("takeplace-unlic", "mod") if takeplace else licence(t, ctx, classes, roles, nlic, lex)
    if ct == "time" and prep not in R.TEMP_PREPS_ALWAYS and lic[1] == "comp:obl" and (
            "timesel" in rulings and lic[0] in ("idiom", "gap:wiktionary") or "timesel-any" in rulings):
        return lic[0] + ":timesel", "comp:obl"     # an idiom whose object denotes a period or date (wait for)
    if ct:
        return "circ:" + ct, "mod"
    return lic


def relabel(text, members, frames, gold, oc, lex, stats, rulings, log=None):
    out = []
    for blk in text.split("\n\n"):
        lines = None
        sid = next((ln.split("=", 1)[1].strip() for ln in blk.split("\n") if ln.startswith("# sent_id")), "")
        for lines_, i, f, t in V.targets(blk, gold):
            lines = lines_
            ctx = shape_context(lines, f)
            src, lab = decide(t, ctx, members, frames, oc, lex, rulings)
            stats[src] += 1
            if log is not None:
                log.append((sid, f[0], t["key"], t["prep"], f[7], lab or f[7], src, t["pp"]))
            if lab is not None and lab != f[7]:
                stats[f"{f[7]} -> {lab}"] += 1
                f[7] = lab                       # DEPREL only; MISC is never written
                lines[i] = "\t".join(f)
        out.append(blk if lines is None else "\n".join(lines))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--verbnet", default=R.DEFAULT_VN)
    ap.add_argument("--lexicon", default=str(LEXICON))
    ap.add_argument("--without", action="append", default=[], choices=RULINGS,
                    help="switch a 2026-10-26 ruling off (for comparison)")
    ap.add_argument("--with", dest="with_", action="append", default=[], choices=OPTIONAL,
                    help="timesel-any: any licence (VerbNet majority, idiom, G2) beats a time-class object")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--log", help="write per-token decisions (TSV) here")
    a = ap.parse_args()
    rulings = tuple(r for r in RULINGS if r not in a.without) + tuple(a.with_)
    members = R.load_verbnet(a.verbnet)
    frames = load_frames(a.verbnet)
    gold = load_pristine(DEFAULT_PRISTINE)
    oc = ObjectClassifier() if "objclass" in rulings else V.ObjectClassifier()
    lex = Lexicon(a.lexicon)
    print(f"VerbNet: {len(members)} members; lexicon: {len(lex.vp)} verb+P idioms, {len(lex.vnp)} verb+noun+P idioms, "
          f"{len(lex.g2)} G2 pairs, {len(lex.fixed)} fixed PPs; rulings {', '.join(rulings) or 'none'}")
    log = [] if a.log else None
    for p in a.files:
        text = open(p, encoding="utf-8").read()
        stats = collections.Counter()
        new = relabel(text, members, frames, gold, oc, lex, stats, rulings, log)
        if a.inplace and new != text:
            open(p, "w", encoding="utf-8").write(new)
        print(f"{p}{'' if a.inplace else ' (dry run)'}: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    if log is not None:
        with open(a.log, "w", encoding="utf-8") as fh:
            for row in log:
                fh.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
