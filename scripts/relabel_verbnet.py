#!/usr/bin/env python3
"""Decide English verb-level `comp:obl` vs `mod` from VerbNet, not token by token.

WHY. `relabel_ext.py` asked qwen3:8b, per token, whether a PP under a verb was a complement or a
modifier. Measured on EWT+GUM train (2026-10-25) the result was only 85.8% pure per (verb lemma,
preposition), 78.0% among pairs seen 20-99 times, and a hand check found about 15 of 24 sampled
minority labels arbitrary ("thank you for your time" both ways, "look for" both ways in one sense).
By user decision the distinction now follows a published valency lexicon.

WHAT. For every token the SUD treebank marks `udep` that is an ADP headed by a VERB (noun-level `of`
is not touched; it keeps `udep`), in this order:

  b. RULE -> mod. A temporal/circumstantial PP, using build_gold.py's rules: during/throughout/
     amid(st) always; in/on/at/by/over/within/since/until/before/after -- and `for` (duration), which
     those rules lacked -- when the PP's object is a temporal noun, a year, a decade or a clock
     time; and fixed discourse/adverbial PPs (DISCOURSE_PPS: for example, in fact, at least, by
     mistake ...), added after a hand check of 50 changed tokens. Overrides (a).
  a. VERBNET -> comp:obl. The verb's VerbNet 3.3 classes license the preposition in a syntactic
     frame (see `licensed`). Polysemy: `--polysemy majority` (default) requires the preposition to
     be licensed in at least half of the classes the lemma is a member of; `union` (any class)
     over-generates and is kept for comparison. A verb with a particle (`compound@prt`) is looked up
     as `lemma_particle` (VerbNet has 236 such members: take_on, carry_out ...); if that member
     does not exist the token is residue.
  c. RESIDUE -> the LLM's existing label (verb not in VerbNet, pair not licensed, or particle verb
     not listed),
  d. ... SMOOTHED to the TRAIN majority of the residue labels for its (verb key, preposition);
     ties and types unseen in train keep their own label. (a) is constant per type by construction
     and (b) is the rule-conditioned exception, so only (c) is smoothed.

PREPOSITION CLASSES. A frame names its preposition literally (`<PREP value="to towards">`, `?` =
optional, `|` = alternatives, `out_of` -> `out`) or by selectional restriction, expanded with
VerbNet's preposition hierarchy (PREP_CLASSES below); a `-` restriction subtracts. A frame's PREP
with neither is ignored. Frames are inherited by subclasses.

DATA. VerbNet 3.3 is NOT in this repo: NLTK data package `verbnet3` (482,025-byte zip, sha256
fa0136a7699c52f0bd532dc5adc0914745aa4369a52ae1465cb11841060ec1de), unpacked at
../lexicons/verbnet3/verbnet3/*.xml beside this repo; override with --verbnet or $VERBNET_DIR.

    relabel_verbnet.py FILE...                         # dry run: report only, writes nothing
    relabel_verbnet.py --inplace FILE...               # rewrite DEPREL of the targets
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_VN = os.environ.get("VERBNET_DIR", str(HERE.parent / "lexicons/verbnet3/verbnet3"))
TRAIN = str(HERE / "assets/en_ewtgum-sud-train.relabeled_ext.conllu")

sys.path.insert(0, str(HERE / "scripts"))
from restore_noun_of import load_pristine, DEFAULT_PRISTINE  # noqa: E402

# VerbNet's preposition hierarchy (verbs.colorado.edu VerbNet reference, "prepositions").
_DEST_CONF = {"into", "onto"}
_DEST_DIR = {"for", "at", "to", "towards", "toward"}
_DIR = {"across", "along", "around", "down", "over", "past", "round", "through", "towards",
        "toward", "up"}
_SRC = {"from", "out", "off"}
_LOC = {"about", "above", "against", "along", "alongside", "amid", "among", "amongst", "around",
        "astride", "at", "athwart", "atop", "before", "behind", "below", "beneath", "beside",
        "besides", "between", "beyond", "by", "from", "in", "inside", "near", "next", "off", "on",
        "opposite", "out", "outside", "over", "past", "round", "through", "throughout", "under",
        "underneath", "upon", "within"}
PREP_CLASSES = {"dest_conf": _DEST_CONF, "dest_dir": _DEST_DIR, "dest": _DEST_CONF | _DEST_DIR,
                "dir": _DIR | _DEST_CONF | _DEST_DIR, "src": _SRC,
                "path": _SRC | _DIR | _DEST_CONF | _DEST_DIR, "loc": _LOC,
                "spatial": _LOC | _SRC | _DIR | _DEST_CONF | _DEST_DIR}

TEMP_PREPS_ALWAYS = {"during", "throughout", "amid", "amidst"}
TEMP_OBJ_PREPS = {"in", "on", "at", "by", "over", "within", "since", "until", "before", "after",
                  "for"}
TEMP_NOUNS = {
    "morning", "afternoon", "evening", "night", "midnight", "noon", "day", "week", "weekend",
    "fortnight", "month", "year", "decade", "century", "hour", "minute", "moment", "dawn", "dusk",
    "today", "tomorrow", "yesterday", "spring", "summer", "autumn", "fall", "winter", "christmas",
    "easter", "time", "era", "age", "period", "future", "past", "outset", "beginning", "end",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december"}
# Time objects the noun list cannot name: decades ('80s, 1980s), clock times (1:00, 3 pm), numeric dates (11-20-2000).
TEMP_OBJ_FORM = re.compile(r"'?\d0'?s|\d{3}0'?s|\d{1,2}:\d\d|\d{1,2}(am|pm|a\.m\.|p\.m\.)|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", re.I)
TEMP_OBJ_LEMMAS = {"am", "pm", "a.m.", "p.m.", "o'clock", "noon", "midnight"}
# Discourse / sentence adverbial PPs: fixed prep + object lemma, never a verb's complement.
DISCOURSE_PPS = {("for", "example"), ("for", "instance"), ("in", "fact"), ("in", "general"),
                 ("in", "particular"), ("in", "addition"), ("in", "short"), ("in", "total"),
                 ("at", "least"), ("at", "most"), ("at", "all"), ("at", "first"), ("at", "last"),
                 ("by", "way"), ("of", "course"), ("in", "case"), ("by", "mistake"),
                 ("by", "accident"), ("on", "purpose"), ("in", "way"), ("in", "manner"),
                 ("in", "search"), ("for", "sake")}


def _prep_set(prep):
    v = prep.get("value")
    if v:
        out = set()
        for w in re.split(r"[\s|]+", v):
            w = w.strip("?").lower()
            if w:
                out.add(w.split("_")[0])
        return out
    plus, minus = set(), set()
    for s in prep.iter("SELRESTR"):
        cls = PREP_CLASSES.get(s.get("type"), set())
        (plus if s.get("Value") == "+" else minus).update(cls)
    if not plus and minus:
        plus = set(PREP_CLASSES["spatial"])
    return plus - minus


def load_verbnet(vn_dir):
    """member (lowercase, particles joined by `_`) -> list of (class id, licensed prepositions)."""
    members = collections.defaultdict(list)
    files = sorted(glob.glob(os.path.join(vn_dir, "*.xml")))
    if not files:
        sys.exit(f"no VerbNet XML under {vn_dir} (see the DATA note in this script)")

    def walk(node, inherited):
        lic = set(inherited)
        for fr in node.findall("./FRAMES/FRAME"):
            for p in fr.iter("PREP"):
                lic |= _prep_set(p)
        for m in node.findall("./MEMBERS/MEMBER"):
            members[m.get("name").lower()].append((node.get("ID"), frozenset(lic)))
        for sub in node.findall("./SUBCLASSES/VNSUBCLASS"):
            walk(sub, lic)

    for f in files:
        walk(ET.parse(f).getroot(), set())
    return members


def licensed(members, key, prep, polysemy):
    cls = members.get(key)
    if not cls:
        return None                      # not in VerbNet
    n = sum(1 for _, lic in cls if prep in lic)
    if polysemy == "union":
        return n > 0
    return 2 * n >= len(cls)


def sentences(text):
    for blk in text.split("\n\n"):
        lines = blk.split("\n")
        sid = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("# sent_id")), None)
        rows = {}
        for i, l in enumerate(lines):
            if l and not l.startswith("#"):
                f = l.split("\t")
                if len(f) == 10 and f[0].isdigit():
                    rows[f[0]] = (i, f)
        yield blk, lines, sid, rows


def targets(text, gold):
    """Yield (lines, line index, columns, verb key, particle, prep, temporal) per target."""
    for blk, lines, sid, rows in sentences(text):
        kids = collections.defaultdict(list)
        for tid, (_, f) in rows.items():
            kids[f[6]].append(f)
        for tid, (i, f) in rows.items():
            if f[3] != "ADP" or f[6] not in rows:
                continue
            h = rows[f[6]][1]
            if h[3] != "VERB":
                continue
            g = gold.get((sid, f[0]))
            if g is None or g[0] != f[1] or g[1] != f[6] or g[2] != "udep":
                continue
            if f[7] not in ("mod", "comp:obl"):
                continue                 # e.g. fix_seem_like_comppred's `comp:pred`: not ours
            prt = next((k[2].lower() for k in kids[h[0]] if k[7].startswith("compound@prt")), None)
            prep = f[2].lower()
            obj = next((k for k in kids[f[0]] if k[7].startswith("comp:obj")), None) or \
                next((k for k in kids[f[0]] if k[3] in ("NOUN", "PROPN", "NUM", "PRON")), None)
            olem = obj[2].lower() if obj is not None else None
            temporal = prep in TEMP_PREPS_ALWAYS or (prep, olem) in DISCOURSE_PPS or (
                prep in TEMP_OBJ_PREPS and obj is not None and
                (olem in TEMP_NOUNS or (re.fullmatch(r"\d{4}", obj[1]) and 1000 <= int(obj[1]) <= 2100)
                 or TEMP_OBJ_FORM.fullmatch(obj[1])
                 or any(k[2].lower() in TEMP_OBJ_LEMMAS for k in kids[obj[0]])))
            yield lines, i, f, h[2].lower(), prt, prep, temporal


def decide(members, lemma, prt, prep, temporal, polysemy):
    """-> (source, label or None); None = residue (keep / smooth the LLM label)."""
    if temporal:
        return "b_rule", "mod"
    key = f"{lemma}_{prt}" if prt else lemma
    if prt and key not in members:
        return "c_residue", None
    lic = licensed(members, key, prep, polysemy)
    if lic:
        return "a_verbnet", "comp:obl"
    return "c_residue", None


def residue_table(members, train_text, gold, polysemy):
    t = collections.defaultdict(collections.Counter)
    for lines, i, f, lemma, prt, prep, temporal in targets(train_text, gold):
        src, lab = decide(members, lemma, prt, prep, temporal, polysemy)
        if lab is None and f[7] in ("mod", "comp:obl"):
            t[(f"{lemma}_{prt}" if prt else lemma, prep)][f[7]] += 1
    out = {}
    for k, c in t.items():
        top = c.most_common(2)
        if len(top) == 1 or top[0][1] != top[1][1]:
            out[k] = top[0][0]
    return out


def rewrite(text, members, gold, table, polysemy, stats):
    """Block by block; only the DEPREL cell of a target changes."""
    return "\n\n".join(_relabel_block(blk, members, gold, table, polysemy, stats)
                       for blk in text.split("\n\n"))


def _relabel_block(blk, members, gold, table, polysemy, stats):
    lines = None
    for lines_, i, f, lemma, prt, prep, temporal in targets(blk, gold):
        lines = lines_
        src, lab = decide(members, lemma, prt, prep, temporal, polysemy)
        if lab is None:
            k = (f"{lemma}_{prt}" if prt else lemma, prep)
            t = table.get(k)
            if t is None:
                lab = f[7]
            else:
                lab = t
                if t != f[7]:
                    src = "d_smoothed"
        stats[src] += 1
        if lab != f[7]:
            stats[f"{f[7]} -> {lab}"] += 1
            f[7] = lab
            lines[i] = "\t".join(f)
    return blk if lines is None else "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--verbnet", default=DEFAULT_VN)
    ap.add_argument("--train", default=TRAIN, help="file whose residue labels give the smoothing table")
    ap.add_argument("--polysemy", choices=("majority", "union"), default="majority")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--dump-table", help="write the smoothing table as JSON")
    a = ap.parse_args()
    members = load_verbnet(a.verbnet)
    gold = load_pristine(DEFAULT_PRISTINE)
    table = residue_table(members, open(a.train, encoding="utf-8").read(), gold, a.polysemy)
    print(f"VerbNet: {len(members)} members from {a.verbnet}; smoothing table {len(table)} types "
          f"from {a.train}; polysemy={a.polysemy}")
    if a.dump_table:
        json.dump({"|".join(k): v for k, v in sorted(table.items())}, open(a.dump_table, "w"), indent=0)
    for p in a.files:
        text = open(p, encoding="utf-8").read()
        stats = collections.Counter()
        new = rewrite(text, members, gold, table, a.polysemy, stats)
        if a.inplace and new != text:
            open(p, "w", encoding="utf-8").write(new)
        print(f"{p}{'' if a.inplace else ' (dry run)'}: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
