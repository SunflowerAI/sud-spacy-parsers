#!/usr/bin/env python3
"""Four typological features per treebank, measured from its own trees. TRAINING LANGUAGES ONLY.

    OV / VO          order of object and verb          (WALS 83A)
    SV / VS          order of subject and verb         (WALS 82A)
    HM / DM          head-marking / dependent-marking  (WALS 23A: verbal agreement vs nominal case)
    SEX / NOSEX      sex-based noun classification     (WALS 31A)

Each is a 2-BIT FIELD and the bits are NOT mutually exclusive: `11` means both orders (or both loci)
are attested, `10`/`01` means one, and **`00` means UNKNOWN** -- too little evidence to say. Eight
bits in total.

⚠ **THIS SCRIPT MUST NEVER BE RUN OVER A TEST LANGUAGE.** A profile derived from a treebank's own
gold trees is an ORACLE for a language the parser is supposed to have never seen, and that is exactly
what left v1's +12.74 zero-shot result ungated. Test profiles come from `typology_external.py`
(Grambank/WALS/literature) and `check_generic_inputs_v2.py` refuses a corpus in which any test
language carries a bit with `source == "treebank"`.

⚠ **SUD MAKES THE ADPOSITION THE HEAD OF ITS COMPLEMENT**, so there is no UD-style `case` relation to
count and adpositional marking is NOT evidence for dependent-marking here. WALS 23A counts affixal
case anyway, so the two agree on this by accident rather than by design; say so rather than letting
the next reader look for the missing relation.

Measurement is separated from thresholding on purpose: `measure()` does the expensive pass and
`bits_from()` is pure, so `--sensitivity` can re-threshold the whole release without re-reading it.

Three guards below each exist because the naive predicate was measured and was wrong:

  * **the core-case guard.** Classical Chinese carries `Case` on 31 % of its nouns -- `Case=Loc`
    37 529 and `Case=Tem` 7 688, and nothing else. Those are semantic labels on locative and temporal
    nouns, not morphological case, and without the guard an isolating language is called
    dependent-marking.
  * **the VerbForm=Fin fallback.** Sanskrit-Vedic annotates ZERO `VerbForm=Fin` while carrying Person
    on ~90 % of its verbs. Restricted to finite verbs the predicate finds no denominator and calls
    Sanskrit unmarked.
  * **`Com` counts as sex-based.** Tamil's noun genders are Neut 2 269 / Com 526 / Masc 1, and WALS
    codes Tamil sex-based; Danish and Swedish common-vs-neuter have the same shape.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from prep_generic import strip_subtype  # noqa: E402

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]

#: A `Case` inventory that touches none of these is not morphological case on core arguments. See
#: the Classical Chinese note above.
CORE_CASES = {"Nom", "Acc", "Erg", "Abs"}

#: `Com` (common gender) belongs here: a common-vs-neuter system IS sex-based in origin and WALS
#: codes it so.
SEX_GENDERS = {"Masc", "Fem", "Com"}

#: Where to look for evidence of sex-based NOUN classification. DET and ADJ are in because gender on
#: them is agreement with a noun; PRON is out because referential pronoun gender (English he/she) is
#: not a classification of nouns. The rate is taken as the MAXIMUM over these three rather than
#: pooled, since which one carries the annotation is a treebank policy, not a fact about the
#: language: French GSD marks 71 % of ADJ and 0 % of NOUN.
GENDER_DOMAIN = ("NOUN", "DET", "ADJ")


def feats_of(col: str) -> dict:
    if col == "_":
        return {}
    out = {}
    for kv in col.split("|"):
        k, _, v = kv.partition("=")
        if k:
            out[k] = v
    return out


def measure(paths, exclude_pron_obj=True):
    """One pass over a treebank's CoNLL-U. Returns raw counts; no thresholds are applied here."""
    m = {
        "obj_before": 0, "obj_n": 0,
        "subj_before": 0, "subj_n": 0,
        "verb_n": 0, "verb_fin_n": 0,
        # Person/Number are counted TWICE, once over all VERB and once over the finite ones only,
        # so the ratio always divides a count by the set it was drawn from.
        "vperson_all": 0, "vnumber_all": 0,
        "vperson_fin": 0, "vnumber_fin": 0,
        "person_all": collections.Counter(), "number_all": collections.Counter(),
        "person_fin": collections.Counter(), "number_fin": collections.Counter(),
        "noun_n": 0, "noun_case_n": 0, "noun_feats_n": 0,
        "case_vals": collections.Counter(), "gender_vals": collections.Counter(),
        # Gender is counted per UPOS. Many treebanks annotate it ONLY on the agreeing words:
        # French GSD and Arabic PADT put Gender on 63-100 % of DET and ADJ and on ZERO nouns, so a
        # NOUN-only predicate calls French and Arabic genderless. Agreement on a determiner or an
        # adjective IS evidence that nouns are classified, which is the feature being measured.
        "gdom_tot": collections.Counter(), "gdom_any": collections.Counter(),
        "gdom_sex": collections.Counter(), "gdom_class": collections.Counter(),
        # Diagnostic only, deliberately outside the domain: English marks Gender on 11 % of PRON
        # and nothing else, and referential pronoun gender is not noun CLASSIFICATION.
        "pron_gender_n": 0, "pron_n": 0,
    }
    for path in paths:
        rows, idx = [], {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    _flush(m, rows, idx, exclude_pron_obj)
                    rows, idx = [], {}
                    continue
                if line[0] == "#":
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8 or "-" in f[0] or "." in f[0]:
                    continue
                idx[f[0]] = len(rows)
                rows.append(f)
        _flush(m, rows, idx, exclude_pron_obj)
    return m


def _flush(m, rows, idx, exclude_pron_obj):
    if not rows:
        return
    for i, f in enumerate(rows):
        upos, head, dep = f[3], f[6], strip_subtype(f[7])
        ft = feats_of(f[5])

        if upos == "VERB":
            m["verb_n"] += 1
            fin = ft.get("VerbForm") == "Fin"
            m["verb_fin_n"] += fin
            if "Person" in ft:
                m["vperson_all"] += 1
                m["person_all"][ft["Person"]] += 1
                if fin:
                    m["vperson_fin"] += 1
                    m["person_fin"][ft["Person"]] += 1
            if "Number" in ft:
                m["vnumber_all"] += 1
                m["number_all"][ft["Number"]] += 1
                if fin:
                    m["vnumber_fin"] += 1
                    m["number_fin"][ft["Number"]] += 1
        elif upos in ("NOUN", "PROPN"):
            if upos == "NOUN":
                m["noun_n"] += 1
                if ft:
                    m["noun_feats_n"] += 1
                if "Case" in ft:
                    m["noun_case_n"] += 1
                    m["case_vals"][ft["Case"]] += 1
        elif upos == "PRON":
            m["pron_n"] += 1
            m["pron_gender_n"] += "Gender" in ft

        if upos in GENDER_DOMAIN:
            m["gdom_tot"][upos] += 1
            g = ft.get("Gender")
            if g:
                m["gender_vals"][g] += 1
                m["gdom_any"][upos] += 1
                if set(g.split(",")) & SEX_GENDERS:
                    m["gdom_sex"][upos] += 1
            if "NounClass" in ft or "Class" in ft:
                m["gdom_any"][upos] += 1
                m["gdom_class"][upos] += 1

        if head == "0" or head not in idx:
            continue
        h = rows[idx[head]]
        before = i < idx[head]

        if dep == "comp:obj" and h[3] == "VERB":
            if not (exclude_pron_obj and upos == "PRON"):
                m["obj_n"] += 1
                m["obj_before"] += before
        # AUX is required: SUD makes the auxiliary the head of its lexical verb, so the subject of a
        # periphrastic clause hangs off the AUX and a VERB-only predicate would miss it entirely.
        elif dep == "subj" and h[3] in ("VERB", "AUX"):
            m["subj_n"] += 1
            m["subj_before"] += before


def _spread(counter, n, floor=0.05):
    """How many values are attested on at least `floor` of `n`. One value is a constant, not a
    paradigm: a treebank that writes `Person=3` and nothing else is not evidence of agreement."""
    return sum(1 for v in counter.values() if n and v / n >= floor)


def bits_from(m, theta_minor=0.30, theta_mark=0.50,
              min_arcs=100, min_verbs=200, min_nouns=500, gender_floor=0.20):
    """Pure. Raw counts in, eight bits plus the intermediate proportions out."""
    b = dict.fromkeys(FIELDS, 0)
    raw = {}

    # --- order -------------------------------------------------------------------------------
    for pre, before, n in (("O", m["obj_before"], m["obj_n"]), ("S", m["subj_before"], m["subj_n"])):
        raw[f"p_{pre}V"] = round(before / n, 4) if n else None
        raw[f"n_{pre}V"] = n
        if n < min_arcs:
            continue                                  # both bits stay 0: UNKNOWN, not "in between"
        p = before / n
        if pre == "O":
            b["OV"], b["VO"] = int(p >= theta_minor), int(1 - p >= theta_minor)
        else:
            b["SV"], b["VS"] = int(p >= theta_minor), int(1 - p >= theta_minor)

    # --- head-marking ------------------------------------------------------------------------
    # Restrict to finite verbs where the treebank marks finiteness; where it marks none at all, every
    # VERB is the denominator (the Sanskrit case).
    # Use the finite verbs where the treebank marks finiteness on enough of them; where it marks
    # none at all -- Sanskrit-Vedic annotates ZERO VerbForm=Fin while carrying Person on ~90 % of
    # its verbs -- every VERB is the denominator, and the numerator moves with it.
    if m["verb_fin_n"] >= min_verbs:
        fin, nper, nnum = m["verb_fin_n"], m["vperson_fin"], m["vnumber_fin"]
        pvals, nvals, domain = m["person_fin"], m["number_fin"], "finite"
    else:
        fin, nper, nnum = m["verb_n"], m["vperson_all"], m["vnumber_all"]
        pvals, nvals, domain = m["person_all"], m["number_all"], "all-verbs"
    raw["n_verb"], raw["n_verb_fin"], raw["hm_domain"] = m["verb_n"], m["verb_fin_n"], domain
    raw["p_person"] = round(nper / fin, 4) if fin else None
    raw["p_vnumber"] = round(nnum / fin, 4) if fin else None
    raw["hm_route"] = None
    if fin >= min_verbs:
        if nper / fin >= theta_mark and _spread(pvals, fin) >= 2:
            b["HM"], raw["hm_route"] = 1, "person"
        elif nnum / fin >= theta_mark and _spread(nvals, fin) >= 2:
            # A language marking number but not person on the verb is still head-marking. Recorded
            # separately because it is the weaker evidence and a reader should be able to see which
            # route fired.
            b["HM"], raw["hm_route"] = 1, "number"

    # --- dependent-marking -------------------------------------------------------------------
    n_noun = m["noun_n"]
    raw["n_noun"] = n_noun
    raw["p_case"] = round(m["noun_case_n"] / n_noun, 4) if n_noun else None
    raw["case_vals"] = dict(m["case_vals"].most_common())
    raw["core_case"] = sorted(set(m["case_vals"]) & CORE_CASES)
    if n_noun >= min_nouns and m["noun_case_n"] / n_noun >= theta_mark \
            and (set(m["case_vals"]) & CORE_CASES) and len(m["case_vals"]) >= 2:
        b["DM"] = 1

    # --- sex-based noun classification ---------------------------------------------------------
    # Domain is NOUN. That is narrower than WALS 31A, which counts pronominal gender and therefore
    # calls English sex-based; "sex-based NOUN classification" is the feature actually asked for, and
    # the disagreement is reported by the calibration matrix rather than papered over.
    raw["gender_vals"] = dict(m["gender_vals"].most_common())
    # Max over the domain, not a pooled rate: nouns outnumber determiners and adjectives, so pooling
    # lets a treebank that marks gender only on agreeing words fall under any sensible floor.
    def rate(counter):
        best = 0.0
        for pos in GENDER_DOMAIN:
            tot = m["gdom_tot"][pos]
            if tot >= 200:
                best = max(best, counter[pos] / tot)
        return best

    # `p_any` decides whether the SYSTEM is annotated at all; the VALUE INVENTORY decides whether
    # it is sex-based. Requiring a sex-marked proportion instead would fail every system whose
    # non-sex class is the majority -- Tamil's noun genders are Neut 2 269 / Com 526 / Masc 1, a
    # rational/irrational split in which sex is plainly a criterion but marks a fifth of the nouns.
    p_any, p_sex, p_class = rate(m["gdom_any"]), rate(m["gdom_sex"]), rate(m["gdom_class"])
    raw["p_gender"] = round(p_any, 4)
    raw["p_gender_sex"], raw["p_gender_class"] = round(p_sex, 4), round(p_class, 4)
    raw["gender_by_pos"] = {pos: [m["gdom_sex"][pos], m["gdom_tot"][pos]] for pos in GENDER_DOMAIN}
    raw["noun_feats_fill"] = round(m["noun_feats_n"] / n_noun, 4) if n_noun else None
    raw["pron_gender_rate"] = (round(m["pron_gender_n"] / m["pron_n"], 4) if m["pron_n"] else None)
    informative = bool(sum(m["gdom_tot"].values()) >= min_nouns
                       and (m["gender_vals"] or m["noun_feats_n"] / max(n_noun, 1) >= 0.05))
    has_sex = bool(set(",".join(m["gender_vals"]).split(",")) & SEX_GENDERS)
    if informative and has_sex and p_any >= gender_floor and len(m["gender_vals"]) >= 2:
        b["SEX"], raw["class_system"] = 1, "sex"
    elif not informative:
        raw["class_system"] = "unknown"
    elif p_any > 0:
        b["NOSEX"], raw["class_system"] = 1, "nonsex"
    else:
        b["NOSEX"], raw["class_system"] = 1, "none"

    return b, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--out", default="assets_typ/typology_treebank.json")
    ap.add_argument("--split", default="train",
                    help="which split to profile; falls back to test then dev")
    ap.add_argument("--theta-minor", type=float, default=0.30)
    ap.add_argument("--theta-mark", type=float, default=0.50)
    ap.add_argument("--min-arcs", type=int, default=100)
    ap.add_argument("--min-verbs", type=int, default=200)
    ap.add_argument("--min-nouns", type=int, default=500)
    ap.add_argument("--gender-floor", type=float, default=0.20)
    ap.add_argument("--keep-pron-obj", action="store_true",
                    help="count pronominal objects too (default: excluded, because WALS 83A is "
                         "about full-NP order and object clitics invert it)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these corpus names")
    ap.add_argument("--sensitivity", action="store_true",
                    help="re-threshold at theta +/- 0.05 and report which bits flip")
    a = ap.parse_args()

    inv = json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))
    corpora = inv["corpora"]
    if a.only:
        want = set(a.only)
        corpora = [c for c in corpora if c["name"] in want]
    print(f"profiling {len(corpora)} corpora from {a.inventory}")

    table, raws = {}, {}
    for i, c in enumerate(corpora, 1):
        paths = c["paths"].get(a.split) or c["paths"].get("test") or c["paths"].get("dev")
        if not paths:
            print(f"  !! {c['name']}: no usable split", file=sys.stderr)
            continue
        m = measure(paths, exclude_pron_obj=not a.keep_pron_obj)
        bits, raw = bits_from(m, a.theta_minor, a.theta_mark, a.min_arcs, a.min_verbs,
                              a.min_nouns, a.gender_floor)
        key = c["lcode"] or c["lang_name"]
        table[key] = {
            "bits": [bits[f] for f in FIELDS],
            "sources": {f: "treebank" for f in FIELDS},
            "corpus": c["name"], "split": a.split,
            "class_system": raw.get("class_system", "unknown"),
            "raw": raw,
        }
        raws[key] = m
        if i % 25 == 0 or i == len(corpora):
            print(f"  {i}/{len(corpora)}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"fields": FIELDS, "theta_minor": a.theta_minor,
                        "theta_mark": a.theta_mark, "min_arcs": a.min_arcs,
                        "min_verbs": a.min_verbs, "min_nouns": a.min_nouns,
                        "gender_floor": a.gender_floor,
                        "exclude_pron_obj": not a.keep_pron_obj,
                        "source": "treebank",
                        "warning": "ORACLE for any language held out of training -- training "
                                   "languages only; test profiles come from typology_external.py"},
               "languages": table},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(table)} profiles -> {a.out}")

    hdr = "lang       " + " ".join(f"{f:>5s}" for f in FIELDS) + "   p_OV  p_SV  p_Per p_Cas class"
    print("\n" + hdr)
    for k in sorted(table):
        t, r = table[k], table[k]["raw"]
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else "  . "  # noqa: E731
        print(f"{k:10s} " + " ".join(f"{b:5d}" for b in t["bits"])
              + f"   {fmt(r.get('p_OV'))}  {fmt(r.get('p_SV'))}  "
                f"{fmt(r.get('p_person'))}  {fmt(r.get('p_case'))}  {t['class_system']}")

    # Collision rate is a GO signal, not a warning: with 8 bits over ~50 languages, heavy collision
    # is what stops the channel being a language identifier in disguise. v1 measured a 13-row
    # language embedding at -0.02 macro LAS, i.e. nothing, and binarising its 9 graded parameters
    # left 12 of 13 profiles distinct -- a language id wearing a typology hat.
    codes = ["".join(str(b) for b in t["bits"]) for t in table.values()]
    uniq = len(set(codes))
    print(f"\ndistinct profiles: {uniq}/{len(codes)} = {uniq / max(len(codes), 1):.2f}")
    if len(codes) and uniq / len(codes) > 0.6:
        print("  !! above 0.6 -- the profile is close to a language identifier; the langid control "
              "arm is what settles whether that matters")
    for code, n in collections.Counter(codes).most_common(8):
        who = [k for k in sorted(table) if "".join(str(b) for b in table[k]["bits"]) == code]
        print(f"  {code}  x{n:2d}  {' '.join(who[:12])}")

    if a.sensitivity:
        print("\nsensitivity: bits that flip at theta +/- 0.05")
        for dm in (-0.05, 0.05):
            flips = []
            for k in sorted(raws):
                alt, _ = bits_from(raws[k], a.theta_minor + dm, a.theta_mark + dm, a.min_arcs,
                                   a.min_verbs, a.min_nouns, a.gender_floor)
                base = table[k]["bits"]
                d = [FIELDS[i] for i in range(8) if alt[FIELDS[i]] != base[i]]
                if d:
                    flips.append(f"{k}:{','.join(d)}")
            print(f"  theta{dm:+.2f}  {len(flips)} languages  {' '.join(flips[:20])}")


if __name__ == "__main__":
    main()
