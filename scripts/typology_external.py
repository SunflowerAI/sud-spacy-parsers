#!/usr/bin/env python3
"""The same four typological features, from PUBLISHED DATABASES. This is the test-language path.

Grambank first for three of the four fields, because its features are natively binary and
non-exclusive -- which is exactly the two-hot semantics this encoding needs, and something WALS's
mutually-exclusive value sets have to be translated into:

    HM   GB089-GB094   S/A/P indexed by an affix or clitic on the verb      ~2 310 languages
    DM   GB070-GB073   morphological case, core and oblique, nominal and pronominal
                       non-pronominal only, to match the treebank predicate  ~2 260
    SEX  GB051         "gender/noun class system where sex is a factor"     2 206
                       GB052/053/054/192 give the non-sex-based systems for the second bit

and WALS for the two word-order fields, where its coverage is the deeper of the two:

    OV/VO  WALS 83A (1 518)   fallback Grambank GB133 (verb-final transitive,  2 336)
    SV/VS  WALS 82A (1 496)   fallback Grambank GB131 (verb-initial transitive, 2 348)
    HM/DM  WALS 23A / 25A as a fallback where Grambank has no row

⚠ **EVERY BIT CARRIES ITS OWN `source`.** That is not bookkeeping: `check_generic_inputs_v2.py`
refuses to let a test language into the corpus if any of its bits says `treebank`, and that refusal
is the whole difference between this experiment and v1's ungated +12.74. A bit with no database
behind it is either left UNKNOWN (`00`) or supplied by hand through `--manual`, which requires a
citation string and is tagged `literature`.

⚠ **iso3 -> WALS code is many-to-one.** WALS carries dialect-level entries under separate codes that
share an ISO 639-3 code, so a naive join silently picks whichever row came last. We resolve to the
entry with the most coverage across the four parameters and record `wals_code` per language, so the
join is auditable rather than merely deterministic.

`00` on the HM/DM field genuinely means "no database row", but a language that is neither
head-marking nor dependent-marking -- an isolating language -- also lands on `00`. That overload is
deliberate and is what the `g2_typ12` arm measures the cost of; `hm_known`/`dm_known` are recorded
here so that arm can be built without re-deriving anything.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys
import urllib.request

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]

GB = "https://raw.githubusercontent.com/grambank/grambank/master/cldf"
WALS = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf"
GLOTTO = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf"

#: UD names macrolanguages where Glottolog and Grambank name an individual language. Without these
#: Arabic, Persian, Estonian, Odia, Uzbek and Yiddish reach NEITHER database -- `ara` and `fas` are
#: not languoids, so the glottocode bridge cannot resolve them either. The member chosen is the one
#: the treebank actually represents.
ISO_ALIAS = {
    "ara": "arb",   # Modern Standard Arabic
    "fas": "pes",   # Western Farsi
    "est": "ekk",   # Standard Estonian
    "ori": "ory",   # Odia
    "uzb": "uzn",   # Northern Uzbek
    "yid": "ydd",   # Eastern Yiddish
    "msa": "zsm",   # Standard Malay
    "zho": "cmn",   # Mandarin
    "nor": "nob",   # Bokmal
    "swa": "swh",   # Coastal Swahili
    "aze": "azj",   # North Azerbaijani
    "grn": "gug",   # Paraguayan Guarani
    "kur": "kmr",   # Northern Kurdish
    "que": "quy",   # Ayacucho Quechua
    "mlg": "plt",   # Plateau Malagasy
    "srp": "hbs", "hrv": "hbs", "bos": "hbs",   # Glottolog treats these as Serbian-Croatian
}

#: Grambank feature groups. "any of these is 1" is the test -- these are independent yes/no
#: questions about a language, not alternatives.
GB_HEAD = ["GB089", "GB090", "GB091", "GB092", "GB093", "GB094"]
#: NON-PRONOMINAL case only. Grambank splits these out (GB071/GB073 are the pronominal ones, and
#: GB408/GB409 count adpositional flagging too), and the treebank predicate measures Case on
#: NOUN/PROPN -- so including them here would give the DM bit one meaning in training and a broader
#: one at test time. English is the case in point: no nominal case, but `him`/`them` carry it.
GB_DEP = ["GB070", "GB072"]
GB_SEX = "GB051"
GB_NONSEX = ["GB052", "GB053", "GB054", "GB192"]
GB_VINITIAL, GB_VFINAL = "GB131", "GB133"
GB_WANT = set(GB_HEAD + GB_DEP + GB_NONSEX + [GB_SEX, GB_VINITIAL, GB_VFINAL])

WALS_WANT = {"82A", "83A", "23A", "25A", "31A"}


def fetch(url, dest: pathlib.Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {url}")
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    return dest


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_glottolog(cache: pathlib.Path):
    """iso3 -> glottocode, and a lowercased name index for the last-resort join.

    ⚠ Grambank's own `ISO639P3code` column is EMPTY on every row -- it is keyed by Glottocode
    alone -- so an iso3 join against it silently matches nothing at all. That is not a coverage
    problem that shows up as a few gaps; it is the entire database returning zero.
    """
    rows = read_csv(fetch(f"{GLOTTO}/languages.csv", cache / "glottolog_languages.csv"))
    iso2glot = {r["ISO639P3code"]: r["ID"] for r in rows if r.get("ISO639P3code")}
    by_name = {}
    for r in rows:
        nm = (r.get("Name") or "").strip().lower()
        if nm and nm not in by_name:
            by_name[nm] = r["ID"]
    return iso2glot, by_name


def load_grambank(cache: pathlib.Path):
    """glottocode -> {parameter: "0"|"1"}."""
    langs = read_csv(fetch(f"{GB}/languages.csv", cache / "grambank_languages.csv"))
    vals = read_csv(fetch(f"{GB}/values.csv", cache / "grambank_values.csv"))
    out = collections.defaultdict(dict)
    for v in vals:
        p = v.get("Parameter_ID")
        if p not in GB_WANT:
            continue
        val = (v.get("Value") or "").strip()
        if val in ("", "?"):
            continue
        out[v.get("Language_ID", "")][p] = val
    return out, {r["ID"]: r for r in langs}


def load_wals(cache: pathlib.Path):
    """glottocode -> (wals_code, {parameter: code name}). Resolves the many-to-one join by coverage."""
    langs = read_csv(fetch(f"{WALS}/languages.csv", cache / "wals_languages.csv"))
    codes = read_csv(fetch(f"{WALS}/codes.csv", cache / "wals_codes.csv"))
    vals = read_csv(fetch(f"{WALS}/values.csv", cache / "wals_values.csv"))
    code_name = {c["ID"]: c["Name"] for c in codes if c["Parameter_ID"] in WALS_WANT}

    by_wals = collections.defaultdict(dict)
    for v in vals:
        if v.get("Parameter_ID") not in WALS_WANT:
            continue
        nm = code_name.get(v.get("Code_ID", ""))
        if nm:
            by_wals[v["Language_ID"]][v["Parameter_ID"]] = nm

    # Key by GLOTTOCODE, not iso3: WALS carries dialect-level entries that share an ISO code
    # (twenty-one separate "Arabic (...)" rows), and glottocodes separate them properly.
    per_glot = collections.defaultdict(list)
    for r in langs:
        g = (r.get("Glottocode") or "").strip()
        if g:
            per_glot[g].append(r["ID"])
    out = {}
    for g, wcodes in per_glot.items():
        # Most parameters covered wins; the WALS id breaks ties so the choice is reproducible.
        best = max(wcodes, key=lambda w: (len(by_wals.get(w, {})), w))
        out[g] = (best, by_wals.get(best, {}))
    return out


def resolve(iso, name, iso2glot, by_name):
    """`(glottocode, how)`. iso3, then the macrolanguage alias, then Glottolog's own name index.

    `how` is recorded per language so a name-matched join -- the loosest of the three -- is visible
    rather than indistinguishable from an exact code match.
    """
    if not iso:
        pass
    elif iso in iso2glot:
        return iso2glot[iso], "iso3"
    elif iso in ISO_ALIAS and ISO_ALIAS[iso] in iso2glot:
        return iso2glot[ISO_ALIAS[iso]], f"alias:{ISO_ALIAS[iso]}"
    g = by_name.get((name or "").strip().lower())
    return (g, "name") if g else (None, "unresolved")


def bits_for(glot, gram, wals):
    """Eight bits plus a per-bit source map. Absent evidence leaves a field at `00`."""
    b = dict.fromkeys(FIELDS, 0)
    src = {}
    g = gram.get(glot, {})
    wcode, w = wals.get(glot, (None, {}))
    extra = {"wals_code": wcode, "hm_known": False, "dm_known": False,
             "class_system": "unknown"}

    def one(v):
        return v == "1"

    # --- order: WALS first, Grambank's verb-position pair as the fallback --------------------
    if w.get("83A") in ("OV", "VO", "No dominant order"):
        b["OV"] = int(w["83A"] in ("OV", "No dominant order"))
        b["VO"] = int(w["83A"] in ("VO", "No dominant order"))
        src["OV"] = src["VO"] = "wals:83A"
    elif GB_VFINAL in g or GB_VINITIAL in g:
        # Verb-initial implies the object follows; verb-final implies it precedes; a language coded
        # neither is verb-medial, i.e. VO. Both coded 1 is a genuinely mixed language.
        vf, vi = one(g.get(GB_VFINAL, "0")), one(g.get(GB_VINITIAL, "0"))
        b["OV"], b["VO"] = int(vf), int(vi or not (vf or vi))
        src["OV"] = src["VO"] = f"grambank:{GB_VFINAL}/{GB_VINITIAL}"

    if w.get("82A") in ("SV", "VS", "No dominant order"):
        b["SV"] = int(w["82A"] in ("SV", "No dominant order"))
        b["VS"] = int(w["82A"] in ("VS", "No dominant order"))
        src["SV"] = src["VS"] = "wals:82A"
    elif GB_VFINAL in g or GB_VINITIAL in g:
        vf, vi = one(g.get(GB_VFINAL, "0")), one(g.get(GB_VINITIAL, "0"))
        b["SV"], b["VS"] = int(vf or not (vf or vi)), int(vi)
        src["SV"] = src["VS"] = f"grambank:{GB_VFINAL}/{GB_VINITIAL}"

    # --- locus of marking: Grambank first, WALS 23A then 25A as the fallback -----------------
    head_rows = [g[p] for p in GB_HEAD if p in g]
    dep_rows = [g[p] for p in GB_DEP if p in g]
    if head_rows or dep_rows:
        if head_rows:
            b["HM"] = int(any(one(v) for v in head_rows))
            src["HM"] = "grambank:GB089-094"
            extra["hm_known"] = True
        if dep_rows:
            b["DM"] = int(any(one(v) for v in dep_rows))
            src["DM"] = "grambank:GB070,GB072"
            extra["dm_known"] = True
    else:
        locus = w.get("23A") or w.get("25A")
        key = "wals:23A" if w.get("23A") else "wals:25A"
        if locus in ("Head marking", "Dependent marking", "Double marking", "No marking"):
            b["HM"] = int(locus in ("Head marking", "Double marking"))
            b["DM"] = int(locus in ("Dependent marking", "Double marking"))
            src["HM"] = src["DM"] = key
            extra["hm_known"] = extra["dm_known"] = True
        # "Other" is deliberately not mapped: it is a residue category, not a value this encoding
        # has a slot for, and guessing would be worse than leaving the field unknown.

    # --- sex-based noun classification -------------------------------------------------------
    if GB_SEX in g:
        if one(g[GB_SEX]):
            b["SEX"], extra["class_system"] = 1, "sex"
        else:
            b["NOSEX"] = 1
            extra["class_system"] = ("nonsex" if any(one(g[p]) for p in GB_NONSEX if p in g)
                                     else "none")
        src["SEX"] = src["NOSEX"] = f"grambank:{GB_SEX}"
    elif w.get("31A") in ("Sex-based", "Non-sex-based", "No gender"):
        b["SEX"] = int(w["31A"] == "Sex-based")
        b["NOSEX"] = int(w["31A"] in ("Non-sex-based", "No gender"))
        src["SEX"] = src["NOSEX"] = "wals:31A"
        extra["class_system"] = {"Sex-based": "sex", "Non-sex-based": "nonsex",
                                 "No gender": "none"}[w["31A"]]

    return b, src, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--cache", default="assets_typ/cldf")
    ap.add_argument("--out", default="assets_typ/typology_external.json")
    ap.add_argument("--manual", default="assets_typ/typology_manual.json",
                    help="hand-supplied bits for languages neither database covers. Each entry "
                         "needs a `citation`; entries are tagged source `literature`.")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    cache = pathlib.Path(a.cache)
    iso2glot, by_name = load_glottolog(cache)
    gram, gram_langs = load_grambank(cache)
    wals = load_wals(cache)
    print(f"glottolog: {len(iso2glot)} iso3 -> glottocode")
    print(f"grambank:  {len(gram)} languoids with any wanted feature")
    print(f"wals:      {len(wals)} glottocodes")

    manual = {}
    mp = pathlib.Path(a.manual)
    if mp.exists():
        manual = json.loads(mp.read_text(encoding="utf-8"))
        print(f"manual:   {len(manual)} hand-supplied entries")

    inv = json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))
    corpora = inv["corpora"]
    if a.only:
        want = set(a.only)
        corpora = [c for c in corpora if c["name"] in want]

    table, missing = {}, []
    for c in corpora:
        key = c["lcode"] or c["lang_name"]
        iso = c["iso3"]
        glot, how = resolve(iso, c["lang_name"], iso2glot, by_name)
        b, src, extra = (bits_for(glot, gram, wals) if glot
                         else (dict.fromkeys(FIELDS, 0), {}, {}))
        man = manual.get(key) or manual.get(iso) or {}
        for f, v in (man.get("bits") or {}).items():
            if f not in FIELDS:
                sys.exit(f"--manual: {key} names an unknown field {f!r}")
            if not man.get("citation"):
                sys.exit(f"--manual: {key} supplies bits without a `citation`")
            b[f] = int(v)
            src[f] = f"literature:{man['citation']}"
        got = [f for f in FIELDS if f in src]
        if len(got) < len(FIELDS):
            missing.append((key, iso, [f for f in FIELDS if f not in src]))
        table[key] = {
            "bits": [b[f] for f in FIELDS],
            "sources": {f: src.get(f, "none") for f in FIELDS},
            "iso3": iso, "glottocode": glot, "join": how, "lang_name": c["lang_name"],
            "macroarea": (gram_langs.get(glot) or {}).get("Macroarea", ""),
            **extra,
        }

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"fields": FIELDS, "source": "external",
                        "note": "Grambank + WALS + hand-supplied literature values. No bit here "
                                "derives from a treebank; that is what makes the zero-shot claim "
                                "clean."},
               "languages": table},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(table)} profiles -> {a.out}")

    # Coverage is the thing that decides which languages can be TESTED at all, so it is reported
    # per field rather than as a single number.
    cov = collections.Counter()
    for t in table.values():
        for f in FIELDS:
            cov[f] += t["sources"][f] != "none"
    print("\nfield coverage over the inventory:")
    for f in FIELDS:
        print(f"  {f:6s} {cov[f]:4d}/{len(table)}")
    joins = collections.Counter(t["join"] for t in table.values())
    print("\njoin method:", dict(joins))
    print(f"\n{len(missing)} languages missing at least one field")
    for key, iso, fs in missing[:25]:
        print(f"  {key:8s} {iso:5s} missing {','.join(fs)}")
    if len(missing) > 25:
        print(f"  ... and {len(missing) - 25} more")


if __name__ == "__main__":
    main()
