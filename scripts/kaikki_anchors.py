#!/usr/bin/env python3
"""Wiktionary (kaikki.org wiktextract) glosses -> English anchor bags, for la and te.

Same contract as `apte_anchors.py`: a headword maps to the BAG of English content words in its
glosses, and the alignment step averages their English vectors to get one target. Wiktionary glosses
are already English prose ("a horse", "to carry, bear"), so the only work is stripping the
parenthesised usage labels and the grammatical boilerplate that would otherwise dominate the bag --
"genitive", "singular", "of" -- since inflected-form entries ("dative singular of equus") are the
majority of the Latin extract and their gloss is pure metalanguage.

Inflected-form entries are not discarded: the LEMMA named in the gloss is a better anchor for them
than nothing, so `equo` inherits `equus`'s bag through --follow-forms.

Streams from a URL or a local file; the extract is 1.2 GB for Latin and is not kept.
"""
import argparse, collections, json, re, sys, urllib.request

STOP = set("""a an the of or and to in on for with by from as is are was were be been being at into
this that these those it its his her their our your my he she they we you i not no nor but if then
than so such other another same very much more most less least any all some each every both few
many one two three first second name see also cf esp etc used which who whom whose what when where
while after before over under again further here there once during about against between through
above below up down out off only own too can will just now like say said thing things person persons
form forms word words plural singular genitive dative ablative accusative nominative vocative
locative masculine feminine neuter gender number case tense mood voice active passive perfect
imperfect future present past participle infinitive imperative subjunctive indicative supine gerund
gerundive comparative superlative diminutive alternative obsolete archaic dated rare nonstandard
misspelling spelling variant inflection inflected conjugation declension stem root suffix prefix
""".split())
LABEL = re.compile(r"\([^)]*\)")
FORM_OF = re.compile(r"\b(?:of)\s+([A-Za-zĀ-ſ஀-௿ఀ-౿'\-]+)\s*$")

def gloss_words(g):
    g = LABEL.sub(" ", g)
    return [w.lower() for w in re.findall(r"[A-Za-z]+", g)
            if len(w) >= 3 and w.lower() not in STOP]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--file")
    ap.add_argument("--lang-code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gloss", type=int, default=25)
    ap.add_argument("--follow-forms", action="store_true", default=True)
    ap.add_argument("--alias-forms", nargs="*", default=[],
                    help="also key the entry under its `forms` with these tags -- Chinese entries "
                         "carry the traditional/simplified counterpart there, and lzh is written "
                         "traditional while much of the source vocabulary is not")
    a = ap.parse_args()

    src = open(a.file, "rb") if a.file else urllib.request.urlopen(a.url)
    bag = collections.defaultdict(collections.Counter)
    formof = {}                     # inflected form -> lemma named in its gloss
    n = kept = 0
    for raw in src:
        n += 1
        if n % 200000 == 0:
            print(f"  {n} lines, {len(bag)} headwords", file=sys.stderr, flush=True)
        try:
            e = json.loads(raw)
        except Exception:
            continue
        if e.get("lang_code") != a.lang_code:
            continue
        w = e.get("word")
        if not w:
            continue
        names = [w]
        if a.alias_forms:
            for fo in e.get("forms", []) or []:
                tags = set(fo.get("tags", []) or [])
                if tags & set(a.alias_forms) and fo.get("form"):
                    names.append(fo["form"])
        for s in e.get("senses", []):
            for g in s.get("glosses", []) or []:
                ws = gloss_words(g)
                if ws:
                    for nm in names:
                        bag[nm].update(ws)
                    kept += 1
                else:
                    m = FORM_OF.search(g.strip().rstrip("."))
                    if m and m.group(1) != w:
                        formof[w] = m.group(1)
    if a.follow_forms:
        inherited = 0
        for f, lem in formof.items():
            if f not in bag and lem in bag:
                bag[f] = bag[lem]; inherited += 1
        print(f"  inflected forms inheriting a lemma bag: {inherited}", file=sys.stderr)
    out = {k: dict(v.most_common(a.max_gloss)) for k, v in bag.items()}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{a.out}: {n} lines, {kept} glosses -> {len(out)} headwords, "
          f"{sum(len(v) for v in out.values())} headword-gloss links")

main()
