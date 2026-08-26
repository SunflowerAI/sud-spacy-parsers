#!/usr/bin/env python3
"""Sanskrit -> English anchors from Apte, for fitting the sa rotation into the shared space.

Apte is the only Sanskrit-English resource we hold, and it is a DEFINITION dictionary, not a
translation table: an entry glosses a headword with a run of English prose. So this emits, per
headword, the BAG of English content words in its glosses. The alignment step turns that bag into a
target vector by averaging the English vectors of its members -- far more robust than forcing a 1:1
translation out of prose that does not contain one.

Two shape mismatches have to be fixed or the anchor set is almost empty:

  * Apte cites headwords in the NOMINATIVE SINGULAR (`agastyaH`, `anuyAtraM`); DCS lemmas are STEMS
    (`agastya`, `anuyAtra`). In SLP1 those citation endings are the visarga `H` and anusvara `M`,
    both inflection. Strip them and keep both forms, as `build_apte_lexicon.py` does.
  * Apte is SLP1, DCS and our sa arm are IAST. Transliterate.

The dictionary is used only to FIT a rotation; no Apte content is redistributed in the released
asset, which holds vectors alone. (The CDSL digitisation has its own terms on top of Apte 1890's
public domain by age -- see `build_apte_lexicon.py`.)
"""
import argparse, collections, json, re
from indic_transliteration.sanscript import transliterate

ENTRY = re.compile(r"<L>.*?(?:<LEND>|$)", re.S)
K1 = re.compile(r"<k1>([^<]*)")
# everything that is not the English gloss
SANSKRIT = re.compile(r"\{#.*?#\}", re.S)
CITATION = re.compile(r"<ls>.*?</ls>", re.S)
ABBREV = re.compile(r"<ab>.*?</ab>", re.S)
BRACKET = re.compile(r"\[.*?\]", re.S)
SENSE = re.compile(r"\{@.*?@\}|\{%|%\}|\{\d+\}", re.S)
TAG = re.compile(r"<[^>]*>", re.S)

STOP = set("""a an the of or and to in on for with by from as is are was were be been being at into
this that these those it its his her their our your my he she they we you i not no nor but if then
than so such other another same very much more most less least any all some each every both few
many one two three first second name see also cf esp etc ep ved comp used which who whom whose what
when where while after before over under again further here there once during about against between
through above below up down out off only own too s t can will just don now like say said thing
things person persons man men woman women part parts kind sort way ways form forms word words
""".split())

def gloss_words(body):
    for rx in (SANSKRIT, CITATION, ABBREV, BRACKET, SENSE):
        body = rx.sub(" ", body)
    body = TAG.sub(" ", body)
    body = body.replace("¦", " ").replace("˚", " ").replace("−", " ").replace("--", " ")
    out = []
    for w in re.findall(r"[A-Za-z]+", body):
        w = w.lower()
        if len(w) >= 3 and w not in STOP:
            out.append(w)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["assets_apte/ap90.txt", "assets_apte/ap.txt"])
    ap.add_argument("--out", default="assets_vec/dict/sa-en.json")
    ap.add_argument("--max-gloss", type=int, default=25,
                    help="keep at most this many gloss words per headword (long entries are "
                         "encyclopaedic and drift off the headword's own sense)")
    a = ap.parse_args()

    bag = collections.defaultdict(collections.Counter)
    n = 0
    for src in a.sources:
        txt = open(src, encoding="utf-8", errors="replace").read()
        for m in ENTRY.finditer(txt):
            e = m.group(0)
            k = K1.search(e)
            if not k or not k.group(1):
                continue
            slp = k.group(1).strip()
            body = e.split("¦", 1)[1] if "¦" in e else e
            ws = gloss_words(body)[: a.max_gloss]
            if not ws:
                continue
            n += 1
            forms = {slp}
            if len(slp) > 2 and slp[-1] in "HM":
                forms.add(slp[:-1])
            for f in forms:
                iast = transliterate(f, "slp1", "iast")
                bag[iast].update(ws)
    out = {k: dict(v.most_common(a.max_gloss)) for k, v in bag.items()}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{a.out}: {n} entries -> {len(out)} IAST headwords, "
          f"{sum(len(v) for v in out.values())} headword-gloss links")

main()
