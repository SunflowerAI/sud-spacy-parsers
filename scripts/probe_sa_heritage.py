import sys, os, re, logging
logging.disable(logging.INFO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_common import key_variants, read_keys, write
from indic_transliteration import sanscript
from sanskrit_parser.util.lexical_lookup_factory import LexicalLookupFactory
from sanskrit_parser.util.inriatagmapper import inriatags

# inriatags pairs an Inria code with the DEVANAGARI name that get_tags actually returns, SLP1-encoded.
# Build the map by transliterating rather than hand-typing 17 SLP1 strings — one typo would read as
# "this analyser never predicts the dative" and be invisible.
CODE = {"na-nom":("Case","Nom"), "na-voc":("Case","Voc"), "na-acc":("Case","Acc"),
        "na-ins":("Case","Ins"), "na-dat":("Case","Dat"), "na-abl":("Case","Abl"),
        "na-gen":("Case","Gen"), "na-loc":("Case","Loc"),
        "sg":("Number","Sing"), "du":("Number","Dual"), "pl":("Number","Plur"),
        "mas":("Gender","Masc"), "fem":("Gender","Fem"), "neu":("Gender","Neut"),
        "fst":("Person","1"), "snd":("Person","2"), "trd":("Person","3")}
TAG = {}
for code, name in inriatags:
    if code in CODE:
        TAG[sanscript.transliterate(name, sanscript.DEVANAGARI, sanscript.SLP1)] = CODE[code]
        TAG[name] = CODE[code]          # a few entries are already romanised
assert len([v for v in TAG.values() if v[0] == "Case"]) >= 8, TAG

inria = LexicalLookupFactory.create("inria")
keys = read_keys(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SA_KEYS","sa_test_tokens.tsv")))
rows = []
for k in keys:
    lem, vals = set(), {"Case": set(), "Number": set(), "Gender": set(), "Person": set()}
    found = False
    for v in key_variants(k):
        for stem, tags in (inria.get_tags(v) or []):
            found = True
            lem.add(re.sub(r"#\d+$", "", str(stem)))
            for t in tags:
                hit = TAG.get(str(t))
                if hit:
                    vals[hit[0]].add(hit[1])
    rows.append([k, int(found), ",".join(sorted(lem))] +
                [",".join(sorted(vals[f])) for f in ("Case", "Number", "Gender", "Person")])
write(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SA_OUT_H","res_heritage.tsv")), rows)
print("heritage: probed", len(keys), "keys;", sum(r[1] for r in rows), "recognised")
