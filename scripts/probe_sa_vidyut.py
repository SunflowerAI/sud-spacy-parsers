import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_common import key_variants, read_keys, write
from vidyut.kosha import Kosha

VIB = {"praTamA":"Nom","dvitIyA":"Acc","tftIyA":"Ins","caturTI":"Dat",
       "paYcamI":"Abl","zazWI":"Gen","saptamI":"Loc","samboDanam":"Voc"}
LIN = {"puM":"Masc","strI":"Fem","napuMsaka":"Neut"}
VAC = {"eka":"Sing","dvi":"Dual","bahu":"Plur"}
# ⚠ Purusha.praTama is the THIRD person; Vibhakti.praTamA is the nominative. Same name, different
# category — mapping them from one dict would silently produce Case=Nom on every finite verb.
PUR = {"praTama":"3","maDyama":"2","uttama":"1"}

kosha = Kosha(sys.argv[1] if len(sys.argv) > 1 else "vidyut-data/kosha")
keys = read_keys(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SA_KEYS","sa_test_tokens.tsv")))
rows = []
for k in keys:
    lem, case, num, gen, per = set(), set(), set(), set(), set()
    found = False
    for v in key_variants(k):
        for e in kosha.get(v):
            found = True
            if e.lemma:
                lem.add(str(e.lemma))
            if getattr(e, "vibhakti", None) is not None:
                case.add(VIB.get(str(e.vibhakti), "?"))
            if getattr(e, "linga", None) is not None:
                gen.add(LIN.get(str(e.linga), "?"))
            if getattr(e, "vacana", None) is not None:
                num.add(VAC.get(str(e.vacana), "?"))
            if getattr(e, "purusha", None) is not None:
                per.add(PUR.get(str(e.purusha), "?"))
    rows.append([k, int(found), ",".join(sorted(lem)), ",".join(sorted(case)),
                 ",".join(sorted(num)), ",".join(sorted(gen)), ",".join(sorted(per))])
write(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SA_OUT_V","res_vidyut.tsv")), rows)
print("vidyut: probed", len(keys), "keys;", sum(r[1] for r in rows), "recognised")
