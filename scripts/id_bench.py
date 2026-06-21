#!/usr/bin/env python3
"""Tune the Indonesian comp/mod prompt with curated same-preposition contrastive few-shot.
Teaches that di/pada/dengan are COMPLEMENTS only with selecting verbs (terletak di, bergantung
pada, berhubungan dengan), but MODIFIERS as free locatives, temporals, instrumentals, or manner
(bekerja di kantor, pada tahun 1945, menulis dengan pena). Test split matches lang_bench (seed 0).
"""
import importlib.util, json, random
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)
_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)
_sb = importlib.util.spec_from_file_location("b", "scripts/lang_bench.py")
b = importlib.util.module_from_spec(_sb); _sb.loader.exec_module(b)

def S(vp, pp, v, g): return {"verb_phrase": vp, "prep_phrase": pp, "verb": v, "gold": g}

CONTRAST = [
    # COMPLEMENT: verb selects the phrase (locative/source/predicative argument)
    S("Kota itu terletak di Jawa", "di Jawa", "terletak", "complement"),
    S("Dia berasal dari Bandung", "dari Bandung", "berasal", "complement"),
    S("Komite terdiri dari lima orang", "dari lima orang", "terdiri", "complement"),
    S("Hasilnya bergantung pada cuaca", "pada cuaca", "bergantung", "complement"),
    S("Hal itu berhubungan dengan masalah ekonomi", "dengan masalah ekonomi", "berhubungan", "complement"),
    S("Dia dikenal sebagai pahlawan", "sebagai pahlawan", "dikenal", "complement"),
    S("Meja itu terbuat dari kayu", "dari kayu", "terbuat", "complement"),
    S("Keputusan itu didasarkan pada hukum", "pada hukum", "didasarkan", "complement"),
    S("Program ini berfokus pada pendidikan", "pada pendidikan", "berfokus", "complement"),
    S("Mereka menuju ke pasar", "ke pasar", "menuju", "complement"),
    # MODIFIER: free locative / temporal / instrumental / manner / reason
    S("Dia lahir pada tahun 1945", "pada tahun 1945", "lahir", "modifier"),
    S("Gedung itu dibangun pada abad ke-19", "pada abad ke-19", "dibangun", "modifier"),
    S("Dia bekerja di kantor", "di kantor", "bekerja", "modifier"),
    S("Saya membaca di taman", "di taman", "membaca", "modifier"),
    S("Dia menulis dengan pena", "dengan pena", "menulis", "modifier"),
    S("Dia berbicara dengan cepat", "dengan cepat", "berbicara", "modifier"),
    S("Mereka tinggal selama tiga tahun", "selama tiga tahun", "tinggal", "modifier"),
    S("Acara dibatalkan karena hujan", "karena hujan", "dibatalkan", "modifier"),
    S("Dia datang setelah rapat", "setelah rapat", "datang", "modifier"),
    S("Bangunan itu ada sejak 1990", "sejak 1990", "ada", "modifier"),
]


def test_split(seed=0, ns=8, per_class=120):
    gold = [json.loads(l) for l in open("gold_id.jsonl") if l.strip()]
    rng = random.Random(seed)
    comp = [x for x in gold if x["gold"] == "complement"]; rng.shuffle(comp)
    mod = [x for x in gold if x["gold"] == "modifier"]; rng.shuffle(mod)
    gold_shots = comp[:ns] + mod[:ns]; rng.shuffle(gold_shots)
    ct, mt = comp[ns:], mod[ns:]
    n = min(per_class, len(ct), len(mt))
    test = ct[:n] + mt[:n]; rng.shuffle(test)
    return gold_shots, test


def run(name, prefix, test):
    preds = [d.query(prefix + e.suffix(c)) for c in test]
    preds = [p if p in ("complement", "modifier") else "?" for p in preds]
    acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
    rc = {c: sum(preds[i] == c for i, t in enumerate(test) if t["gold"] == c)
          / max(1, sum(t["gold"] == c for t in test)) for c in ("complement", "modifier")}
    print(f"  {name:24} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
          f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}")


def main():
    gold_shots, test = test_split()
    print(f"id test {len(test)} ({len(test)//2}/class); curated shots {len(CONTRAST)}\n")
    variants = {
        "native_gold8 (baseline)": b.NATIVE_DEF["id"] + "Examples:\n\n" + b.shots_block(gold_shots),
        "native_contrast":         b.NATIVE_DEF["id"] + "Examples:\n\n" + b.shots_block(CONTRAST),
        "native_contrast+gold":    b.NATIVE_DEF["id"] + "Examples:\n\n" + b.shots_block(CONTRAST + gold_shots),
    }
    for name, pre in variants.items():
        run(name, pre, test)


if __name__ == "__main__":
    main()
