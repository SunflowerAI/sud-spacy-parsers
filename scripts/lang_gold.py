#!/usr/bin/env python3
"""Build high-confidence comp/mod gold from `udep` adposition-of-verb cases, per language.

Scope cleaning: only nominal adpositional phrases (no VERB inside the adposition's subtree)
of <= 8 tokens. Within those:
  MODIFIER  = reason/temporal adpositions (because/after/during/since...) or adposition +
              temporal-noun/year object.
  COMPLEMENT = verb lexically selects the adposition (curated (verb, adp) frames).
Cases matching neither are left out (treated as still-ambiguous). Resources are best-effort
and meant to be eyeballed; writes gold_<lang>.jsonl.
"""
import argparse, importlib.util, json, re
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)

FILES = {
    "zh": ["assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "ko": ["assets_ko/SUD_Korean-GSD/ko_gsd-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "id": ["assets_id/SUD_Indonesian-GSD/id_gsd-sud-%s.conllu" % s for s in ("train", "dev", "test")],
}

# adpositions that are inherently reason/temporal adjuncts -> MODIFIER
MOD_ADP = {
    "zh": {"因", "由于", "为了", "根据", "按照", "经过", "自从"},
    "ko": {"때문+에", "후", "이후", "전", "동안", "때", "중", "중+에", "만+에", "무렵", "사이"},
    "id": {"selama", "sejak", "setelah", "sebelum", "hingga", "sampai", "karena", "menjelang"},
}
# adpositions that take a temporal/year object -> MODIFIER (if object is temporal)
TEMP_OBJ_ADP = {"zh": {"于", "在", "自"}, "ko": set(), "id": {"pada", "di", "pada"}}
TEMP_NOUN = {
    "zh": {"年", "月", "日", "时", "时候", "期间", "初", "末", "世纪", "年代", "凌晨",
           "早上", "上午", "中午", "下午", "晚上", "当天", "当时", "前", "后"},
    "ko": set(),
    "id": {"tahun", "bulan", "hari", "minggu", "pagi", "siang", "sore", "malam", "abad",
           "dekade", "masa", "waktu", "saat", "zaman", "era", "pukul",
           "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
           "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
           "september", "oktober", "november", "desember"},
}
# (verb-root, adposition) frames where the verb lexically selects the adposition -> COMPLEMENT
COMP_FRAMES = {
    "zh": {("成立", "于"), ("位", "于"), ("位于", "于"), ("生", "于"), ("出生", "于"),
           ("诞生", "于"), ("分布", "于"), ("建", "于"), ("建立", "于"), ("创立", "于"),
           ("创建", "于"), ("源", "于"), ("起源", "于"), ("始", "于"), ("坐落", "于"),
           ("处", "于"), ("属", "于"), ("致力", "于"), ("适用", "于"), ("毕业", "于"),
           ("来", "自"), ("源", "自"), ("来自", "自"), ("发生", "在"), ("出现", "在"),
           ("合作", "与"), ("有关", "与")},
    "ko": {("의하", "에"), ("대하", "에"), ("관하", "에"), ("이르", "에"), ("도착", "에"),
           ("기인", "에"), ("해당", "에"), ("속하", "에"), ("참여", "에"), ("가입", "에"),
           ("응하", "에"), ("대응", "에"), ("반대", "에"), ("찬성", "에"), ("의존", "에"),
           ("달하", "에"), ("이르", "까지"), ("변하", "로"), ("바뀌", "로"), ("들어가", "로")},
    "id": {("letak", "di"), ("ada", "di"), ("asal", "dari"), ("diri", "dari"),
           ("diri", "atas"), ("kenal", "sebagai"), ("kenal", "dengan"), ("temu", "dengan"),
           ("gantung", "pada"), ("gantung", "dari"), ("buat", "dari"), ("hubung", "dengan"),
           ("kait", "dengan"), ("fokus", "pada"), ("dasar", "pada"), ("acu", "pada"),
           ("acu", "kepada"), ("bicara", "tentang"), ("cerita", "tentang"), ("tuju", "ke"),
           ("ubah", "menjadi"), ("jadi", "sebagai")},
}


def root(lemma, lang):
    return lemma.split("+")[0] if lang == "ko" else lemma


def is_year(form):
    return bool(re.fullmatch(r"\d{4}", form)) and 1000 <= int(form) <= 2100


def prep_object(toks, prep_id):
    kids = [t for t in toks if t["head"] == prep_id]
    for t in kids:
        if t["deprel"].startswith("comp:obj"):
            return t
    for t in kids:
        if t["upos"] in ("NOUN", "PROPN", "NUM", "PRON"):
            return t
    return None


def obj_is_temporal(toks, prep_id, lang):
    obj = prep_object(toks, prep_id)
    return bool(obj) and (root(obj["lemma"], lang).lower() in TEMP_NOUN[lang] or is_year(obj["form"]))


def classify(toks, prep, head, lang):
    vroot, adp = root(head["lemma"], lang), prep["lemma"]
    if (vroot, adp) in COMP_FRAMES[lang]:
        # zh: 成立于1997 / 发生在2011年 are temporal-WHEN, not locative complements
        if lang == "zh" and obj_is_temporal(toks, prep["id"], lang):
            return "modifier"
        return "complement"
    if adp in MOD_ADP[lang]:
        return "modifier"
    if adp in TEMP_OBJ_ADP.get(lang, set()) and obj_is_temporal(toks, prep["id"], lang):
        return "modifier"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=FILES)
    args = ap.parse_args()
    lang = args.lang
    items, dropped = [], 0
    for f in FILES[lang]:
        for sid, toks in d.parse_conllu(f):
            by = {t["id"]: t for t in toks}
            for t in toks:
                if t["upos"] != "ADP" or t["deprel"] != "udep" or t["head"] == 0:
                    continue
                head = by.get(t["head"])
                if not head or head["upos"] != "VERB":
                    continue
                sub = d.descendants(toks, t["id"])
                if any(by[i]["upos"] == "VERB" for i in sub if i != t["id"]):
                    continue  # scope: drop clausal phrases
                if len(sub) > 8:
                    continue
                label = classify(toks, t, head, lang)
                if label is None:
                    dropped += 1
                    continue
                items.append({"verb": head["form"], "adp": t["lemma"],
                              "prep_phrase": d.render(toks, sub),
                              "verb_phrase": d.render(toks, d.descendants(toks, head["id"])),
                              "gold": label})
    print(f"=== {lang}: confident gold {dict(Counter(i['gold'] for i in items))} "
          f"(dropped {dropped} unclear nominal udep)")
    for cls in ("complement", "modifier"):
        print(f"--- {cls} samples ---")
        for i in [x for x in items if x["gold"] == cls][:10]:
            print(f"   [{i['verb']} · {i['adp']}]  {i['prep_phrase'][:50]}")
    with open(f"gold_{lang}.jsonl", "w") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"wrote gold_{lang}.jsonl ({len(items)})\n")


if __name__ == "__main__":
    main()
