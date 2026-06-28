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
    "fa": ["assets_fa/SUD_Persian-PerDT/fa_perdt-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "ar": ["assets_ar/SUD_Arabic-PADT/ar_padt-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "la": ["assets_la/la_ittbproiel-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "sa": ["assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "lzh": ["assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "ja": ["assets_ja/SUD_Japanese-GSD/ja_gsd-sud-%s.conllu" % s for s in ("train", "dev", "test")],
    "yue": ["assets_yue/SUD_Cantonese-HK/yue_hk-sud-%s.conllu" % s for s in ("train", "dev", "test")],
}

# Languages whose `udep` sits on bare case-marked nominals (almost no adpositions): the gold
# is built from the dependent's morphological Case, not an adposition lemma (cf. Korean).
CASE_LANGS = {"sa"}
# Classical Chinese: SUD leaves most coverbs as `udep` and commits very few, so both the
# (verb, adp) frames and the mod-adposition heuristic are too sparse/noisy to trust. The
# highest-precision signal available is SUD's own committed comp:obl/mod ADP<-VERB labels,
# so for lzh the whole gold is built from those (relaxed scope: 於-complements take clausal
# objects, so the no-nested-VERB filter is dropped here).
COMMITTED_GOLD_LANGS = {"lzh"}
# Classical Chinese coverb (於/于 ...): the object noun carries the treebank's own semantic
# category in FEATS Case and XPOS — Case=Tem ("時" time nouns) and Case=Loc (place nouns). We
# use that, plus the head verb's semantic class (XPOS field 3), instead of leaning on the LLM
# (which is barely above chance here). A temporal object is always a WHEN adjunct -> mod. A
# locative object is a selected locus (-> comp:obl) only under verbs that lexically take a
# location — motion (移動), posture (姿勢), placement (設置), existence (存在), life/birth-death
# (生物); under any other verb the place is a circumstantial adjunct -> mod.
LZH_LOC_COMP_VCLASS = {"移動", "姿勢", "設置", "存在", "生物"}

# Sanskrit: confident MODIFIER cases (committed compShare <= ~.05 -> circumstantial place/
# time/source/address/predication). Complements come from the derived (verb, Case) frames —
# notably the recipient DATIVE of giving verbs (dā/prayam/pradā + Dat), per Pāṇinian
# sampradāna; a blanket Dat -> comp is avoided because the dative-of-purpose is adjunctival.
# (Vedic ritual "offer into the fire-LOC" loci are committed udep/mod, not comp, so Loc -> mod.)
SA_MOD_CASES = {"Loc", "Abl", "Voc", "Nom"}

# adpositions that are inherently reason/temporal adjuncts -> MODIFIER. fa is empty: PerDT
# commits ~99% of verb-PPs as comp:obl, so Persian modifiers come only from temporal objects.
MOD_ADP = {
    "zh": {"因", "由于", "为了", "根据", "按照", "经过", "自从"},
    "ko": {"때문+에", "후", "이후", "전", "동안", "때", "중", "중+에", "만+에", "무렵", "사이"},
    "id": {"selama", "sejak", "setelah", "sebelum", "hingga", "sampai", "karena", "menjelang"},
    "fa": set(),
    "ar": {"بَعدَ", "قَبلَ", "كَ", "خِلَالَ", "أَثنَاءَ", "مُنذُ", "عِندَ", "حَتَّى"},
    "la": {"secundum", "propter", "pro", "per", "ob", "ante", "post", "sine", "circa", "ergo"},
    "lzh": set(),  # lzh gold is committed-label only (see COMMITTED_GOLD_LANGS)
    # Japanese: comparative より and terminative まで particles are circumstantial -> mod.
    "ja": {"より", "まで"},
    # Cantonese coverbs that are inherently circumstantial -> mod: basis (根據/按照/依照
    # according-to), instrumental/means (透過 through), purpose (為咗 in-order-to), reason
    # (因為 because), and the postpositional temporal relators (之後/之前/後/前 after/before).
    "yue": {"根據", "按照", "依照", "透過", "為咗", "因為", "之後", "之前", "後", "前",
            "嗰陣時", "嗰時"},
}
# adpositions that take a temporal/year object -> MODIFIER (if object is temporal)
TEMP_OBJ_ADP = {
    "zh": {"于", "在", "自"}, "ko": set(), "id": {"pada", "di", "pada"},
    "fa": {"در", "از", "تا", "به"}, "ar": {"فِي", "خِلَالَ", "عِندَ", "مُنذُ", "بَعدَ", "قَبلَ"},
    "la": {"in", "ad", "ante", "post", "per"}, "lzh": {"於", "于"},
    "ja": {"に", "で", "から", "まで"},
    # Cantonese locative/source/terminal coverbs that take a temporal object -> WHEN adjunct.
    "yue": {"喺", "响", "由", "到"},
}
TEMP_NOUN = {
    "zh": {"年", "月", "日", "时", "时候", "期间", "初", "末", "世纪", "年代", "凌晨",
           "早上", "上午", "中午", "下午", "晚上", "当天", "当时", "前", "后"},
    "ko": set(),
    "id": {"tahun", "bulan", "hari", "minggu", "pagi", "siang", "sore", "malam", "abad",
           "dekade", "masa", "waktu", "saat", "zaman", "era", "pukul",
           "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
           "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
           "september", "oktober", "november", "desember"},
    "fa": {"سال", "ماه", "روز", "هفته", "ساعت", "دقیقه", "صبح", "ظهر", "عصر", "شب", "بامداد",
           "قرن", "دهه", "زمان", "وقت", "هنگام", "موقع", "دوره", "دوران", "امروز", "دیروز",
           "فردا", "اکنون", "آغاز", "پایان", "ابتدا", "انتها", "بار"},
    "ar": {"يَوم", "شَهر", "عَام", "سَنَة", "سَاعَة", "دَقِيقَة", "أُسبُوع", "صَبَاح", "مَسَاء",
           "لَيل", "لَيلَة", "ظُهر", "فَجر", "قَرن", "عَقد", "وَقت", "زَمَن", "زَمَان", "فَترَة",
           "عَصر", "حِين", "لَحظَة", "بِدَايَة", "نِهَايَة", "مَوسِم"},
    "la": {"annus", "dies", "mensis", "hora", "tempus", "nox", "aetas", "saeculum",
           "hebdomada", "mane", "vesper", "aestas", "hiems", "uer", "autumnus", "momentum",
           "initium", "finis", "principium", "hodie", "cras", "heri", "nunc"},
    "lzh": {"年", "月", "日", "時", "歲", "春", "夏", "秋", "冬", "旦", "夕", "朝", "夜",
            "世", "古", "今", "昔", "初", "末", "晨", "暮", "始", "終"},
    "ja": {"年", "月", "日", "時", "分", "秒", "週", "時間", "時代", "時期", "時刻", "頃",
           "朝", "昼", "夜", "夕方", "午前", "午後", "正午", "世紀", "年代", "期間", "当時",
           "現在", "今日", "明日", "昨日", "今", "後", "前", "間", "際", "末", "初", "始め",
           "初め", "中", "春", "夏", "秋", "冬", "月曜", "火曜", "水曜", "木曜", "金曜",
           "土曜", "日曜"},
    "yue": {"而家", "今日", "聽日", "琴日", "頭先", "以前", "以後", "之後", "之前", "嗰陣時",
            "嗰時", "舊時", "將來", "時候", "年", "月", "日", "點", "點鐘", "朝早", "晏晝",
            "夜晚", "夜", "朝", "宜家", "而今", "依家", "陣間", "陣", "禮拜", "星期"},
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
    # Japanese: GSD commits almost no comp:obl on verb particles (they stay udep), so frames
    # are curated from the frequent (verb-lemma, particle) udep collocations + linguistic
    # judgement: に/へ goal-result-recipient, と quotative/reciprocal/result, から source.
    "ja": {("成る", "に"), ("為る", "に"), ("行く", "に"), ("来る", "に"), ("入る", "に"),
           ("向かう", "に"), ("向ける", "に"), ("達する", "に"), ("至る", "に"), ("着く", "に"),
           ("乗る", "に"), ("付く", "に"), ("置く", "に"), ("入れる", "に"), ("加える", "に"),
           ("言う", "に"), ("伝える", "に"), ("教える", "に"), ("与える", "に"), ("渡す", "に"),
           ("送る", "に"), ("関する", "に"), ("対する", "に"), ("基づく", "に"), ("因る", "に"),
           ("応じる", "に"), ("属する", "に"), ("当たる", "に"), ("移す", "に"),
           ("成る", "と"), ("為る", "と"), ("呼ぶ", "と"), ("言う", "と"), ("称する", "と"),
           ("思う", "と"), ("考える", "と"), ("比べる", "と"), ("並ぶ", "と"), ("異なる", "と"),
           ("会う", "と"), ("化する", "と"), ("述べる", "と"),
           ("向かう", "へ"), ("行く", "へ"), ("赴く", "へ"), ("移す", "へ"), ("入る", "へ"),
           ("受ける", "から"), ("離れる", "から"), ("成る", "から"), ("始まる", "から"),
           ("来る", "から"), ("集める", "から"), ("出る", "から")},
    # Cantonese: the coverb is lexically selected by the verb -> COMPLEMENT. Three frames,
    # each genuinely ambiguous against an adjunct reading the LLM must otherwise resolve:
    #   畀 bei2 (dative/benefactive 'give'): recipient of a transfer verb -> comp:obl
    #   到 dou3 ('arrive/to'): goal of a motion verb -> comp:obl  (vs terminal-time -> mod)
    #   喺/响 hai2 ('at/in'): selected locus of a position/placement/copula verb -> comp:obl
    #   由 jau4 ('from'): source of a motion verb -> comp:obl
    "yue": {("送", "畀"), ("攞", "畀"), ("打賞", "畀"), ("講", "畀"), ("寄", "畀"), ("畀", "畀"),
            ("聽", "畀"), ("頂", "畀"), ("賣", "畀"), ("交", "畀"), ("還", "畀"), ("俾", "畀"),
            ("去", "到"), ("嚟", "到"), ("唻", "到"), ("返", "到"), ("行", "到"), ("走", "到"),
            ("搬", "到"), ("送", "到"), ("帶", "到"),
            ("住", "喺"), ("徛", "喺"), ("坐", "喺"), ("企", "喺"), ("放", "喺"), ("擺", "喺"),
            ("瞓", "喺"), ("進入", "喺"), ("位於", "喺"), ("係", "喺"), ("帶孫", "喺"),
            ("玩", "响"), ("係", "响"),
            ("嚟", "由"), ("唻", "由"), ("係", "由")},
}


def _feat(feats, key):
    import re
    m = re.search(rf"{key}=([^|]+)", feats or "")
    return m.group(1) if m else "_"


def _derive_comp_frames(lang, thresh=0.85, minc=8):
    """Data-driven (verb-lemma, adposition) complement frames: pairs that are >= `thresh`
    comp:obl among committed ADP<-VERB deps in the TRAIN split (>= `minc` comp instances).
    fa/ar/la each have ~150-200 such frames; deriving them keeps the gold reproducible and
    avoids hand-listing them. zh/ko/id keep their hand-curated literals above."""
    fr = Counter()
    seen = Counter()
    for sid, toks in d.parse_conllu(FILES[lang][0]):
        by = {t["id"]: t for t in toks}
        for t in toks:
            if t["upos"] != "ADP":
                continue
            h = by.get(t["head"])
            if not h or h["upos"] != "VERB":
                continue
            key = (h["lemma"], t["lemma"])
            if t["deprel"].startswith("comp:obl"):
                fr[key] += 1; seen[key] += 1
            elif t["deprel"].split(":")[0].split("@")[0] == "mod":
                seen[key] += 1
    return {k for k, c in fr.items() if c >= minc and c / seen[k] >= thresh}


def derive_sa_comp_frames(thresh=0.85, minc=4):
    """Sanskrit (verb-lemma, Case) complement frames from committed VERB<-nominal deps."""
    fr, seen = Counter(), Counter()
    for sid, toks in d.parse_conllu(FILES["sa"][0]):
        by = {t["id"]: t for t in toks}
        for t in toks:
            h = by.get(t["head"])
            if not h or h["upos"] != "VERB" or t["upos"] not in ("NOUN", "PRON", "ADJ", "NUM"):
                continue
            key = (h["lemma"], _feat(t.get("feats"), "Case"))
            if t["deprel"].startswith("comp:obl"):
                fr[key] += 1; seen[key] += 1
            elif t["deprel"].split(":")[0].split("@")[0] == "mod":
                seen[key] += 1
    return {k for k, c in fr.items() if c >= minc and c / seen[k] >= thresh}


for _l in ("fa", "ar", "la"):
    COMP_FRAMES[_l] = _derive_comp_frames(_l)
# Classical Chinese commits very few verb-PPs, so the default minc=8/thresh=.85 derives *no*
# frames (the verb-selection signal is real but sparse). Loosen to minc=2/thresh=.70 — this
# recovers ~15 clean selectional 於/與 frames (至於 arrive-at, 達於 reach, 在於 lie-in, 異於
# differ-from, 甚於 exceed, 長於 excel-at, 怒於 angry-at, ...) that the LLM otherwise adjudicates
# unreliably. Per-label `comp:obl` (over committed deps) stays the calibration target.
COMP_FRAMES["lzh"] = _derive_comp_frames("lzh", thresh=0.70, minc=2)

# Sanskrit is case-based (handled via Case features, not adposition lemmas); give it empty
# adposition tables so classify()/obj_is_temporal don't KeyError on its few stray ADP udep.
MOD_ADP["sa"] = set()
TEMP_NOUN["sa"] = set()
COMP_FRAMES["sa"] = set()


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


def xpos_field(xpos, i):
    p = (xpos or "").split(",")
    return p[i] if len(p) > i else ""


def lzh_coverb_label(toks, prep, head):
    """Classical Chinese coverb decided from the object's annotated semantic category (FEATS
    Case) and the head verb's semantic class (XPOS field 3). Temporal object -> mod; locative
    object -> comp:obl under a locus-selecting verb class, else mod. Returns None (-> caller's
    frame logic / model) when the object carries no Case (person/abstract objects)."""
    obj = prep_object(toks, prep["id"])
    if not obj:
        return None
    case = _feat(obj.get("feats"), "Case")
    if case == "Tem":
        return "modifier"
    if case == "Loc" and head["upos"] == "VERB":
        return "complement" if xpos_field(head.get("xpos"), 3) in LZH_LOC_COMP_VCLASS else "modifier"
    return None


def classify(toks, prep, head, lang):
    if lang == "lzh":
        lab = lzh_coverb_label(toks, prep, head)
        if lab is not None:
            return lab
    vroot, adp = root(head["lemma"], lang), prep["lemma"]
    if (vroot, adp) in COMP_FRAMES[lang]:
        # temporal-object override: 成立于1997 / کرد در ۱۹۹۹ are temporal-WHEN adjuncts, not
        # locative complements (zh first had this; same logic for the new prepositional langs)
        if lang in ("zh", "fa", "ar", "la", "lzh", "ja", "yue") and obj_is_temporal(toks, prep["id"], lang):
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
    if lang in COMMITTED_GOLD_LANGS:
        _write_gold(lang, _harvest_committed(lang), 0)
        return
    if lang in CASE_LANGS:
        sa_frames = derive_sa_comp_frames()
        for f in FILES[lang]:
            for sid, toks in d.parse_conllu(f):
                by = {t["id"]: t for t in toks}
                for t in toks:
                    if t["deprel"] != "udep" or t["head"] == 0:
                        continue
                    if t["upos"] not in ("NOUN", "PRON", "ADJ", "NUM"):
                        continue
                    head = by.get(t["head"])
                    if not head or head["upos"] != "VERB":
                        continue
                    sub = d.descendants(toks, t["id"])
                    if any(by[i]["upos"] == "VERB" for i in sub if i != t["id"]):
                        continue  # scope: drop clausal phrases
                    if len(sub) > 8:
                        continue
                    case = _feat(t.get("feats"), "Case")
                    if (head["lemma"], case) in sa_frames:
                        label = "complement"
                    elif case in SA_MOD_CASES:
                        label = "modifier"
                    else:
                        dropped += 1
                        continue
                    items.append({"verb": head["form"], "adp": "Case=" + case,
                                  "prep_phrase": d.render(toks, sub),
                                  "verb_phrase": d.render(toks, d.descendants(toks, head["id"])),
                                  "gold": label})
        _write_gold(lang, items, dropped)
        return
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
    _write_gold(lang, items, dropped)


def base_deprel(deprel):
    return deprel.split(":")[0].split("@")[0]


def _harvest_committed(lang):
    """Build the gold from SUD's committed comp:obl/mod ADP<-VERB labels (the treebank's own
    confident decisions). Scope: adposition subtree <= 10 tokens; the no-nested-VERB filter is
    dropped (lzh complements take clausal objects). Temporal objects override comp:obl -> mod."""
    out = []
    for f in FILES[lang]:
        for sid, toks in d.parse_conllu(f):
            by = {t["id"]: t for t in toks}
            for t in toks:
                if t["upos"] != "ADP" or t["head"] == 0:
                    continue
                if t["deprel"].startswith("comp:obl"):
                    label = "complement"
                elif base_deprel(t["deprel"]) == "mod":
                    label = "modifier"
                else:
                    continue
                head = by.get(t["head"])
                if not head or head["upos"] != "VERB":
                    continue
                sub = d.descendants(toks, t["id"])
                if len(sub) > 10:
                    continue
                if label == "complement" and obj_is_temporal(toks, t["id"], lang):
                    label = "modifier"  # temporal-WHEN phrase, not a locative complement
                out.append({"verb": head["form"], "adp": t["lemma"],
                            "prep_phrase": d.render(toks, sub),
                            "verb_phrase": d.render(toks, d.descendants(toks, head["id"])),
                            "gold": label})
    return out


def _write_gold(lang, items, dropped):
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
