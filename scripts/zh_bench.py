#!/usr/bin/env python3
"""Tune the Chinese comp/mod prompt with curated same-coverb contrastive few-shot.

The baseline prompt is complement-biased (mod recall ~0.57): it over-reads 在/于 phrases as
complements. These curated minimal pairs teach that 在/于 is a COMPLEMENT only when the verb
selects a locative argument (位于/分布于/来自), but a MODIFIER with a time object (成立于1997年)
or a freely-added location (在公园散步). Test split matches lang_bench (seed 0) for comparison.
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

# curated contrastive shots: same coverb (于/在/自/对/与/为) in both roles
CONTRAST = [
    # COMPLEMENT: verb selects the (usually locative/source/object) argument
    S("这种植物分布于亚洲", "于亚洲", "分布", "complement"),
    S("这座城市位于南方", "于南方", "位", "complement"),
    S("他来自法国", "自法国", "来", "complement"),
    S("事故起源于疏忽", "于疏忽", "起源", "complement"),
    S("结果取决于努力", "于努力", "取决", "complement"),
    S("我们与邻国合作", "与邻国", "合作", "complement"),
    S("他对结果负责", "对结果", "负责", "complement"),
    S("公司为客户服务", "为客户", "服务", "complement"),
    S("这本书属于图书馆", "于图书馆", "属", "complement"),
    S("他向老师学习", "向老师", "学习", "complement"),
    # MODIFIER: 于/在 + time, freely-added location, reason, purpose, basis
    S("公司成立于1997年", "于1997年", "成立", "modifier"),
    S("他在2010年毕业", "在2010年", "毕业", "modifier"),
    S("这首歌发行于去年", "于去年", "发行", "modifier"),
    S("他于晚上到达", "于晚上", "到达", "modifier"),
    S("他们在公园散步", "在公园", "散步", "modifier"),
    S("会议在周一举行", "在周一", "举行", "modifier"),
    S("由于下雨比赛取消", "由于下雨", "取消", "modifier"),
    S("因身体原因他辞职", "因身体原因", "辞职", "modifier"),
    S("根据法律他被起诉", "根据法律", "起诉", "modifier"),
    S("为了健康他每天跑步", "为了健康", "跑步", "modifier"),
]


def test_split(seed=0, ns=8):
    gold = [json.loads(l) for l in open("gold_zh.jsonl") if l.strip()]
    rng = random.Random(seed)
    comp = [x for x in gold if x["gold"] == "complement"]; rng.shuffle(comp)
    mod = [x for x in gold if x["gold"] == "modifier"]; rng.shuffle(mod)
    gold_shots = comp[:ns] + mod[:ns]; rng.shuffle(gold_shots)
    comp_t, mod_t = comp[ns:], mod[ns:]
    n = min(len(comp_t), len(mod_t))
    test = comp_t[:n] + mod_t[:n]; rng.shuffle(test)
    return gold_shots, test


def run(name, prefix, test):
    preds = [d.query(prefix + e.suffix(c)) for c in test]
    preds = [p if p in ("complement", "modifier") else "?" for p in preds]
    acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
    rc = {c: sum(preds[i] == c for i, t in enumerate(test) if t["gold"] == c)
          / max(1, sum(t["gold"] == c for t in test)) for c in ("complement", "modifier")}
    print(f"  {name:22} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
          f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}")


def main():
    gold_shots, test = test_split()
    print(f"zh test {len(test)} ({len(test)//2}/class); curated shots {len(CONTRAST)}\n")
    variants = {
        "en_gold8 (baseline)":  b.SUD_DEF + "Examples:\n\n" + b.shots_block(gold_shots),
        "en_contrast":          b.SUD_DEF + "Examples:\n\n" + b.shots_block(CONTRAST),
        "native_contrast":      b.NATIVE_DEF["zh"] + "Examples:\n\n" + b.shots_block(CONTRAST),
        "native_contrast+gold": b.NATIVE_DEF["zh"] + "Examples:\n\n" + b.shots_block(CONTRAST + gold_shots),
    }
    for name, pre in variants.items():
        run(name, pre, test)


if __name__ == "__main__":
    main()
