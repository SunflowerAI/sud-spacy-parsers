#!/usr/bin/env python3
"""Benchmark comp-vs-mod prompt variants per language on the confident udep-derived gold
(gold_<lang>.jsonl from lang_gold.py). Compares English vs native-language instructions,
each with native few-shot examples drawn from the gold (held out from the test set).
Prompts are static-prefix + short-suffix for KV-cache reuse.
"""
import argparse, importlib.util, json, random
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)
_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)

SUD_DEF = (
    "You will be given a verb and an adpositional phrase that depends on it. Decide whether "
    "the phrase is a COMPLEMENT or a MODIFIER of the verb.\n"
    "- COMPLEMENT: an argument the verb selects — an obligatory participant in the verb's "
    "meaning; the verb requires or licenses it and typically fixes the adposition. Only "
    "certain verbs admit it.\n"
    "- MODIFIER: an optional circumstantial (time, place, manner, reason, accompaniment) "
    "not part of the verb's argument structure, attachable freely to many verbs.\n"
    "Answer with exactly one word: complement or modifier.\n\n")

NATIVE_DEF = {
    "zh": ("你将看到一个动词和一个依存于它的介词短语。请判断该短语是该动词的补足语还是修饰语。\n"
           "- 补足语（complement）：动词所要求的论元，是动词意义中不可或缺的参与者；动词通常选择特定的介词。"
           "只有某些动词才能带这种短语。\n"
           "- 修饰语（modifier）：可有可无的状语（时间、地点、方式、原因、伴随等），不属于动词的论元结构，"
           "可以自由地加在许多动词上。\n"
           "只用一个英文单词回答：complement 或 modifier。\n\n"),
    "ko": ("동사와, 그 동사에 의존하는 부사어(명사+조사)가 주어집니다. 그 부사어가 동사의 보충어인지 "
           "수식어인지 판단하세요.\n"
           "- 보충어(complement): 동사가 요구하는 논항으로, 동사 의미에 필수적인 참여자입니다. 특정 동사만 "
           "이 성분을 취합니다.\n"
           "- 수식어(modifier): 시간·장소·방식·이유 등 임의적인 부가어로, 동사의 논항 구조에 속하지 않고 "
           "여러 동사에 자유롭게 붙습니다.\n"
           "영어 한 단어로만 답하세요: complement 또는 modifier.\n\n"),
    "id": ("Anda akan diberi sebuah verba dan sebuah frasa preposisi yang bergantung padanya. "
           "Tentukan apakah frasa itu KOMPLEMEN atau MODIFIER dari verba tersebut.\n"
           "- KOMPLEMEN (complement): argumen yang dituntut verba — peserta wajib dalam makna "
           "verba; verba memerlukannya dan biasanya menentukan preposisinya. Hanya verba "
           "tertentu yang dapat memilikinya.\n"
           "- MODIFIER (modifier): keterangan opsional (waktu, tempat, cara, sebab, penyerta) "
           "yang bukan bagian dari struktur argumen verba dan dapat ditambahkan bebas.\n"
           "Jawab dengan tepat satu kata: complement atau modifier.\n\n"),
}


def shots_block(shots):
    return "".join(
        f'In "{s["verb_phrase"]}", is "{s["prep_phrase"]}" a complement or a modifier '
        f'of "{s["verb"]}"?\nAnswer: {s["gold"]}\n\n' for s in shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(f"gold_{args.lang}.jsonl") if l.strip()]
    rng = random.Random(args.seed)
    comp = [x for x in gold if x["gold"] == "complement"]; rng.shuffle(comp)
    mod = [x for x in gold if x["gold"] == "modifier"]; rng.shuffle(mod)
    ns = min(args.shots, len(comp) // 2, len(mod) // 2)
    shots = comp[:ns] + mod[:ns]; rng.shuffle(shots)
    comp_t, mod_t = comp[ns:], mod[ns:]
    n = min(args.per_class, len(comp_t), len(mod_t))
    test = comp_t[:n] + mod_t[:n]; rng.shuffle(test)
    print(f"=== {args.lang}: gold {len(comp)}c/{len(mod)}m | shots {ns}/class | "
          f"test {len(test)} ({n}/class) ===")

    prefixes = {
        "en_fewshot":     SUD_DEF + "Examples:\n\n" + shots_block(shots),
        "native_fewshot": NATIVE_DEF[args.lang] + "Examples:\n\n" + shots_block(shots),
    }
    for name, prefix in prefixes.items():
        preds = [d.query(prefix + e.suffix(c)) for c in test]
        preds = [p if p in ("complement", "modifier") else "?" for p in preds]
        acc = sum(p == t["gold"] for p, t in zip(preds, test)) / len(test)
        rc = {cls: sum(preds[i] == cls for i, t in enumerate(test) if t["gold"] == cls)
              / max(1, sum(t["gold"] == cls for t in test)) for cls in ("complement", "modifier")}
        print(f"  {name:15} acc={acc:.3f}  rec[c]={rc['complement']:.3f}  "
              f"rec[m]={rc['modifier']:.3f}  pred={dict(Counter(preds))}")


if __name__ == "__main__":
    main()
