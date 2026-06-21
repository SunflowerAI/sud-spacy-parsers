#!/usr/bin/env python3
"""Relabel in-scope `udep` adposition-of-verb cases as comp:obl/mod, per language.

Scope = nominal adpositional phrase (no VERB in the adposition's subtree, <= 8 tokens) —
the same clean scope as the benchmark. For each target: if the confident rule (lang_gold.
classify) decides, use that label; otherwise query qwen3:8b with the per-language tuned
prompt (native instructions for id, English for zh/ko) + native few-shot from the gold.
Clausal/long udep are left as udep. Resumable per-language cache. Writes *.relabeled.conllu.
"""
import argparse, importlib.util, json, os, random
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)
_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)
_sg = importlib.util.spec_from_file_location("g", "scripts/lang_gold.py")
g = importlib.util.module_from_spec(_sg); _sg.loader.exec_module(g)
_sb = importlib.util.spec_from_file_location("b", "scripts/lang_bench.py")
b = importlib.util.module_from_spec(_sb); _sb.loader.exec_module(b)
_sz = importlib.util.spec_from_file_location("zb", "scripts/zh_bench.py")
zb = importlib.util.module_from_spec(_sz); _sz.loader.exec_module(zb)
_si = importlib.util.spec_from_file_location("idb", "scripts/id_bench.py")
idb = importlib.util.module_from_spec(_si); _si.loader.exec_module(idb)

# winning instruction per language (see lang_bench / zh_bench / id_bench results)
CHOSEN = {"id": b.NATIVE_DEF["id"], "zh": b.NATIVE_DEF["zh"], "ko": b.SUD_DEF}
# extra curated same-adposition contrastive shots that improved disambiguation
EXTRA_SHOTS = {"zh": zb.CONTRAST, "id": idb.CONTRAST}


def build_prefix(lang, seed=0, shots=8):
    gold = [json.loads(l) for l in open(f"gold_{lang}.jsonl") if l.strip()]
    rng = random.Random(seed)
    comp = [x for x in gold if x["gold"] == "complement"]; rng.shuffle(comp)
    mod = [x for x in gold if x["gold"] == "modifier"]; rng.shuffle(mod)
    ns = min(shots, len(comp) // 2, len(mod) // 2)
    s = EXTRA_SHOTS.get(lang, []) + comp[:ns] + mod[:ns]
    rng.shuffle(s)
    return CHOSEN[lang] + "Examples:\n\n" + b.shots_block(s)


def blocks(path):
    buf = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "":
            if buf:
                yield buf
            buf = []
        else:
            buf.append(line)
    if buf:
        yield buf


def parse_block(lines):
    toks, lineidx = [], {}
    for li, line in enumerate(lines):
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        toks.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3],
                     "head": int(c[6]), "deprel": c[7],
                     "space_after": "SpaceAfter=No" not in c[9]})
        lineidx[int(c[0])] = li
    return toks, lineidx


def in_scope(toks, by, t):
    if t["upos"] != "ADP" or t["deprel"] != "udep" or t["head"] == 0:
        return False
    if by.get(t["head"], {}).get("upos") != "VERB":
        return False
    sub = d.descendants(toks, t["id"])
    if any(by[i]["upos"] == "VERB" for i in sub if i != t["id"]):
        return False
    return len(sub) <= 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=g.FILES)
    args = ap.parse_args()
    lang = args.lang
    prefix = build_prefix(lang)
    cachep = f"relabel_cache_{lang}.jsonl"
    cache = {}
    if os.path.exists(cachep):
        for line in open(cachep):
            if line.strip():
                r = json.loads(line); cache[r["key"]] = r["label"]
    cfh = open(cachep, "a")
    stats = Counter()

    for path in g.FILES[lang]:
        out = []
        for si, lines in enumerate(blocks(path)):
            toks, lineidx = parse_block(lines)
            by = {t["id"]: t for t in toks}
            for t in toks:
                if not in_scope(toks, by, t):
                    continue
                head = by[t["head"]]
                lab = g.classify(toks, t, head, lang)        # confident rule first
                src = "rule"
                if lab is None:
                    key = f"{path}|{si}|{t['id']}"
                    lab = cache.get(key)
                    if lab is None:
                        item = {"verb": head["form"],
                                "prep_phrase": d.render(toks, d.descendants(toks, t["id"])),
                                "verb_phrase": d.render(toks, d.descendants(toks, head["id"]))}
                        raw = d.query(prefix + e.suffix(item))
                        lab = "complement" if raw.startswith("comp") else "modifier"
                        cache[key] = lab
                        cfh.write(json.dumps({"key": key, "label": lab}) + "\n"); cfh.flush()
                    src = "model"
                stats[f"{lab}:{src}"] += 1
                li = lineidx[t["id"]]
                cols = lines[li].split("\t")
                cols[7] = "comp:obl" if lab == "complement" else "mod"
                lines[li] = "\t".join(cols)
            out.append("\n".join(lines))
        outp = path.replace(".conllu", ".relabeled.conllu")
        open(outp, "w", encoding="utf-8").write("\n\n".join(out) + "\n\n")
        print(f"wrote {outp}")
    cfh.close()
    comp = stats["complement:rule"] + stats["complement:model"]
    mod = stats["modifier:rule"] + stats["modifier:model"]
    print(f"{lang}: comp:obl={comp} mod={mod} | by rule={stats['complement:rule']+stats['modifier:rule']} "
          f"by model={stats['complement:model']+stats['modifier:model']}")


if __name__ == "__main__":
    main()
