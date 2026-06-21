#!/usr/bin/env python3
"""Extended-scope `udep` relabelling (cases 1-4 + the tractable residue): beyond ADP-of-VERB.

Scope added on top of the baseline ADP-of-VERB relabel:
  1. ADP dependents of NOUN / PROPN / ADJ heads        (noun & adjective complements/mods)
  4. ADP dependents of VERB, any subtree size, INCLUDING clausal PPs (a VERB in the subtree):
     'depend on how we interpret', 'known of his plans' — same verb-selection question.
  + Participial complex prepositions (VERB dependent of a VERB: 'according to', 'based on',
     'following ...') -> mostly mod.
  + Korean bare NOUN dependent of a VERB, disambiguated by the CASE SUFFIX read off the
     rightmost eojeol of the phrase (head-final): locative/temporal/comitative cases -> mod,
     dative 에게 -> comp, selecting (verb,case) frames -> comp, 로/으로/topic/bare -> model.
  2. zh: 的/之 associative PART under a NOUN -> mod      (deterministic; committed 737:3 mod)
  3. ko: ADV dependent of a VERB         -> mod          (deterministic; committed 3554:1 mod)

Left as `udep` (the genuinely-different residue): SCONJ clausal complements vs adverbial
clauses ('comfort yourself THAT ...'), and partitive NUM/DET/PRON heads ('one of them') — by
user decision the partitives stay ambiguous. NOUN/PROPN/ADJ ADP heads stay nominal-only (a
clausal noun complement is the harder set); only VERB heads drop the nominal filter.

Decision per target: deterministic rule first (zh-的, ko case-suffix, participial, and the
lang_gold frame/temporal rules for zh/ko/id), else qwen3:8b with the SAME tuned per-language
prompt and suffix as the baseline (the suffix already names the head word generically, so it
reads naturally for noun and adjective heads). Resumable: relabel_cache_ext_<lang>.jsonl,
seeded from the baseline cache so decided verb targets are not re-queried. Writes
*.relabeled_ext.conllu.
"""
import argparse, importlib.util, json, os, time
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)
_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)
_sg = importlib.util.spec_from_file_location("g", "scripts/lang_gold.py")
g = importlib.util.module_from_spec(_sg); _sg.loader.exec_module(g)
_sl = importlib.util.spec_from_file_location("lr", "scripts/lang_relabel.py")
lr = importlib.util.module_from_spec(_sl); _sl.loader.exec_module(lr)

EN_FILES = ["assets/en_sud-train.conllu", "assets/en_sud-dev.conllu", "assets/en_sud-test.conllu"]
FILES = {"en": EN_FILES, **g.FILES}
# baseline caches to seed from (verb decisions already made), per language
BASE_CACHE = {"en": "relabel_cache.jsonl", "zh": "relabel_cache_zh.jsonl",
              "ko": "relabel_cache_ko.jsonl", "id": "relabel_cache_id.jsonl"}
ADP_HEADS = ("VERB", "NOUN", "PROPN", "ADJ")

# English participial/complex prepositions -> circumstantial modifier (match on form).
EN_PARTICIPIAL_MOD = {"according", "based", "following", "regarding", "concerning",
                      "including", "excluding", "owing", "depending", "pertaining",
                      "barring", "notwithstanding", "compared", "considering"}

# Korean case particles read off the rightmost eojeol, calibrated from committed VERB
# dependents (compShare = comp/(comp+mod)). Firmly modifier (compShare <= ~0.1):
KO_MOD_CASES = {"에", "에서", "까지", "부터", "과", "와", "보다", "처럼", "로서", "으로서",
                "으로써", "로써", "마다", "대로", "만큼", "같이", "의", "간", "동안", "밖에",
                "년", "월", "일", "시", "분", "로부터", "으로부터", "서", "상", "번", "씩",
                "께서", "라도", "이라도"}
# Dative recipient -> oblique complement (compShare ~0.70):
KO_COMP_CASES = {"에게", "께", "한테"}


def prefix_for(lang):
    return e.PREFIXES["fewshot12_def"] if lang == "en" else lr.build_prefix(lang)


def ko_case(toks, by, tid):
    """Case particle on the phrase: suffix after the last '+' of the rightmost (head-final)
    non-punct eojeol in the subtree. Returns the bare lemma if there is no '+' (no particle)."""
    sub = [i for i in d.descendants(toks, tid) if by[i]["upos"] != "PUNCT"]
    last = by[max(sub)]
    return last["lemma"].split("+")[-1]


def ko_case_label(toks, by, t, head):
    case = ko_case(toks, by, t["id"])
    vroot = head["lemma"].split("+")[0]
    if (vroot, case) in g.COMP_FRAMES["ko"]:        # verb lexically selects this case
        return "complement"
    if case in KO_MOD_CASES:
        return "modifier"
    if case in KO_COMP_CASES:
        return "complement"
    if t["upos"] == "ADV":                          # ADV residue is ~99.97% modifier
        return "modifier"
    return None                                     # bare/topic/로·으로 NOUN -> model


def has_verb_descendant(toks, by, t):
    sub = d.descendants(toks, t["id"])
    return any(by[i]["upos"] == "VERB" for i in sub if i != t["id"])


def targets(lang, toks, by):
    """Yield (token, head, bucket) for every extended-scope udep target."""
    for t in toks:
        if t["deprel"] != "udep" or t["head"] == 0:
            continue
        head = by.get(t["head"])
        if not head:
            continue
        if lang == "zh" and t["upos"] == "PART" and t["lemma"] in ("的", "之") and head["upos"] == "NOUN":
            yield t, head, "zh_de"
            continue
        if lang == "ko" and t["upos"] in ("ADV", "NOUN") and head["upos"] == "VERB":
            if has_verb_descendant(toks, by, t):       # nominal only (skip relative clauses)
                continue
            yield t, head, "ko_case"
            continue
        if t["upos"] == "VERB" and head["upos"] == "VERB":   # participial complex preposition
            yield t, head, "participial"
            continue
        if t["upos"] == "ADP" and head["upos"] in ADP_HEADS:
            # VERB heads: unfiltered — include clausal PPs (same selection question). Noun /
            # propn / adj heads stay nominal-only (clausal noun complements are the harder set).
            if head["upos"] != "VERB" and has_verb_descendant(toks, by, t):
                continue
            yield t, head, f"adp_{head['upos']}"


def rule_label(lang, toks, by, t, head, bucket):
    if bucket == "zh_de":
        return "modifier"
    if bucket == "ko_case":
        return ko_case_label(toks, by, t, head)
    if bucket == "participial":
        return "modifier" if t["form"].lower() in EN_PARTICIPIAL_MOD else None
    if bucket.startswith("adp_") and lang in g.FILES:   # zh/ko/id frame & temporal rules
        return g.classify(toks, t, head, lang)
    return None


def seed_cache(path):
    cache = {}
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line); cache[r["key"]] = r["label"]
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=FILES)
    ap.add_argument("--rules-only", action="store_true", help="no model calls; report only")
    ap.add_argument("--model-sample", type=int, default=0,
                    help="if >0, only call the model for the first N uncached targets (probe)")
    args = ap.parse_args()
    lang = args.lang
    prefix = prefix_for(lang)

    extcache_p = f"relabel_cache_ext_{lang}.jsonl"
    cache = seed_cache(BASE_CACHE.get(lang))          # reuse baseline verb decisions
    cache.update(seed_cache(extcache_p))              # then any prior ext decisions
    cfh = None if args.rules_only else open(extcache_p, "a")

    stats = Counter()                                 # bucket/source/label tallies
    model_calls = 0
    t0 = time.time()
    for path in FILES[lang]:
        out = []
        for si, lines in enumerate(lr.blocks(path)):
            toks, lineidx = lr.parse_block(lines)
            by = {t["id"]: t for t in toks}
            for t, head, bucket in targets(lang, toks, by):
                lab = rule_label(lang, toks, by, t, head, bucket)
                src = "rule"
                if lab is None:
                    key = f"{path}|{si}|{t['id']}"
                    lab = cache.get(key)
                    src = "cache"
                    if lab is None:
                        if args.rules_only or (args.model_sample and model_calls >= args.model_sample):
                            stats[f"{bucket}|needs_model"] += 1
                            continue
                        item = {"verb": head["form"],
                                "prep_phrase": d.render(toks, d.descendants(toks, t["id"])),
                                "verb_phrase": d.render(toks, d.descendants(toks, head["id"]))}
                        raw = d.query(prefix + e.suffix(item))
                        lab = "complement" if raw.startswith("comp") else "modifier"
                        cache[key] = lab; model_calls += 1
                        cfh.write(json.dumps({"key": key, "label": lab}) + "\n"); cfh.flush()
                        src = "model"
                stats[f"{bucket}|{src}|{lab}"] += 1
                if not args.rules_only:
                    li = lineidx[t["id"]]
                    cols = lines[li].split("\t")
                    cols[7] = "comp:obl" if lab == "complement" else "mod"
                    lines[li] = "\t".join(cols)
            out.append("\n".join(lines))
        if not args.rules_only:
            outp = path.replace(".conllu", ".relabeled_ext.conllu")
            open(outp, "w", encoding="utf-8").write("\n\n".join(out) + "\n\n")
            print(f"wrote {outp}")
    if cfh:
        cfh.close()

    # ---- report ----
    buckets = sorted({k.split("|")[0] for k in stats})
    print(f"\n=== {lang}: extended-scope udep targets "
          f"({model_calls} model calls, {(time.time()-t0)/60:.1f} min) ===")
    print(f"{'bucket':10} {'comp':>6} {'mod':>6} {'by_rule':>8} {'by_model+cache':>15} {'needs_model':>12}")
    gc = gm = 0
    for b in buckets:
        comp = stats[f"{b}|rule|complement"] + stats[f"{b}|model|complement"] + stats[f"{b}|cache|complement"]
        mod = stats[f"{b}|rule|modifier"] + stats[f"{b}|model|modifier"] + stats[f"{b}|cache|modifier"]
        byrule = stats[f"{b}|rule|complement"] + stats[f"{b}|rule|modifier"]
        bymc = (stats[f"{b}|model|complement"] + stats[f"{b}|model|modifier"]
                + stats[f"{b}|cache|complement"] + stats[f"{b}|cache|modifier"])
        need = stats[f"{b}|needs_model"]
        gc += comp; gm += mod
        print(f"{b:10} {comp:6} {mod:6} {byrule:8} {bymc:15} {need:12}")
    print(f"{'TOTAL':10} {gc:6} {gm:6}")


if __name__ == "__main__":
    main()
