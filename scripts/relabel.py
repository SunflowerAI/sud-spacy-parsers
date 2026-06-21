#!/usr/bin/env python3
"""Relabel SUD `udep` prepositional dependents of verbs as comp vs mod.

For every token that is a preposition (UPOS=ADP) attached to a verb (UPOS=VERB) by `udep`,
ask qwen3:8b (fewshot12_def prompt, no thinking, temp 0) whether the PP is a complement or
a modifier, and rewrite the relation: complement -> `comp:obl`, modifier -> `mod`. All other
lines are passed through unchanged. Resumable: every decision is cached to disk, so re-running
skips work already done.

Writes assets/en_sud-<split>.relabeled.conllu for split in {train,dev,test}.
"""
import importlib.util, json, os, time
from collections import Counter

_sd = importlib.util.spec_from_file_location("d", "scripts/disambiguate_pp.py")
d = importlib.util.module_from_spec(_sd); _sd.loader.exec_module(d)
_se = importlib.util.spec_from_file_location("e", "scripts/eval_prompts.py")
e = importlib.util.module_from_spec(_se); _se.loader.exec_module(e)

PREFIX = e.PREFIXES["fewshot12_def"]          # static -> KV-cache reused across every call
FILES = ["assets/en_sud-train.conllu", "assets/en_sud-dev.conllu", "assets/en_sud-test.conllu"]
CACHE = "relabel_cache.jsonl"


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); cache[r["key"]] = r["label"]
    return cache


def norm_label(raw):
    if raw.startswith("comp"):
        return "complement"
    if raw.startswith("mod"):
        return "modifier"
    return None  # unexpected -> caller falls back


def blocks(path):
    """Yield sentence blocks as lists of raw lines (without trailing newline)."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
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
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        toks.append({"id": int(cols[0]), "form": cols[1], "lemma": cols[2],
                     "upos": cols[3], "head": int(cols[6]), "deprel": cols[7],
                     "space_after": "SpaceAfter=No" not in cols[9]})
        lineidx[int(cols[0])] = li
    return toks, lineidx


def main():
    cache = load_cache()
    cache_fh = open(CACHE, "a")
    stats = Counter()
    # count targets up front for a progress denominator
    total = 0
    for path in FILES:
        for lines in blocks(path):
            toks, _ = parse_block(lines)
            by = {t["id"]: t for t in toks}
            for t in toks:
                if t["upos"] == "ADP" and t["deprel"] == "udep" and t["head"] != 0 \
                        and by.get(t["head"], {}).get("upos") == "VERB":
                    total += 1
    print(f"Targets (udep ADP-of-VERB): {total}  | cached so far: {len(cache)}\n", flush=True)

    done = 0
    t0 = time.time()
    for path in FILES:
        out_lines = []
        for si, lines in enumerate(blocks(path)):
            toks, lineidx = parse_block(lines)
            by = {t["id"]: t for t in toks}
            for t in toks:
                if not (t["upos"] == "ADP" and t["deprel"] == "udep" and t["head"] != 0
                        and by.get(t["head"], {}).get("upos") == "VERB"):
                    continue
                head = by[t["head"]]
                key = f"{path}|{si}|{t['id']}"
                label = cache.get(key)
                if label is None:
                    item = {"verb": head["form"],
                            "prep_phrase": d.render(toks, d.descendants(toks, t["id"])),
                            "verb_phrase": d.render(toks, d.descendants(toks, head["id"]))}
                    label = norm_label(d.query(PREFIX + e.suffix(item)))
                    if label is None:
                        label = "modifier"; stats["fallback"] += 1
                    cache[key] = label
                    cache_fh.write(json.dumps({"key": key, "label": label}) + "\n")
                    cache_fh.flush()
                stats[label] += 1
                done += 1
                li = lineidx[t["id"]]
                cols = lines[li].split("\t")
                cols[7] = "comp:obl" if label == "complement" else "mod"
                lines[li] = "\t".join(cols)
                if done % 500 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {done}/{total}  ({rate:.1f}/s, "
                          f"comp={stats['complement']} mod={stats['modifier']})", flush=True)
            out_lines.append("\n".join(lines))
        out_path = path.replace(".conllu", ".relabeled.conllu")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out_lines) + "\n\n")
        print(f"Wrote {out_path}", flush=True)

    cache_fh.close()
    print(f"\nDone. complement={stats['complement']} modifier={stats['modifier']} "
          f"fallback={stats['fallback']}  in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
