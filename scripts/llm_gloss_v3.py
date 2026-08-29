#!/usr/bin/env python3
"""Gloss a treebank's tokens into English with a local LLM, grounded on Wiktionary.

WHY, given that Wiktionary glosses already exist. Two things a dictionary cannot do and this can:

  * COVER a token it has no entry for. Wiktionary reaches 37 % of held-out tokens by surface form;
    a real annotator glosses every one, and the three treebanks that ship a human `Gloss=` column
    run 67-81 %. So the measured gloss-fill result was taken on a channel filled roughly half as
    densely as deployment would fill it.
  * DISAMBIGUATE. A Wiktionary bag is every sense at once -- Georgian divani gives
    "divan sofa muslim council state supreme court ottoman empire collection poems ..." -- and
    averaging that is a blurry centroid rather than the word's meaning in this sentence.

⚠ THE WIKTIONARY BAG GOES IN THE PROMPT, which is what makes a 9 GB local model usable here. With
the candidate senses supplied, glossing Old Armenian is a SELECTION task rather than a recall task.
Without them the model is being asked what it knows about a language it has barely seen, and it
will answer confidently either way.

⚠ ONE CALL PER WINDOW OF ~15 TOKENS, WITH THE WHOLE SENTENCE AS CONTEXT. Per-token would be 10 000
calls a language; per-SENTENCE was tried first and failed on length, because a model cannot hold
array alignment over a long list. Measured on gemma4, fallback by sentence length:

    0-9 tokens   0 %      20-29   67 %      40-49   87 %
    10-19       18 %      30-39   76 %      50+    100 %

Yoruba's median sentence is 25 tokens, so a THIRD to a HALF of sentences never got a model gloss at
all and quietly fell back to the dictionary -- which dragged the measured quality toward the
dictionary's and made it look as though the model was barely better. Windows bound the output length
where the failures actually come from, and a failing window now falls back ALONE rather than taking
its whole sentence with it.

⚠ A WRONG-LENGTH REPLY IS A FAILURE, NEVER PADDED OR TRUNCATED. A misaligned gloss list attaches
every gloss to the wrong token, and nothing downstream would notice: the channel would fill at a
normal rate with confident nonsense, which is worse than leaving it empty.

⚠ THIS CHANGES THE DEPLOYMENT CLAIM and the write-up must say so. "You supply glosses" becomes "a
model supplies glosses" -- which needs no annotation at all, but makes the MODEL's language coverage
the binding constraint. It is also worth being honest that a model able to gloss K'iche' in context
is not far from doing more of the task than glossing.

GRADED BEFORE IT IS TRUSTED. `--grade` scores against a treebank's own human `Gloss=` column,
reporting agreement in the shared English space -- cos(vec(model gloss), vec(human gloss)) against a
shuffled control -- so the glosser is measured on Classical Armenian and Yoruba, where humans
already did the job, before it is believed anywhere else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = os.environ.get("OLLAMA_URL") or _HOST.rstrip("/") + "/api/generate"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

# ⚠ FIXED TEXT FIRST, VARIABLE TEXT LAST -- but do not expect KV reuse from it. Measured on a
# QUIET server, consecutive windows of one sentence take 5.12s / 5.46s / 5.45s of prompt_eval while
# the SAME prompt repeated takes 0.43s. So Ollama's cache is worth 12x and this workload captures
# none of it: the shared prefix is ~120 of ~474 tokens and the bulk -- the per-token candidate
# lines -- is unique to every call by construction. Ordering still costs nothing and helps if a
# future server does partial-prefix reuse.
#
# The lever that DOES work is prompt LENGTH. Trimming candidates 8 -> 3 and tightening the wording
# took 517 tokens / 5.9s to 447 / 4.8s, about 18 %.
PROMPT = """You are an interlinear glossing engine for {lang}.
Return ONLY a JSON array of strings, one gloss per numbered token, in order.
Each gloss is 1-3 lowercase English words giving that token's MEANING in this context.
For punctuation return the mark itself. For a grammatical morpheme with no English word, return the
closest English function word (for example "of", "to", "not", "the").
Where candidates are offered, choose the one that fits; give a better one if none fit.
No explanation, no keys, no markdown fence.

Sentence: {context}
Tokens:
{lines}
Array length: {n}"""

TAB = "\t"
SPLIT = re.compile(r"[-.:=,;/\[\]()<>+~]+|_")


def ask(prompt, timeout=240):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                         "think": False, "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def parse_array(text, n):
    """The JSON array, or None. A wrong LENGTH returns None -- see the docstring warning."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    return [str(x).strip().lower() for x in arr]


def sentences(conllu_glob):
    """(tokens, gold glosses) per sentence, gold being None where the treebank has none."""
    for fn in sorted(glob.glob(conllu_glob)):
        toks, gold = [], []
        for ln in open(fn, encoding="utf-8", errors="replace"):
            if not ln.strip():
                if toks:
                    yield toks, gold
                toks, gold = [], []
                continue
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split(TAB)
            if len(c) < 10 or "-" in c[0] or "." in c[0]:
                continue
            toks.append(c[1])
            m = re.search(r"(?:^|\|)Gloss=([^|]*)", c[9])
            gold.append(m.group(1) if m and m.group(1) not in ("", "_") else None)
        if toks:
            yield toks, gold


def grade(pairs):
    import numpy as np
    from sud_generic_embed_v3 import load_vectors
    T = load_vectors("assets_vec/generic_vec_v3.npz")

    def vec(s):
        vs = []
        for p in (q for part in SPLIT.split(s.replace("_", " ")) for q in part.split()):
            if p.isalpha() and not (p.isupper() and len(p) > 1):
                r = T.row("en", p)
                if r is not None:
                    vs.append(T.V[r])
        if not vs:
            return None
        m = np.mean(vs, 0)
        n = np.linalg.norm(m)
        return m / n if n else None

    exact = sum(1 for g, h in pairs if g.strip() == h.strip().lower())
    gv = [(vec(g), vec(h)) for g, h in pairs]
    ok = [(a, b) for a, b in gv if a is not None and b is not None]
    sims = [float(a @ b) for a, b in ok]
    rng = np.random.default_rng(0)
    hb = [b for _, b in ok]
    perm = rng.permutation(len(hb))
    shuf = [float(ok[i][0] @ hb[perm[i]]) for i in range(len(ok))]
    print(f"\nGRADED against {len(pairs)} human glosses:")
    print(f"  exact string match      {exact / len(pairs):.1%}")
    print(f"  cos(model, human)       {np.mean(sims):+.4f}   over {len(sims)} comparable")
    print(f"  cos, SHUFFLED control   {np.mean(shuf):+.4f}")
    print(f"  beat their shuffled partner: "
          f"{sum(1 for a, b in zip(sims, shuf) if a > b) / max(len(sims), 1):.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--lang-name", required=True, help="English name of the language, for the prompt")
    ap.add_argument("--conllu", required=True, help="glob of source .conllu")
    ap.add_argument("--wiktionary", default=None, help="assets_vec/dict/<lang>-en.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--max-cands", type=int, default=3)
    ap.add_argument("--window", type=int, default=15,
                    help="tokens per call. Failure is almost purely a function of this: 0 %% below "
                         "10, 67 %% at 20-29, 100 %% above 50.")
    ap.add_argument("--grade", action="store_true",
                    help="score against the treebank's own human Gloss= column")
    a = ap.parse_args()

    wik = {}
    if a.wiktionary and os.path.exists(a.wiktionary):
        raw = json.load(open(a.wiktionary, encoding="utf-8"))
        wik = {k: list(v)[:a.max_cands] for k, v in raw.items()}

    cache_path = a.cache or f"caches/llm_gloss_{a.lang}.json"
    pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    out, n_sent, n_fail, pairs = {}, 0, 0, []
    for toks, gold in sentences(a.conllu):
        if a.max_sents and n_sent >= a.max_sents:
            break
        n_sent += 1
        key = "".join(toks)
        if key in cache:
            glosses = cache[key]
        else:
            ctx = " ".join(toks)
            glosses, W = [], a.window
            for s0 in range(0, len(toks), W):
                chunk = toks[s0:s0 + W]
                lines = []
                for i, tk in enumerate(chunk, 1):
                    c = wik.get(tk) or wik.get(tk.lower()) or []
                    lines.append(f"{i}. {tk}" + (f" [{', '.join(c)}]" if c else ""))
                prompt = PROMPT.format(lang=a.lang_name, context=ctx,
                                       lines="\n".join(lines), n=len(chunk))
                got = None
                for _ in range(2):
                    try:
                        got = parse_array(ask(prompt), len(chunk))
                    except Exception:
                        got = None
                    if got:
                        break
                if not got:
                    n_fail += 1
                    got = [" ".join((wik.get(tk) or wik.get(tk.lower()) or [])[:3]) for tk in chunk]
                glosses += got
            cache[key] = glosses
            if n_sent % 25 == 0:
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {n_sent} sentences, {n_fail} windows fell back",
                      file=sys.stderr, flush=True)
        out[key] = glosses
        if a.grade:
            pairs += [(g, h) for g, h in zip(glosses, gold) if h and g]

    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    dest = a.out or f"assets_vec/dict/{a.lang}-llm.json"
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False)
    total = sum(len(v) for v in out.values())
    filled = sum(1 for v in out.values() for g in v if g)
    print(f"{a.lang}: {n_sent} sentences, {total} tokens, {filled / max(total, 1):.1%} glossed, "
          f"{n_fail} WINDOWS fell back to the dictionary")
    print(f"wrote {dest}")

    if a.grade and pairs:
        grade(pairs)


if __name__ == "__main__":
    main()
