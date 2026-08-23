#!/usr/bin/env python3
"""Fill an empty LEMMA/FEATS column with a local LLM — and measure the filling against Tamil gold.

WHY THIS EXISTS. SUD_Telugu-MTG carries no lemma column and 115 FEATS values in 6 465 tokens, so
the Latin/Sanskrit recipe's two parser input channels have nothing to read. The library survey
(`docs/dravidian.md`) found nothing off the shelf: the IIIT-LTRC/anusāraka analyser is a 2001
C-and-Perl package with hardcoded build paths, `apertium-tel` contains six noun roots, Indic NLP
does unsupervised segmentation with no lemma or features, and Stanza's Telugu model is trained on
MTG itself so its lemmatiser returns nothing. The local LLM this project already runs for the
`udep` relabelling is the one remaining route.

⚠ **THE CALIBRATION IS THE POINT, NOT AN EXTRA.** An annotator with no gold to check it against is
a source of confident noise, and `NEGATIVE-RESULTS.md` already records LLM multi-way relabelling
failing here. What makes this defensible is that **Tamil has gold lemmas and gold FEATS for exactly
this task**, in the same family, the same typology and the same annotation scheme. So `--score`
runs the identical prompt over Tamil and reports lemma accuracy and per-feature P/R against gold,
and the Telugu output is only ever quoted with that error rate attached. Run the Tamil calibration
FIRST; if it is poor, the honest outcome is that Telugu keeps its base arm and no `lemvec` arm.

    llm_morph_annotate.py --score --conllu assets_ta/ta_ttb-sud-dev.conllu --lang ta -n 80
    llm_morph_annotate.py --conllu assets_te/te_mtg-sud-train.conllu --lang te \\
        --out assets_te/te_mtg-sud-train.llm.conllu

THE PROMPT GIVES THE MODEL WHAT THE TREEBANK ALREADY KNOWS and asks only for what it does not:
the sentence, its transliteration (both treebanks ship `Translit`), and each token's FORM and gold
UPOS — MTG's UPOS is annotated and there is no reason to make the model re-derive it, or to let it
contradict the tree the parser will be trained on. The feature inventory is CLOSED and named in the
prompt, taken from Tamil's own, so the output lands in a vocabulary the morphologiser can train on
rather than in whatever UD-adjacent strings the model favours today.

Requests are cached on (model, prompt version, sentence) in a JSON file, as `disambiguate_pp.py`
caches its own, so a re-run costs nothing and an interrupted run resumes. Temperature 0, no
thinking, one sentence per request — the Ollama ceiling is ~3 calls/s and parallel requests give no
speedup on this machine (CLAUDE.md), so there is nothing to gain by batching harder.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/generate"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

#: Bump when the prompt changes, so a cache built by the old one is not silently reused.
PROMPT_VERSION = 7

#: The closed inventory the model is allowed to use — Tamil TTB's own, so the calibration transfers.
FEATURE_VALUES = {
    "Case": ["Nom", "Acc", "Dat", "Gen", "Loc", "Ins", "Com", "Abl", "Ben"],
    "Number": ["Sing", "Plur"],
    "Gender": ["Masc", "Fem", "Neut", "Com"],
    "Person": ["1", "2", "3"],
    "Polite": ["Form"],
    "Tense": ["Past", "Pres", "Fut"],
    "VerbForm": ["Fin", "Inf", "Part", "Ger", "Conv"],
    "Mood": ["Ind", "Imp", "Cnd", "Pot"],
    "Voice": ["Act", "Pass"],
    "Polarity": ["Pos", "Neg"],
    "PronType": ["Prs", "Dem", "Int", "Ind", "Rel"],
    "Animacy": ["Anim", "Inan"],
    "NumType": ["Card", "Ord"],
    "AdpType": ["Post"],
}
LANG_NAME = {"ta": "Tamil", "te": "Telugu", "ml": "Malayalam"}


def read_sentences(path):
    """Yield (comments, [cols, ...]) per sentence, keeping non-word rows in place."""
    comments, rows = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if rows:
                yield comments, rows
            comments, rows = [], []
            continue
        if line.startswith("#"):
            comments.append(line)
        else:
            rows.append(line.split("\t"))
    if rows:
        yield comments, rows


def words(rows):
    return [c for c in rows if len(c) == 10 and c[0].isdigit()]


def translit_of(comments, rows):
    for line in comments:
        if line.startswith("# translit ="):
            return line.split("=", 1)[1].strip()
    out = []
    for c in words(rows):
        for item in c[9].split("|"):
            if item.startswith("Translit="):
                out.append(item.split("=", 1)[1])
                break
    return " ".join(out)


def surface(comments, rows) -> str:
    """The sentence as a READER would see it — the treebank's own `# text`.

    ⚠ NOT `" ".join(FORM)`. Where a treebank has multiword tokens the FORMs are its SYNTACTIC
    words, and joining them with spaces produces a string that is not the language: Tamil TTB
    splits 9.67 % of its orthographic words, so `துறைகளையும்` came out as `துறைகளைய் உம்` and
    `வந்துள்ளதாக` as `வந்த் உள்ளத் ஆக`. That mangled string was being shown to the model as
    context, in the twelve exemplars as well as the target. `# text` is the real surface, and the
    per-token listing below still carries the syntactic words, which is the pairing an annotator
    actually works from.
    """
    for line in comments:
        if line.startswith("# text ="):
            return line.split("=", 1)[1].strip()
    out = ""
    for c in words(rows):
        out += c[1] + ("" if "SpaceAfter=No" in c[9] else " ")
    return out.strip()


def render_block(comments, rows, gold: bool) -> str:
    """One sentence as prompt material. `gold=True` shows the answer, `False` asks for it."""
    ws = words(rows)
    head = f"Sentence: {surface(comments, rows)}\n"
    trans = translit_of(comments, rows)
    if trans:
        head += f"Transliteration: {trans}\n"
    # ⚠ AN EXEMPLAR MUST BE SHAPED EXACTLY LIKE THE ANSWER, and the first version was not.
    # It showed five columns (INDEX FORM UPOS LEMMA FEATS) while the instructions asked for three
    # (INDEX LEMMA FEATS). The model imitated the EXEMPLARS -- emitting `FORM<TAB>UPOS<TAB>FEATS`
    # and dropping the index -- so every line failed a parser that keys on a leading digit and fell
    # back to the identity lemma with no features. That is what the "12-shot is worse" result
    # actually was: 18 of 40 sentences produced ZERO usable lines. So an exemplar is rendered as an
    # input block and an answer block, each in the shape its counterpart will really have.
    body = "\n".join(f"{i + 1}\t{c[1]}\t{c[3]}" for i, c in enumerate(ws))
    if gold:
        body += "\nANSWER:\n" + "\n".join(
            f"{i + 1}\t{c[2]}\t{c[5]}" for i, c in enumerate(ws))
    return head + body + "\n"


def load_shots(path, n, min_len=4, max_len=12):
    """`n` fully-annotated example sentences, chosen DETERMINISTICALLY from a gold treebank.

    ⚠ THE EXEMPLARS MUST NOT COME FROM THE SCORED SPLIT. They are drawn from `train` and the
    calibration is scored on `dev`, so a high few-shot number is generalisation and not recall of a
    sentence the model was just shown.

    Chosen by length rather than at random: a two-token sentence teaches nothing about a FEATS
    bundle and a thirty-token one crowds the prefix, and a deterministic choice keeps the prompt --
    and therefore the cache key -- stable across runs.
    """
    out = []
    for comments, rows in read_sentences(path):
        ws = words(rows)
        if not (min_len <= len(ws) <= max_len):
            continue
        if sum(1 for c in ws if c[5] != "_") < len(ws) // 2:
            continue                     # an exemplar has to actually demonstrate FEATS
        out.append((comments, rows))
        if len(out) >= n:
            break
    return out


def build_prompt(lang: str, comments, rows, shots=()) -> str:
    """⚠ EVERY STATIC PART COMES FIRST AND THE SENTENCE LAST.

    That ordering is not cosmetic. Ollama caches the KV prefix, so an instruction block, a feature
    inventory and twelve exemplars that are byte-identical across requests are computed once and
    reused; put the sentence at the top, as the first version of this prompt did, and every request
    re-computes the whole prefix. On a 12-shot prompt that is most of the work. The same structure
    is what `disambiguate_pp.py`'s prompts use, for the same reason.
    """
    inventory = "\n".join(f"  {k}: {', '.join(v)}" for k, v in FEATURE_VALUES.items())
    name = LANG_NAME.get(lang, lang)
    parts = [
        f"You are annotating {name} for a Universal Dependencies treebank.\n\n"
        f"For each token you are given its index, its surface form and its part of speech. Give "
        f"the LEMMA (the dictionary/citation form, in {name} script) and the MORPHOLOGICAL "
        f"FEATURES.\n\n"
        f"Use ONLY these features and values:\n{inventory}\n\n"
        f"Rules:\n"
        f"- Output exactly one line per token, in the same order, as: INDEX<TAB>LEMMA<TAB>FEATS\n"
        f"- Write the feature NAME exactly as listed above. The name for case is `Case`, so an "
        f"oblique noun is `Case=Gen`, never `Gen=...`.\n"
        f"- FEATS is Name=Value pairs joined by | , or _ if none apply.\n"
        f"- BE COMPLETE. A finite verb states Mood, Number, Person, Polarity, Tense, VerbForm and "
        f"Voice; a noun states Case, Gender, Number and Person. Omitting a feature that applies is "
        f"an error, not caution.\n"
        f"- THE LEMMA IS NOT THE FORM. Strip every case, number, tense, person and politeness "
        f"ending. A noun lemma is the bare nominative singular stem; a verb lemma is the bare "
        f"root.\n"
        f"- Only for PUNCT, SCONJ, CCONJ, PART, INTJ and uninflected ADV is the lemma equal to "
        f"the form.\n"
        f"- Do not explain, do not add a header, do not renumber. Output only the lines.\n"
    ]
    if shots:
        parts.append(f"\nHere are {len(shots)} correctly annotated examples. The last two columns "
                     f"are the LEMMA and the FEATS you must produce.\n")
        for i, (scom, srow) in enumerate(shots, 1):
            parts.append(f"\n--- example {i} ---\n" + render_block(scom, srow, gold=True))
    else:
        parts.append("\nFormat example (the language is not %s; it shows the SHAPE of an answer "
                     "only):\n1\tdomus\tCase=Dat|Number=Sing\n"
                     "2\tvenio\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\n"
                     "3\t.\t_\n" % name)
    parts.append("\n--- now annotate this one ---\n"
                 + render_block(comments, rows, gold=False) + "ANSWER:\n")
    return "".join(parts)


#: ⚠ OLLAMA'S DEFAULT `num_ctx` IS 4096, AND A 12-SHOT PROMPT DOES NOT FIT IN IT.
#: Measured: the 12-exemplar Tamil prompt is `prompt_eval_count` = 4 147 tokens. Ollama does not
#: refuse an over-long prompt -- it evicts the FRONT of it, which is where the instructions and the
#: feature inventory live, and then generates from what is left. The result is not an error but a
#: quietly worse answer: the first scored run of the 12-shot condition returned only 434 answer
#: lines for 685 gold tokens and answered just 22 of 40 sentences in full, which read as the model
#: "becoming conservative" (precision 0.882, recall 0.188) when it was really being truncated.
#: Any comparison between a short prompt and a long one is meaningless unless this is set.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))


def query(prompt: str, model: str) -> str:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "think": False,
                         "options": {"temperature": 0, "num_ctx": NUM_CTX,
                                     "num_predict": NUM_PREDICT}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["response"].strip()


def parse_response(raw: str, n: int, forms: list[str]) -> list[tuple[str, str]]:
    """(lemma, feats) per token. Anything unparseable falls back to IDENTITY lemma and no feats.

    Never to `_` as a lemma: spaCy keeps CoNLL-U `_` as a LITERAL string, which is the trap
    `scripts/prep_te.py` exists to avoid. A refusal here must degrade to the same identity fallback.

    TWO PASSES, and the second is not leniency for its own sake. The strict pass keys each line on
    a leading INDEX, which is what the prompt asks for. But a model given twelve exemplars will
    sometimes drop the index and emit `LEMMA<TAB>FEATS`, one line per token, in the right order —
    correct content in an unrequested shape. Measured on the 12-shot Tamil run that was **15 of 40
    sentences**, 239 tokens, all discarded and silently replaced by the identity fallback, which
    depressed recall by roughly a third and would have been read as the model failing.

    The positional pass therefore fires ONLY when alignment is unambiguous: the strict pass found
    almost nothing, and the number of candidate lines EXACTLY equals the number of tokens. Anything
    else is left unparsed and counted, because a misaligned rescue writes one token's morphology
    onto another and nothing downstream would ever notice.
    """
    out: dict[int, tuple[str, str]] = {}
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            parts = [p for p in line.strip().split(None, 2) if p]
        if len(parts) < 3 or not parts[0].strip().rstrip(".").isdigit():
            continue
        idx = int(parts[0].strip().rstrip("."))
        lemma, feats = parts[1].strip(), parts[2].strip()
        if 1 <= idx <= n and lemma:
            out[idx] = (lemma, normalise_feats(feats))

    if len(out) < n // 2:
        rows = [ln.strip().split("\t") for ln in raw.splitlines() if ln.strip()]
        two = [r for r in rows if len(r) == 2]
        if len(two) == n and len(rows) == n:
            for i, (lemma, feats) in enumerate(two, 1):
                if lemma.strip():
                    out.setdefault(i, (lemma.strip(), normalise_feats(feats)))

    # The caller needs to know which tokens were really answered, not how many lines happened to
    # parse on the first pass: after the positional rescue those are different numbers, and the
    # first version of the diagnostic reported 239 tokens "unanswered" in a run whose recall was
    # 0.700 — a counter contradicting the result it sits above.
    return ([out.get(i + 1, (forms[i], "_")) for i in range(n)], set(out))


def normalise_feats(feats: str) -> str:
    """Keep only pairs from the closed inventory, alphabetised. Anything else is dropped."""
    if not feats or feats in ("_", "-", "None", "none"):
        return "_"
    keep = []
    for item in feats.split("|"):
        if "=" not in item:
            continue
        key, value = (p.strip() for p in item.split("=", 1))
        if key in FEATURE_VALUES and value in FEATURE_VALUES[key]:
            keep.append(f"{key}={value}")
    return "|".join(sorted(set(keep))) or "_"


def load_cache(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--lang", required=True, choices=sorted(LANG_NAME))
    ap.add_argument("--out")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cache", default=None)
    ap.add_argument("-n", type=int, default=0, help="limit to the first N sentences")
    ap.add_argument("--shots", type=int, default=12,
                    help="few-shot exemplars, drawn from --shots-from. 12 matches the "
                         "fewshot12 setting that won the udep prompt benchmark. 0 is "
                         "the zero-shot control.")
    ap.add_argument("--shots-from", default="assets_ta/ta_ttb-sud-train.conllu",
                    help="gold treebank the exemplars come from. MUST be a different "
                         "split from the one being scored.")
    ap.add_argument("--score", action="store_true",
                    help="compare against the file's OWN gold LEMMA/FEATS (Tamil calibration)")
    args = ap.parse_args()

    pathlib.Path("caches").mkdir(exist_ok=True)
    cache_path = args.cache or f"caches/cache_llm_morph_{args.lang}.json"
    cache = load_cache(cache_path)
    shots = load_shots(args.shots_from, args.shots) if args.shots else []
    if shots:
        print(f"{len(shots)} exemplars from {args.shots_from}", file=sys.stderr)
    sents = list(read_sentences(args.conllu))
    if args.n:
        sents = sents[:args.n]

    lem_ok = lem_n = 0
    feat_tp = collections.Counter()
    feat_fp = collections.Counter()
    feat_fn = collections.Counter()
    out_lines: list[str] = []
    dirty = short = short_tokens = malformed = rescued = 0

    for s, (comments, rows) in enumerate(sents):
        ws = words(rows)
        forms = [c[1] for c in ws]
        prompt = build_prompt(args.lang, comments, rows, shots)
        key = f"{args.model}|v{PROMPT_VERSION}|shots={len(shots)}|{' '.join(forms)}"
        if key not in cache:
            try:
                cache[key] = query(prompt, args.model)
            except Exception as exc:                      # a dead request must not lose the run
                print(f"  sentence {s}: {type(exc).__name__}: {exc}", file=sys.stderr)
                cache[key] = ""
            dirty += 1
            if dirty % 5 == 0:
                pathlib.Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False),
                                                    encoding="utf-8")
                print(f"  {s + 1}/{len(sents)} sentences", file=sys.stderr)
        pred, filled = parse_response(cache[key], len(ws), forms)
        lines = [ln for ln in cache[key].splitlines() if ln.strip()]
        strict = sum(1 for ln in lines
                     if ln.strip().split("\t")[0].strip().rstrip(".").isdigit())
        if len(filled) < len(ws):
            # A response with plenty of lines but none the parser accepts is a FORMAT deviation,
            # not a short answer, and conflating them sent an earlier investigation to `num_ctx`.
            if not filled and len(lines) >= len(ws):
                malformed += 1
            else:
                short += 1
            short_tokens += len(ws) - len(filled)
        rescued += len(filled) - strict if len(filled) > strict else 0

        if args.score:
            for cols, (lemma, feats) in zip(ws, pred):
                gold_lemma, gold_feats = cols[2], cols[5]
                if gold_lemma != "_":
                    lem_n += 1
                    lem_ok += lemma == gold_lemma
                g = {i for i in (gold_feats.split("|") if gold_feats != "_" else []) if "=" in i}
                p = set(feats.split("|")) if feats != "_" else set()
                g = {i for i in g if i.split("=")[0] in FEATURE_VALUES}
                for item in g & p:
                    feat_tp[item.split("=")[0]] += 1
                for item in p - g:
                    feat_fp[item.split("=")[0]] += 1
                for item in g - p:
                    feat_fn[item.split("=")[0]] += 1
        else:
            it = iter(pred)
            out_lines.extend(comments)
            for cols in rows:
                cols = list(cols)
                if len(cols) == 10 and cols[0].isdigit():
                    lemma, feats = next(it)
                    cols[2] = lemma or cols[1]
                    if cols[5] == "_":
                        cols[5] = feats
                out_lines.append("\t".join(cols))
            out_lines.append("")

    pathlib.Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    if args.score:
        print(f"{args.conllu}: {len(sents)} sentences, model {args.model}, "
              f"shots={len(shots)}, num_ctx={NUM_CTX}")
        # ⚠ Reported ALWAYS, not only when nonzero. A short answer is absorbed by the identity
        # fallback in `parse_response`, so without this line a truncated run is indistinguishable
        # from a model that simply declined to give features.
        print(f"  UNANSWERED   {short_tokens} tokens: {short}/{len(sents)} sentences SHORT, "
              f"{malformed}/{len(sents)} MALFORMED; {rescued} tokens recovered by the "
              f"positional pass")
        print(f"  LEMMA exact  {lem_ok}/{lem_n} = {lem_ok / max(lem_n, 1):.2%}")
        print(f"  {'feature':12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>7s} {'R':>7s} {'F':>7s}")
        keys = sorted(set(feat_tp) | set(feat_fp) | set(feat_fn),
                      key=lambda k: -(feat_tp[k] + feat_fn[k]))
        for k in keys:
            tp, fp, fn = feat_tp[k], feat_fp[k], feat_fn[k]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            print(f"  {k:12s} {tp:6d} {fp:6d} {fn:6d} {p:7.3f} {r:7.3f} {f:7.3f}")
        tp, fp, fn = sum(feat_tp.values()), sum(feat_fp.values()), sum(feat_fn.values())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {'MICRO':12s} {tp:6d} {fp:6d} {fn:6d} {p:7.3f} {r:7.3f} "
              f"{2 * p * r / (p + r) if p + r else 0:7.3f}")
    elif args.out:
        pathlib.Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
