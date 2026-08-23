#!/usr/bin/env python3
"""LLM pass over every `udep` the rules could not decide — a CONSTRAINED multi-way choice.

The comp/mod pipeline asks one binary question, and `apply_udep_rules.py` then commits the 10 730
residual tokens whose label the treebank settles by majority. What is left (32 415) is neither: the
correct relation may be `subj`, `comp:obj`, `cc`, `mod@relcl`, `discourse` — anything in the SUD
inventory — and it genuinely varies within one signature. Arabic ADJ <- NOUN is the clearest case:
accusative dependents split `subj` 59 % / `mod` 15 % / `comp:obj` 14 %, so no rule can fire and only
reading the sentence decides.

Two design choices carry the work:

**The label set is CONSTRAINED by the treebank, not by the model's imagination.** For each token the
candidates are exactly the relations the annotators used for the same (head UPOS, dep UPOS, dep
lemma) signature elsewhere. That keeps the model inside the analysis this treebank actually adopts —
it cannot invent `obl` for a corpus that writes `comp:obl`, or pick a relation the construction never
takes. 79 % of the residue has such candidates; the other 21 % has NO committed precedent at all and
is skipped rather than guessed at, since there would be nothing to constrain or validate against.

**The answer is a DIGIT, not a label.** SUD relations contain colons and `@` subtypes
(`comp:obl@agent`, `discourse@sp`), which `disambiguate_pp.query` — normalising to the first
whitespace-delimited token — would mangle. Numbering the options sidesteps that entirely.

Prompt shape follows the project convention: a long static prefix (task + relation glosses) and a
short variable suffix (the sentence and the options), so Ollama reuses the cached prefix KV.
Resumable: every decision is flushed to `caches/relabel_cache_residue_<lang>.jsonl`.

    relabel_residue.py --lang ar --limit 200      # try a slice first
    relabel_residue.py --all
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import time

_HERE = pathlib.Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("aud", _HERE / "udep_residue_audit.py")
aud = importlib.util.module_from_spec(_s)
_s.loader.exec_module(aud)
d = aud.d

GLOSS = {
    "subj": "the subject of the head",
    "comp:obj": "a direct object / core complement of the head",
    "comp:obl": "an oblique complement the head lexically selects (required by the verb)",
    "comp:pred": "a predicative complement (after a copula or similar)",
    "comp:aux": "part of an auxiliary/verbal chain headed by the dependent",
    "mod": "an optional modifier or adjunct (could be removed without breaking the sentence)",
    "mod@relcl": "a relative clause modifying the head",
    "mod@poss": "a possessor of the head",
    "cc": "a coordinating conjunction linking the head to a following conjunct",
    "conj:coord": "a coordinated element (X and Y)",
    "conj:appos": "an apposition (a renaming of the head)",
    "det": "a determiner of the head",
    "discourse": "a discourse marker or interjection, outside the clause's argument structure",
    "flat": "part of a flat multi-word name or fixed sequence",
    "compound": "part of a compound with the head",
    "unk": "no independent grammatical relation of its own",
    "punct": "punctuation",
}

PREFIX = """You are annotating a dependency treebank in Surface-Syntactic Universal Dependencies \
(SUD). SUD uses functional heads: adpositions, subordinators and auxiliaries head the phrases they \
introduce, and relations are named by FUNCTION, not by part of speech.

You will see a sentence, a HEAD word, and a DEPENDENT word that attaches to it. The relation \
between them is currently unlabelled. Choose which numbered relation the dependent bears to the \
head. The options are exactly the relations this treebank uses for this construction elsewhere.

Answer with the NUMBER ONLY. No explanation, no punctuation, no other text.

"""


def build_prompt(sentence, head_form, dep_form, dep_span, options, shots=()):
    lines = []
    if shots:
        lines.append("Examples of each relation, taken from this same treebank:")
        for lab, sent, hd, dp in shots:
            lines.append(f"  [{lab}]  {sent}")
            lines.append(f"      HEAD {hd}  <-  DEPENDENT {dp}")
        lines.append("")
        lines.append("Now the case to decide:")
    lines += [f"Sentence: {sentence}",
             f"HEAD: {head_form}",
             f"DEPENDENT: {dep_form}"]
    if dep_span and dep_span != dep_form:
        lines.append(f"DEPENDENT's full phrase: {dep_span}")
    lines.append("")
    lines.append("Which relation does the DEPENDENT bear to the HEAD?")
    for i, lab in enumerate(options, 1):
        if lab == ABSTAIN:
            lines.append(f"  {i}. none of the above — the correct relation is not listed, "
                         f"or it is genuinely unclear")
            continue
        g = GLOSS.get(lab) or GLOSS.get(lab.split("@")[0].split(":")[0], "")
        lines.append(f"  {i}. {lab}" + (f" — {g}" if g else ""))
    lines.append("")
    lines.append("Number:")
    return PREFIX + "\n".join(lines)


ABSTAIN = "__ABSTAIN__"


def collect(path, max_options, min_options=3):
    """Residual udep tokens plus the candidate labels the treebank licenses for each.

    Candidates come from a TWO-LEVEL signature, and the back-off is not cosmetic. Keying only on
    (head POS, dep POS, dep LEMMA) is precise but starves rare signatures: inspection of the first
    pass found the linguistically correct label MISSING from the options in four languages out of
    seven — Persian `صبح چهارشنبه` "Wednesday morning" under `افزود` "added" was offered
    comp:obj/subj/comp:pred/compound@lvc with no `mod`, and duly came out `compound@lvc`. With only
    two or three options and no way to decline, a generically-plausible oblique absorbs all the
    uncertainty, which is most of why `comp:obl` took 45 % of the first pass.

    So when the lemma-specific level yields fewer than `min_options`, the coarser (head POS, dep POS)
    level tops it up — that level offers `mod` in every one of those failing cases. Lemma-specific
    labels stay first, since they are the more specific evidence.
    """
    committed = collections.defaultdict(collections.Counter)
    coarse = collections.defaultdict(collections.Counter)
    # one real committed instance per (coarse signature, label), for CONTRASTIVE examples.
    # The comp/mod pipeline established that few-shot beats definitions alone (`fewshot12_def` is
    # the canonical en prompt, and curated contrastive sets were built for zh 在/于 and id di/pada);
    # this residue prompt had NO examples, only my English glosses. Harvesting them from the
    # treebank needs no curation and shows the model what each label means IN THIS TREEBANK rather
    # than in my paraphrase.
    exemplar = {}
    rows = []
    for sid, toks in d.parse_conllu(path):
        by = {t["id"]: t for t in toks}
        for t in toks:
            h = by.get(t["head"])
            if h is None:
                continue
            sig = aud.signature(h, t)
            if aud.is_udep(t["deprel"]):
                rows.append((sid, t, h, toks, by, sig))
            else:
                committed[sig][t["deprel"]] += 1
                ckey = (h["upos"], t["upos"])
                coarse[ckey][t["deprel"]] += 1
                ek = (ckey, t["deprel"])
                if ek not in exemplar and 3 <= len(toks) <= 40:
                    joiner = "" if all(not x["space_after"] for x in toks[:-1]) else " "
                    exemplar[ek] = (joiner.join(x["form"] for x in toks)[:150],
                                    h["form"], t["form"])
    out = []
    for sid, t, h, toks, by, sig in rows:
        dist = committed.get(sig)
        if not dist:
            continue                      # no precedent: nothing to constrain a choice with
        # Every label the annotators ever used for this signature, most frequent first — NO
        # minimum-count filter. Requiring >= 2 occurrences looked like sensible noise control and
        # was not: for Arabic `اليوم` "today" under `تؤكد` "confirm" it left only comp:obj and subj
        # as options, dropping `mod` — the temporal adjunct reading that is almost certainly right.
        # A rare committed label is exactly the evidence that a construction ALSO takes that
        # relation, which is the whole point of offering a choice.
        # RESERVE slots for the coarse level rather than topping up only when the lemma level is
        # thin. A threshold is the wrong mechanism: the Persian case that failed had FOUR
        # lemma-specific options (compound@lvc/subj/comp:obj/comp:pred) and so never triggered a
        # back-off, yet `mod` — the right answer for a temporal adjunct — was still absent. Keeping
        # two slots for the coarser (head POS, dep POS) distribution guarantees the construction's
        # common relations are always reachable, whatever the lemma happens to attest.
        keep = max(2, max_options - 2)
        opts = [lab for lab, _ in dist.most_common(keep)]
        for lab, _ in coarse.get((sig[0], sig[1]), collections.Counter()).most_common():
            if len(opts) >= max_options:
                break
            if lab not in opts:
                opts.append(lab)
        # `mod` and `comp:obl` are ALWAYS offered, attested or not. Checking the data killed an
        # assumption of mine: the coarse level does not rescue these cases either, because the
        # residue is residual PRECISELY where a treebank systematically declines to commit. fa
        # VERB->NOUN is compound@lvc 61 % / subj 23 % / comp:obj 11 % with `mod` absent entirely;
        # ar and lzh are the same shape. So a temporal adjunct like `صبح چهارشنبه` "Wednesday
        # morning" has no attested label to be given, and constraining to attestation guarantees a
        # wrong answer. mod-vs-comp is the macro-distinction SUD is built on and is grammatically
        # available to any dependent, so offering both invents nothing.
        for lab in ("mod", "comp:obl"):
            if lab not in opts:
                opts.append(lab)
        if len(opts) < 2:
            continue                      # a single candidate is a rule, not a question
        sub = sorted(d.descendants(toks, t["id"]))
        joiner = "" if all(not x["space_after"] for x in toks[:-1]) else " "
        span = joiner.join(x["form"] for x in toks if x["id"] in sub)
        sent = joiner.join(x["form"] for x in toks)
        opts = opts + [ABSTAIN]           # always let the model decline rather than guess
        shots = []
        for lab in opts:
            ex = exemplar.get(((sig[0], sig[1]), lab))
            if ex:
                shots.append((lab, ex[0], ex[1], ex[2]))
        out.append({"sent_id": sid, "tid": t["id"], "sig": "|".join(sig), "shots": shots,
                    "head": h["form"], "dep": t["form"], "span": span[:200],
                    "sentence": sent[:400], "options": opts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-options", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    langs = list(aud.TREEBANKS) if a.all else [a.lang]
    for lang in langs:
        src = pathlib.Path(str(aud.TREEBANKS[lang]).replace(".conllu", ".udep_ruled.conllu"))
        if not src.exists():
            src = pathlib.Path(aud.TREEBANKS[lang])
        if not src.exists():
            print(f"  {lang}: no treebank"); continue
        items = collect(src, a.max_options)
        if a.limit:
            items = items[:a.limit]
        pathlib.Path("caches").mkdir(exist_ok=True)
        cache_path = pathlib.Path(f"caches/relabel_cache_residue_{lang}.jsonl")
        cache = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                try:
                    r = json.loads(line); cache[(r["sent_id"], r["tid"])] = r["label"]
                except Exception:
                    pass
        todo = [it for it in items if (it["sent_id"], it["tid"]) not in cache]
        print(f"  {lang}: {len(items)} decidable, {len(cache)} cached, {len(todo)} to query")
        if a.dry_run:
            if todo:
                it = todo[0]
                print("  --- sample prompt ---")
                print(build_prompt(it["sentence"], it["head"], it["dep"], it["span"],
                                   it["options"])[-600:])
            continue
        t0 = time.time(); done = collections.Counter()
        with cache_path.open("a", encoding="utf-8") as fh:
            for i, it in enumerate(todo, 1):
                prompt = build_prompt(it["sentence"], it["head"], it["dep"], it["span"],
                                      it["options"], it.get("shots", ()))
                try:
                    ans = d.query(prompt)
                except Exception:
                    continue
                digits = "".join(ch for ch in ans if ch.isdigit())
                idx = int(digits) - 1 if digits else -1
                label = it["options"][idx] if 0 <= idx < len(it["options"]) else ABSTAIN
                if label == ABSTAIN:
                    label = None          # stays `udep`; an honest non-answer beats a forced one
                done[label or "(abstained)"] += 1
                fh.write(json.dumps({"sent_id": it["sent_id"], "tid": it["tid"],
                                     "sig": it["sig"], "label": label}, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 250 == 0:
                    r = i / max(time.time() - t0, 1e-9)
                    print(f"    {i}/{len(todo)}  {r:.1f}/s  eta {(len(todo)-i)/max(r,1e-9)/60:.0f} min",
                          flush=True)
        print(f"    {lang} done: {dict(done.most_common(6))}")


if __name__ == "__main__":
    main()
