#!/usr/bin/env python3
"""Bootstrap gold for SUD's `Reported=Yes` MISC feature, which no treebank here annotates.

`Reported=Yes` marks reported speech governed as an ordinary complement. It supersedes an older
`parataxis:obj` / `parataxis@rep` analysis, and that history is the key to what it picks out: the
paratactic analysis existed for DIRECT speech, which is quoted verbatim rather than syntactically
integrated. So the target is a complement of a speech/writing verb that is direct speech.

There is no gold to learn from -- `Reported` occurs zero times in every treebank in this project,
and the deprel form `@reported`/`@rep` amounts to 8 Latin tokens -- so the class is built the way
`lang_gold.py` builds the comp/mod benchmark: rules commit the confident cases, the LLM adjudicates
the genuinely ambiguous residue. Precedent for synthesising a class from scratch is the Japanese
`comp:obl` relabel, which went from F 0.000 to 0.720.

DIRECT-SPEECH EVIDENCE (the useful part). Two independent signals, and which one works is a
property of the language's punctuation habits, not of the phenomenon:

  * quotation marks in the complement's subtree -- the obvious test, and the only one that fires
    in Arabic (712/2297 candidates) and Persian (99/1606);
  * a `discourse` dependent inside the complement. This is the discriminating test, not a
    quotative marker: only verbatim speech can host a discourse marker, because an indirect clause
    is reported from the narrator's viewpoint and cannot carry the speaker's interjections. It is
    what makes Latin and Sanskrit tractable at all -- neither uses quotation marks (0 quoted
    candidates in both), and the markers found are exactly the expected ones: en `no`/`well`/
    `yes`/`yep`, la `autem`/`quidem`/`uero` inside a complement of `dico`, sa `vai`/`eva`/`hi`.
  * Sanskrit additionally has `iti`, the quotative particle that closes a verbatim quote (908
    candidates) -- the functional equivalent of a closing quotation mark.

INDIRECT EVIDENCE, which commits the negative: an overt complementiser (en `that`, fa `که`,
ar `أن`/`إن`), or Latin's accusative-and-infinitive, where the complement verb is an infinitive.
A clause with those and no direct-speech evidence is reported indirectly and gets no feature.

Everything else -- a speech-verb complement with neither kind of evidence -- goes to the model.

    OLLAMA_MODEL=gemma4:latest scripts/sud_reported_gold.py --lang ar
    scripts/sud_reported_gold.py --lang en --no-model     # rules only, for a quick look

Writes `*.reported.conllu` next to the source (baselines untouched) and a resumable decision cache
`caches/relabel_cache_reported_<lang>.jsonl`, flushed per decision as in `lang_relabel.py`.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_spec = importlib.util.spec_from_file_location("d", str(_HERE / "disambiguate_pp.py"))
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

# Source files per language, mirroring what the released arm trains on.
FILES = {
    "en": ["assets/en_ewt-sud-%s.relabeled_ext.conllu" % s for s in ("train", "dev", "test")],
    # en_gum: the second English arm (EWT + the ten non-NonCommercial GUM genres).
    # A separate key, not a replacement -- `en` must keep pointing at the EWT-only files
    # that the released CC BY-SA en_sud_ewt wheel was built from.
    "en_gum": ["assets/en_ewtgum-sud-%s.relabeled_ext.conllu" % s for s in ("train", "dev", "test")],
    "ar": ["assets_ar/SUD_Arabic-PADT/ar_padt-sud-%s.relabeled_ext.conllu" % s for s in ("train", "dev", "test")],
    "fa": ["assets_fa/SUD_Persian-PerDT/fa_perdt-sud-%s.relabeled_ext.conllu" % s for s in ("train", "dev", "test")],
    # Latin's released arm trains on the plain ∪ macron union, so the macron half needs the same
    # annotation. Macronisation rewrites FORM only, leaving LEMMA/DEPREL untouched, so every rule
    # fires identically and the cache keys (sent_id|comp_id) match exactly -- it costs no queries.
    "la": ["assets_la/la_ittbproiel-sud-%s.relabeled_ext.conllu" % s for s in ("train", "dev", "test")]
          + ["assets_la/la_ittbproiel-sud-%s.relabeled_ext.macron.conllu" % s for s in ("train", "dev", "test")],
    # ⚠ csl_mwt, not csl_rev. `corpus_sa_csl_rev` is the SUPERSEDED pausa-normalised
    # representation (CLAUDE.md, "Superseded but kept") -- unrelabelled and on a tokenisation the
    # released sa arm does not share. THE `.reported.` GOLD MUST BE REGENERATED: only
    # `*.csl_rev.reported.conllu` exists on disk, so re-run this script for sa before
    # `eval_sud_reported.py sa` will find its input.
    "sa": ["corpus_sa_mwt_rl2/train.csl_mwt.conllu"]
          + ["assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-%s.relabeled_ext.csl_mwt.conllu"
             % s for s in ("dev", "test")],
}

# Lexicons and character classes live in sud_reported_data.py, shared with the runtime
# component sud_reported_rule.py so the two can never drift apart.
_dspec = importlib.util.spec_from_file_location("sud_reported_data",
                                               str(_HERE / "sud_reported_data.py"))
_data = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_data)

base_lang = _data.base_lang
SPEECH_VERBS = _data.SPEECH_VERBS
is_speech_lemma = _data.is_speech_lemma
COMPLEMENTISERS = _data.COMPLEMENTISERS
LA_INTERROGATIVE = _data.LA_INTERROGATIVE
QUOTES = _data.QUOTES
SA_QUOTATIVE = _data.SA_QUOTATIVE


def subtree(tokens, root_id):
    return d.descendants(tokens, root_id)


def _feat(feats, key):
    for item in feats.split("|"):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return None


def is_complementiser(lang, tok):
    return (tok["lemma"] in COMPLEMENTISERS.get(base_lang(lang), set())
            and tok["upos"] in ("SCONJ", "ADP", "PART"))


def la_finite_direct(by_id, ids, comp):
    """Latin: a FINITE complement with no subordinator is direct speech.

    Latin reports statements indirectly with the accusative-and-infinitive, and every finite
    indirect clause -- indirect question, `quod`/`ut` clause -- carries an overt subordinator,
    which under SUD's functional-head analysis IS the complement token. So a finite complement
    that is not itself a subordinator has no way to be indirect. Confirmed on train: 219 such
    cases, governed by `dico` (148), `scribo` (17), `loquor` (13), `interrogo` (9), `narro` (8),
    and the examples are unambiguous verbatim quotation -- `dicit , meditatus sum in omnibus
    operibus tuis`, `dixit , fiat lux`.

    The ONE exception is the indirect question, which is finite and subordinator-less here because
    the interrogative word is not the clause head. It requires the SUBJUNCTIVE, so mood does the
    work: an indicative clause containing `qui` has a relative pronoun, not an interrogative (75
    such cases), and a subjunctive with no interrogative is a jussive inside a quote (`fiat lux`,
    6 cases). Only subjunctive + interrogative (39) is withheld -- and withheld to the model, not
    committed as indirect, since `qui` is ambiguous enough that some of those are relatives too
    (`dixisset , qui non uult operari , non manducet`).
    """
    if _feat(comp["feats"], "VerbForm") != "Fin":
        return False
    if _feat(comp["feats"], "Mood") != "Sub":
        return True
    return not any(by_id[i]["lemma"] in LA_INTERROGATIVE
                   and by_id[i]["upos"] in ("PRON", "DET", "ADV")
                   for i in ids if i in by_id)


def direct_evidence(lang, by_id, ids, comp):
    """Reasons to call the complement DIRECT speech; empty means no positive evidence."""
    found = []
    toks = [by_id[i] for i in sorted(ids) if i in by_id]
    if any(any(c in QUOTES for c in t["form"]) for t in toks):
        found.append("quote")
    # Only verbatim speech can carry the speaker's own discourse markers: an indirect clause is
    # recast from the narrator's viewpoint and cannot host the original speaker's interjections.
    if any(t["deprel"].split("@")[0] == "discourse" for t in toks):
        found.append("discourse")
    if lang == "sa" and any(t["lemma"] in SA_QUOTATIVE for t in toks):
        found.append("iti")
    if lang == "la" and not is_complementiser(lang, comp) and la_finite_direct(by_id, ids, comp):
        found.append("finite-no-subordinator")
    return found


def indirect_evidence(lang, by_id, ids, comp):
    """Reasons to call it INDIRECT speech."""
    found = []
    # Test the COMPLEMENT token, not the subtree: SUD makes the subordinator the head of the
    # clause it introduces, so a complementiser anywhere else belongs to an embedded clause --
    # which inside a verbatim quote is not evidence of anything.
    if is_complementiser(lang, comp):
        found.append("complementiser")
    # Latin's accusative-and-infinitive: the standard indirect-statement construction.
    if lang == "la" and _feat(comp["feats"], "VerbForm") == "Inf":
        found.append("aci")
    return found


def candidates(lang, path):
    """Speech-verb complements, with whatever direct/indirect evidence each carries."""
    out = []
    verbs = SPEECH_VERBS[base_lang(lang)]          # kept: some call sites want the set
    for sent_id, tokens in d.parse_conllu(path):
        by_id = {t["id"]: t for t in tokens}
        for t in tokens:
            # `comp:*` is where the current guidelines put reported speech, but the SUPERSEDED
            # analysis attached direct speech by `parataxis:obj`/`parataxis@rep`, and treebanks
            # still carry it: 151 parataxis dependents of speech verbs across the five languages,
            # including the only gold `parataxis@rep` instance there is. They are a genuine mix
            # (`said , She was happy to see yourself` is verbatim; the Latin `praedicatur ... non
            # ponitur` is an argumentative parenthesis), so they become candidates and the
            # evidence tests or the model decide -- being paratactic is not on its own enough.
            base = t["deprel"].split("@")[0]
            if not (base.startswith("comp:") or base == "parataxis"):
                continue
            head = by_id.get(t["head"])
            if head is None or head["upos"] not in ("VERB", "AUX"):
                continue
            # is_speech_lemma, not `in verbs`: the runtime rule asks exactly this
            # question and the two must never drift (see the module docstring). On gold
            # lemmas the stem fallback is a near no-op -- one token in the whole treebank.
            if not is_speech_lemma(lang, head["lemma"]):
                continue
            ids = subtree(tokens, t["id"])
            out.append({
                "sent_id": sent_id,
                "tokens": tokens,
                "comp_id": t["id"],
                "head": head,
                # Reported speech is a CLAUSE. A speech verb also takes ordinary nominal and
                # prepositional objects -- `dicit hoc` "says this", `loquor de X` "speak about X"
                # -- which are not reported speech at all and must not reach the model: in Latin
                # they are 4427 of a 4724-case residue, against just 297 clausal ones.
                "clausal": t["upos"] in ("VERB", "AUX", "SCONJ") or "VerbForm" in t["feats"],
                "direct": direct_evidence(lang, by_id, ids, t),
                "indirect": indirect_evidence(lang, by_id, ids, t),
                "clause": d.render(tokens, ids),
                "sentence": d.render(tokens, {x["id"] for x in tokens}),
            })
    return out


PREFIX = (
    "You are annotating reported speech in a syntactically parsed corpus.\n"
    "A clause is DIRECT speech if it reports the speaker's own words verbatim, as a quotation — "
    "it keeps the original pronouns, tense and deixis, and may contain interjections, vocatives "
    "or discourse particles addressed to the original hearer.\n"
    "A clause is INDIRECT speech if the words are recast from the narrator's point of view — "
    "pronouns, tense and deixis are shifted, and the clause is subordinated to the reporting verb.\n"
    "Examples:\n"
    "  He said, \"I'll be there tomorrow.\"  -> direct\n"
    "  He said that he would be there the next day.  -> indirect\n"
    "  She asked, \"Well, who's coming?\"  -> direct\n"
    "  She asked who was coming.  -> indirect\n"
    "Answer with exactly one word: direct or indirect.\n"
)


def build_prompt(c):
    # Static prefix + short variable suffix, so Ollama reuses the cached prefix KV (~4x speedup).
    return (PREFIX
            + f"\nSentence: {c['sentence']}\n"
            + f"Reporting verb: {c['head']['form']}\n"
            + f"Reported clause: {c['clause']}\n"
            + "Is the reported clause direct or indirect?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(FILES))
    ap.add_argument("--no-model", action="store_true",
                    help="rules only; leave the ambiguous residue unannotated")
    ap.add_argument("--limit", type=int, default=0, help="cap model queries (for a smoke test)")
    args = ap.parse_args()

    lang = args.lang
    pathlib.Path("caches").mkdir(exist_ok=True)
    cachep = f"caches/relabel_cache_reported_{base_lang(lang)}.jsonl"
    cache = {}
    if os.path.exists(cachep):
        for line in open(cachep, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[r["key"]] = r["label"]
    cfh = open(cachep, "a", encoding="utf-8")

    totals = {"direct_rule": 0, "indirect_rule": 0, "model": 0,
              "non_clausal": 0, "skipped": 0}
    queried = 0

    for path in FILES[lang]:
        if not os.path.exists(path):
            print(f"  missing {path} -- skip")
            continue
        cands = candidates(lang, path)
        # comp_id -> "Yes", per sentence, for the rewrite below
        decisions = {}
        for c in cands:
            key = f"{c['sent_id']}|{c['comp_id']}"
            if c["direct"] and not c["indirect"]:
                label, totals["direct_rule"] = "direct", totals["direct_rule"] + 1
            elif c["indirect"] and not c["direct"]:
                label, totals["indirect_rule"] = "indirect", totals["indirect_rule"] + 1
            elif not c["clausal"]:
                # A nominal/PP object of a speech verb with no direct-speech evidence: an ordinary
                # complement, not reported speech. Not a decision the model should be asked to make.
                totals["non_clausal"] += 1
                continue
            elif args.no_model:
                totals["skipped"] += 1
                continue
            else:
                label = cache.get(key)
                if label is None:
                    if args.limit and queried >= args.limit:
                        totals["skipped"] += 1
                        continue
                    label = d.query(build_prompt(c))
                    label = "direct" if label.startswith("direct") else "indirect"
                    cache[key] = label
                    cfh.write(json.dumps({"key": key, "label": label},
                                         ensure_ascii=False) + "\n")
                    cfh.flush()
                    queried += 1
                totals["model"] += 1
            if label == "direct":
                decisions.setdefault(c["sent_id"], set()).add(c["comp_id"])

        write_conllu(path, decisions, lang)

    cfh.close()
    print(f"{lang}: {totals}  (model queries this run: {queried})")


def write_conllu(path, decisions, lang):
    """Block-based rewrite: only the MISC cell of a committed token changes."""
    out_path = path.replace(".conllu", ".reported.conllu")
    sent_id = None
    n = 0
    lines = []
    for line in open(path, encoding="utf-8"):
        stripped = line.rstrip("\n")
        if stripped.startswith("#"):
            if stripped.startswith("# sent_id"):
                sent_id = stripped.split("=", 1)[1].strip()
            lines.append(line)
            continue
        if not stripped:
            lines.append(line)
            continue
        fields = stripped.split("\t")
        if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
            lines.append(line)
            continue
        if sent_id in decisions and int(fields[0]) in decisions[sent_id]:
            misc = [x for x in fields[9].split("|")
                    if x and x != "_" and not x.startswith("Reported=")]
            misc.append("Reported=Yes")
            fields[9] = "|".join(sorted(misc))
            n += 1
        lines.append("\t".join(fields) + "\n")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"  {out_path}: {n} Reported=Yes")


if __name__ == "__main__":
    main()
