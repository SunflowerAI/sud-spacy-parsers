#!/usr/bin/env python3
"""Does a monolingual UPOS tagger transfer to another language, and does transliteration help?

The parser needs UPOS more than anything else -- roughly one LAS point per point of tagger error --
so whether tagging can be automated decides whether the ten-sentence adaptation recipe still needs a
human for the tag column. Before building a multilingual tagger with a plug-in embedding, this asks
the cheaper prior question: is there ANY cross-lingual signal in a wordform-reading tagger?

Two conditions per target language:

    raw              the target's own orthography
    transliterated   romanised with `anyascii`, so a Greek or Georgian wordform can share character
                     n-grams with the Latin-script data the tagger was trained on

⚠ **THE MAJORITY BASELINE IS REPORTED FOR EVERY ROW.** A tagger that has never seen a script emits
whatever its bias prefers, which on most treebanks is NOUN -- and that alone scores 25-30 %. This
repo has already reported 56.5 % as a result against a 58.5 % constant. A transfer number below its
own majority baseline is worse than useless, not "weak".

⚠ The source tagger reads NORM/PREFIX/SUFFIX/SHAPE, i.e. the wordform. Transliteration is therefore
the only intervention that can possibly help it; nothing about the architecture is language-neutral.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
import seg_code  # noqa: E402,F401  (registers every released arm's components and
                 #  custom factories -- `sud_tagger` among them. Best-effort by
                 #  design: it prints and skips a language whose optional deps are
                 #  absent rather than refusing to load the rest.)
from anyascii import anyascii  # noqa: E402

#: `anyascii` is a generic codepoint mapper with no language knowledge -- it renders Thai
#: `ภาษาไทย` as `phasaaithy`, with the vowel/consonant order mangled. `uroman` takes an ISO 639-3
#: code and applies language-specific rules, giving `phaasaathai`. Romanisation quality is a
#: confound in this experiment, so both are measured rather than one being assumed adequate.
_UROMAN = None
_MEMO: dict = {}


#: ⚠ MORE STANDARD IS NOT AUTOMATICALLY MORE USEFUL HERE. Wiktra reproduces Wiktionary's
#: community-maintained romanisations, which is the right answer for a reader -- Georgian
#: `kartuli`, Coptic `Metremǹkhēmi` -- but it emits `ó`, `ǹ`, `ē`, and the point of romanising here
#: is to land in the character space an ENGLISH-trained tagger already knows. So wiktra is measured
#: both raw and ASCII-folded, and the folding is not assumed to be a loss.
def romanise(word, how, lcode, wcode=None):
    if how == "none":
        return word
    if how == "anyascii":
        return anyascii(word) or word
    if how in ("wiktra", "wiktra_ascii"):
        key = (word, wcode, how)
        if key not in _MEMO:
            try:
                import wiktra as _w
                out = _w.tr(word, lang=wcode or "und") or word
            except Exception:
                out = word          # no module for this language, or a Lua rule failed
            _MEMO[key] = anyascii(out) or out if how == "wiktra_ascii" else out
        return _MEMO[key]
    global _UROMAN
    if _UROMAN is None:
        import uroman as _ur
        _UROMAN = _ur.Uroman()
    key = (word, lcode)
    if key not in _MEMO:
        try:
            _MEMO[key] = _UROMAN.romanize_string(word, lcode=lcode) or word
        except Exception:
            _MEMO[key] = word
    return _MEMO[key]
from spacy.tokens import Doc  # noqa: E402


def read_conllu(path, limit_sents=0):
    sents, rows = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    sents.append(rows)
                    if limit_sents and len(sents) >= limit_sents:
                        return sents
                rows = []
                continue
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 8 or "-" in f[0] or "." in f[0]:
                continue
            rows.append(f)
    if rows:
        sents.append(rows)
    return sents


def score(nlp, sents, how, lcode, wcode=None):
    ok = tot = 0
    pred_tags = collections.Counter()
    for rows in sents:
        words = [romanise(r[1], how, lcode, wcode) for r in rows]
        gold = [r[3] for r in rows]
        doc = Doc(nlp.vocab, words=words)
        for name, proc in nlp.pipeline:
            if name in ("tok2vec", "morphologizer"):
                doc = proc(doc)
        for t, g in zip(doc, gold):
            if g == "_":
                continue
            tot += 1
            ok += int(t.pos_ == g)
            pred_tags[t.pos_] += 1
    return (ok / max(tot, 1)), tot, pred_tags


def majority(sents):
    c = collections.Counter(r[3] for rows in sents for r in rows if r[3] != "_")
    n = sum(c.values())
    return (c.most_common(1)[0][1] / max(n, 1)), c.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="training_en_sud_xpos/model-best")
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--limit-sents", type=int, default=300)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    nlp = spacy.load(a.source)
    if "morphologizer" not in nlp.pipe_names:
        sys.exit(f"{a.source} has no morphologizer")
    print(f"source tagger: {a.source}  ({nlp.meta.get('lang')})  pipeline {nlp.pipe_names}")

    inv = {c["lcode"] or c["lang_name"]: c for c in
           json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))["corpora"]}
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["languages"]
    langs = a.langs or sorted(k for k, v in man.items() if v["pool"] == "test")

    out = {}
    print(f"\n{'lang':5s} {'script':10s} {'majority':>9s} {'raw':>8s} {'anyasc':>8s} "
          f"{'uroman':>8s} {'wiktra':>8s} {'wik+asc':>8s} {'best-Δ':>8s}  top-pred")
    for lg in langs:
        paths = inv[lg]["paths"].get("test") or inv[lg]["paths"].get("train")
        if not paths:
            continue
        sents = read_conllu(paths[0], a.limit_sents)
        if not sents:
            continue
        base, btag = majority(sents)
        raw_txt = "".join(r[1] for rows in sents[:40] for r in rows)
        latin = sum(1 for ch in raw_txt if ch.isalpha() and ord(ch) < 0x250) / \
            max(sum(1 for ch in raw_txt if ch.isalpha()), 1)
        script = "Latin" if latin > 0.9 else "non-Latin"
        iso = inv[lg]["iso3"] or None
        accs = {}
        for how in ("none", "anyascii", "uroman", "wiktra", "wiktra_ascii"):
            accs[how], n, pr = score(nlp, sents, how, iso, lg)
            if how == "none":
                r_pred = pr
        best = max(accs.values())
        out[lg] = {"majority": base, "tokens": n, "script": script, **accs}
        print(f"{lg:5s} {script:10s} {100 * base:9.1f} " +
              " ".join(f"{100 * accs[h]:8.1f}" for h in
                       ("none", "anyascii", "uroman", "wiktra", "wiktra_ascii")) +
              f" {100 * (best - base):+8.1f}  {r_pred.most_common(1)[0][0]}")

    nl = {k: v for k, v in out.items() if v["script"] == "non-Latin"}
    print(f"\n{sum(1 for v in out.values() if max(v['none'], v['anyascii'], v['uroman'], v['wiktra'], v['wiktra_ascii']) > v['majority'])}"
          f"/{len(out)} languages beat their own majority baseline at all")
    for how in ("anyascii", "uroman", "wiktra", "wiktra_ascii"):
        n_tr = sum(1 for v in out.values() if v[how] > v["none"] + 0.01)
        d = (sum(v[how] - v["none"] for v in nl.values()) / len(nl)) if nl else 0
        print(f"  {how:9s}: helped {n_tr}/{len(out)} languages; "
              f"non-Latin average {100 * d:+.1f} points")
    if a.json:
        json.dump({"source": a.source, "languages": out}, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
