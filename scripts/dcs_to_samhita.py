#!/usr/bin/env python3
"""Turn DCS (Digital Corpus of Sanskrit) CoNLL-U into CSLiser training pairs — and validate the
sandhi engine against real editorial Sanskrit while doing it.

DCS gives, per token, the **unsandhied** form (`Unsandhied=`) and compound membership
(`Case=Cpd`), plus the **real editorial sandhied line** in `# text`. That is exactly the input
`external_sandhi.join_pair` needs, and — for the first time in this project — a reference to check
its output against. CLAUDE.md has always carried the caveat "there is no gold sandhied form in the
treebank, so this is rule-based *generation*, not alignment"; running the engine over DCS and
diffing against `# text` is the missing validation.

So this script does two jobs:

  * `--validate` — generate the orthographic rendering from the unsandhied tokens and compare it to
    DCS's own `# text`. Reports exact-line agreement and a ranked list of the junctions where the
    engine disagrees. Run this BEFORE training on the output: if the engine's sandhi is its own
    idiolect rather than real Sanskrit, the training data is fiction.
  * default — emit the same JSONL triples as `scripts/make_samhita_pairs.py`
    ({sent_id, samhita, csl, labels}), so the CSLiser can train on classical/epic text instead of
    only Vedic ritual prose.

DCS is CC BY 4.0 (github.com/OliverHellwig/sanskrit); record it in NOTICE.md before shipping
anything trained on it.

    dcs_to_samhita.py TEXTDIR... --validate
    dcs_to_samhita.py TEXTDIR... --out data_samhita/dcs.jsonl
"""
import argparse
import collections
import glob
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_vedic_sandhi import generate                      # noqa: E402
from make_samhita_pairs import csl_to_pairs, expand          # noqa: E402


def sentences(path):
    """Yield (sent_id, dcs_text, words, feats, bound) per sentence."""
    sid = text = None
    toks = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if text is not None and toks:
                yield sid, text, toks
            sid = text = None
            toks = []
        elif line.startswith("# text = "):
            text = line.split("= ", 1)[1].strip()
        elif line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0] and c[0].split("-")[0].isdigit():
                continue
            misc = dict(kv.split("=", 1) for kv in c[9].split("|") if "=" in kv)
            toks.append((misc.get("Unsandhied") or c[1], c[5]))
    if text is not None and toks:
        yield sid, text, toks


def build(toks):
    """(words, feats, bound) -> (csl_string, orthographic_string)."""
    words = [unicodedata.normalize("NFC", w) for w, _ in toks]
    feats = [f for _, f in toks]
    # DCS marks a non-final compound member with Case=Cpd — the same role Vedic's Compound=Yes
    # plays, and what `generate` wants as its `internal` mask.
    bound = ["Case=Cpd" in (feats[k] or "") for k in range(len(words) - 1)]
    pieces = generate(words, feats, bound)
    dividers = ["-" if bound[k] else " " for k in range(len(words) - 1)]
    csl = "".join(p + (dividers[i] if i < len(dividers) else "")
                  for i, p in enumerate(pieces))
    samhita, labels = csl_to_pairs(csl)
    return csl, samhita, labels


def norm_cmp(s):
    """Compare modulo whitespace and the punctuation DCS prints but the engine does not."""
    s = unicodedata.normalize("NFC", s)
    return "".join(ch for ch in s if not ch.isspace() and ch not in ",.;:!?'\"|/-‖")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-match", action="store_true",
                    help="emit a sentence ONLY when the generated saṃhitā reproduces DCS's own "
                         "editorial `# text`. DCS's `Unsandhied` is frequently a normalised "
                         "citation form rather than the true underlying form (`mama` where the "
                         "text reads `me`, `vidvāḥ` for `vidvān`, `dhiṣṭhita` for `adhiṣṭhita`), "
                         "and is heuristically reconstructed for most texts — so agreement with "
                         "the printed line is the honest test that a generated string is real "
                         "Sanskrit rather than an artefact of imperfect annotation.")
    a = ap.parse_args()

    files = []
    for d in a.dirs:
        files += sorted(glob.glob(os.path.join(d, "*.conllu")))
    stat = collections.Counter()
    junction_errs = collections.Counter()
    out_fh = open(a.out, "w", encoding="utf-8") if a.out else None
    n = 0
    for f in files:
        for sid, dcs_text, toks in sentences(f):
            if a.limit and n >= a.limit:
                break
            n += 1
            if any(w in ("_", "") for w, _ in toks):
                stat["skip_elided"] += 1
                continue
            try:
                csl, samhita, labels = build(toks)
            except Exception:
                stat["skip_error"] += 1
                continue
            if expand(samhita, labels) != unicodedata.normalize("NFC", csl):
                stat["skip_roundtrip"] += 1
                continue
            matched = norm_cmp(samhita) == norm_cmp(dcs_text)
            if a.require_match and not matched:
                stat["skip_mismatch"] += 1
                continue
            if a.validate:
                ours = norm_cmp(samhita)
                theirs = norm_cmp(dcs_text)
                if matched:
                    stat["match"] += 1
                else:
                    stat["differ"] += 1
                    if stat["differ"] <= 400:
                        # locate the first divergence for a coarse error profile
                        i = next((k for k in range(min(len(ours), len(theirs)))
                                  if ours[k] != theirs[k]), min(len(ours), len(theirs)))
                        junction_errs[(theirs[max(0, i - 1):i + 2], ours[max(0, i - 1):i + 2])] += 1
            if out_fh:
                out_fh.write(json.dumps({"sent_id": sid, "samhita": samhita, "csl": csl,
                                         "labels": labels}, ensure_ascii=False) + "\n")
                stat["written"] += 1
        if a.limit and n >= a.limit:
            break
    if out_fh:
        out_fh.close()
    print(f"{n} sentences from {len(files)} files")
    for k, v in stat.most_common():
        print(f"  {k:16s} {v}")
    if a.validate and stat["match"] + stat["differ"]:
        tot = stat["match"] + stat["differ"]
        print(f"\n  ENGINE vs DCS editorial text: {stat['match']}/{tot} "
              f"({100 * stat['match'] / tot:.2f} %) exact lines")
        print("  most common divergences (DCS context -> ours):")
        for (t, o), c in junction_errs.most_common(12):
            print(f"     x{c:<4d} {t!r} -> {o!r}")


if __name__ == "__main__":
    main()
