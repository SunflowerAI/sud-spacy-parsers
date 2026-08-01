#!/usr/bin/env python3
"""Synthesise (saṃhitā, CSL, per-character labels) training triples for the pre-tokeniser.

THE PROBLEM. `sa_tokenizer` requires input that is already word-segmented: it reads CSL notation
(word spaces, compound `-`, coalescence marks `'`/`"` + `â ê î ô û ...`) and reverts the marked
sandhi. Continuous saṃhitā — real sandhied text with the word breaks not written — is out of scope,
so the model cannot be pointed at an ordinary printed edition. A learned stage that turns saṃhitā
into CSL would close that, leaving everything downstream untouched.

WHY GOLD DATA CAN BE SYNTHESISED RATHER THAN ANNOTATED. CSL and the true sandhied surface differ at
exactly ONE class of junction — vowel coalescence, where CSL splits the fused vowel across the
junction to stay reversible (`rājā uvāca` -> `rāj" ôvāca`) and a printed text writes the single fused
vowel (`rājovāca`). Every other rule in `external_sandhi.py` already emits the true surface, which
CSL keeps verbatim. `external_sandhi.COALESCE_SURFACE` is that one difference, derived from the
engine's own `_coalesce`, so the alignment is exact BY CONSTRUCTION and every label is gold.

LABELS. One label per saṃhitā character, naming the CSL string that character expands into:

    "="      keep the character as it stands                       (most characters)
    "= "     keep it, then a word break
    "=-"     keep it, then a compound / preverb / privative break
    "' ô "   REPLACE it (elision marker, break, the right word's mark, ...)   -- coalescence
    ""       absorbed: the second character of a two-character fused vowel (ai / au)

A label starting with `=` keeps the character and appends the rest; anything else replaces it. `=`
cannot collide with the text (IAST has no `=`). The alphabet is character-INDEPENDENT, so the model
learns "insert a word break here", not "a -> a-space" and "ṃ -> ṃ-space" as separate classes.

The word-vs-compound break distinction comes free from `apply_vedic_sandhi`'s `bound[]`, already
baked into the `# text` line. Hellwig & Nehrdich 2018 deliberately conflate the two; we cannot,
because the compound divider is what `sa_tokenizer` reads to stamp `Compound=Yes` (worth +1.30 LAS).

SPACING. `--spacing` controls how much of the segmentation the INPUT is assumed to give away, and it
matters far more than it looks. An earlier version of this file claimed a spaced text "is strictly
easier and needs no separate training data: the labels are identical either way". **That is wrong.**
A model trained only on `continuous` scores 98.07 split-location F on continuous DCS test but 88.44
on the IAST-spaced version of the SAME sentences and 91.13 on the Devanagari-spaced one — worse,
despite those having a quarter as many breaks left to find. Trained on one break every ~4.5
characters, it over-segments a short pre-spaced chunk. Use `--spacing mixed` for a model that has to
cope with any convention. See `respace`.

    make_samhita_pairs.py IN.sandhi.conllu OUT.jsonl [--spacing mixed] [--vocab OUT.json]
"""
import argparse
import collections
import json
import random
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_sandhi import COALESCE_MARKS, COALESCE_SURFACE     # noqa: E402

MARKERS = ("'", '"')            # left-word elision markers: ' short, " long
DIVIDERS = (" ", "-")


class RoundTripError(Exception):
    pass


def split_csl(csl):
    """CSL string -> (words, dividers), where dividers[i] joins words[i] and words[i+1]."""
    words, divs, cur = [], [], []
    for ch in csl:
        if ch in DIVIDERS:
            words.append("".join(cur))
            divs.append(ch)
            cur = []
        else:
            cur.append(ch)
    words.append("".join(cur))
    return words, divs


def leading_mark(word):
    """The CSL coalescence mark a right-hand word starts with, or None."""
    for m in COALESCE_MARKS:
        if word.startswith(m):
            return m
    return None


def csl_to_pairs(csl):
    """CSL string -> (saṃhitā string, list of per-character labels).

    Walks left to right, consuming a coalescence junction as a unit. `work` is mutable because a
    coalescence eats the mark off the FOLLOWING word, and that word may then be empty (a one-vowel
    particle such as the emphatic `u`, wholly absorbed into its neighbour: `atha u` -> `ath' ô`),
    in which case its own divider is carried into the same label.
    """
    csl = unicodedata.normalize("NFC", csl)
    work, divs = split_csl(csl)
    out, labels, i = [], [], 0
    while i < len(work):
        w = work[i]
        div = divs[i] if i < len(divs) else None
        mark = None
        if div is not None and w and w[-1] in MARKERS:
            mark = leading_mark(work[i + 1])

        if mark is None:                                   # ordinary junction (or end of string)
            if not w:                                      # empty word: nothing to label
                i += 1
                continue
            out.extend(w)
            labels.extend(["="] * len(w))
            if div is not None:
                labels[-1] = "=" + div
            i += 1
            continue

        # ---- coalescence -------------------------------------------------------------------
        marker, body = w[-1], w[:-1]
        surface = COALESCE_SURFACE[mark]
        rest = work[i + 1][len(mark):]
        tail = ""                                          # divider(s) trailing the mark
        if not rest and i + 1 < len(divs):
            tail = divs[i + 1]                             # right word wholly absorbed
            work[i + 1] = ""
        else:
            work[i + 1] = rest
        out.extend(body)
        labels.extend(["="] * len(body))
        out.append(surface[0])
        labels.append(marker + div + mark + tail)
        for extra in surface[1:]:                          # ai / au: second char emits nothing
            out.append(extra)
            labels.append("")
        if not rest:
            i += 2                                         # skip the emptied word
        else:
            i += 1
    return "".join(out), labels


def expand(samhita, labels):
    """Inverse of csl_to_pairs: apply the labels to the saṃhitā string to rebuild the CSL string."""
    out = []
    for ch, lab in zip(samhita, labels):
        out.append(ch + lab[1:] if lab.startswith("=") else lab)
    return "".join(out)


VOW = set("aāiīuūṛṝḷeo")


def respace(samhita, labels, mode):
    """Re-emit a pair under the spacing the SOURCE ORTHOGRAPHY would actually print.

    The three regimes differ only in which boundaries the input hands you for free:

      continuous  no spaces at all — the model finds every boundary. Hardest, and the right
                  fallback when the convention is unknown.
      iast        a space at every plain word break; compounds and vowel coalescence stay solid.
      devanagari  as IAST, but the script ALSO forces solid an avagraha (नमोऽस्तु) and a
                  consonant-final word before a vowel-initial one (वह्निरिद्रः), because
                  Devanagari cannot render a bare consonant before a vowel.

    Training on `continuous` alone is not enough: measured on DCS test, a model trained only on it
    scores 98.07 split-location F on continuous input but 88.44 on IAST-spaced and 91.13 on
    Devanagari-spaced — WORSE, despite those regimes having a quarter as many breaks to find. It is
    calibrated for a break every ~4.5 characters and over-segments a pre-spaced chunk. So the
    spacing regime belongs in the data, not in an inference-time flag.
    """
    if mode == "continuous":
        return samhita, list(labels)
    out_c, out_l = [], []
    for i, (ch, lab) in enumerate(zip(samhita, labels)):
        nxt = samhita[i + 1] if i + 1 < len(samhita) else ""
        give = False
        if lab == "= ":                                  # a plain word break, not bound/coalesced
            avagraha = nxt == "'"
            cons_vowel = ch not in VOW and (nxt in VOW or avagraha)
            give = True if mode == "iast" else not (avagraha or cons_vowel)
        if give:
            out_c.append(ch); out_l.append("=")           # the boundary is GIVEN by the space
            out_c.append(" "); out_l.append("=")
        else:
            out_c.append(ch); out_l.append(lab)
    return "".join(out_c), out_l


def texts(path):
    """Yield (sent_id, csl_text) per block. `# text` precedes `# sent_id` in these files, so this
    reads whole blocks rather than assuming an order. The `# text =` test is exact on purpose —
    UFAL also carries `# text_fr =`, which would silently poison the data."""
    sid = txt = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if txt is not None:
                yield sid, txt
            sid = txt = None
        elif line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif line.startswith("# text ="):
            txt = line.split("=", 1)[1].strip()
    if txt is not None:
        yield sid, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="a *.sandhi.conllu produced by apply_vedic_sandhi.py")
    ap.add_argument("out", help="output JSONL")
    ap.add_argument("--vocab", default=None, help="dump the char + label inventories here")
    ap.add_argument("--spacing", default="continuous",
                    choices=("continuous", "iast", "devanagari", "mixed"),
                    help="which spacing the INPUT is assumed to carry; `mixed` assigns one regime "
                         "per sentence at random, which is what a single model handling any "
                         "convention should be trained on (see `respace`).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-elided", action="store_true",
                    help="keep sentences containing an elided `_` token (they cannot occur in real "
                         "input and the model can never predict them)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = kept = elided = 0
    regimes = collections.Counter()
    chars, labs = {}, {}
    failures = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for sid, csl in texts(args.inp):
            n += 1
            if "_" in csl and not args.keep_elided:
                elided += 1
                continue
            samhita, labels = csl_to_pairs(csl)
            back = expand(samhita, labels)
            if back != unicodedata.normalize("NFC", csl):
                failures.append((sid, csl, back))
                continue
            mode = (rng.choice(("continuous", "iast", "devanagari"))
                    if args.spacing == "mixed" else args.spacing)
            samhita, labels = respace(samhita, labels, mode)
            regimes[mode] += 1
            kept += 1
            for c in samhita:
                chars[c] = chars.get(c, 0) + 1
            for lb in labels:
                labs[lb] = labs.get(lb, 0) + 1
            fh.write(json.dumps({"sent_id": sid, "samhita": samhita, "csl": csl,
                                 "labels": labels}, ensure_ascii=False) + "\n")

    print(f"{args.inp}: {n} sentences -> {kept} written "
          f"({elided} dropped for an elided `_`, {len(failures)} round-trip failures)")
    if failures:
        print("  ROUND-TRIP FAILURES (first 5) — the label set does not cover these:")
        for sid, csl, back in failures[:5]:
            print(f"    {sid}\n      csl  {csl}\n      back {back}")
        sys.exit(1)
    print(f"  {len(chars)} distinct characters, {len(labs)} distinct labels"
          + (f"; regimes {dict(regimes)}" if len(regimes) > 1 else f"; spacing={args.spacing}"))
    top = sorted(labs.items(), key=lambda kv: -kv[1])
    keep_pct = 100 * labs.get("=", 0) / max(1, sum(labs.values()))
    print(f"  '=' (keep) is {keep_pct:.1f}% of characters; top labels: "
          + ", ".join(f"{k!r}:{v}" for k, v in top[:8]))
    if args.vocab:
        with open(args.vocab, "w", encoding="utf-8") as fh:
            json.dump({"chars": chars, "labels": labs}, fh, ensure_ascii=False, indent=1)
        print(f"  vocab -> {args.vocab}")


if __name__ == "__main__":
    main()
