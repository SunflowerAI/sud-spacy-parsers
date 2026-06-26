#!/usr/bin/env python
"""Apply external (and compound-internal) sandhi to the Vedic SUD treebank, rendering the
result in CSL notation — the same representation produced for UFAL by `sa_csl_prep.py`.

The Vedic treebank is pausa (every word pre-sandhi, space-separated; compounds marked
`Compound=Yes` but unjoined). Here we:
  * apply sandhi at every within-sentence junction with `external_sandhi.join_pair`
    (combining each word's left- and right-junction modifications);
  * join a `Compound=Yes` left word to the next with a hyphen (compound-internal,
    re-tokenisable), everything else with a space (external);  same grouping as UFAL, so
    compound runs become hyphen-baked MWT ranges.
Sentence-initial / -final words and elided `_` tokens are left in pausa.

Usage: .venv/bin/python scripts/apply_vedic_sandhi.py IN.conllu OUT.conllu
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retokenize import read_conllu                       # noqa: E402
from external_sandhi import join_pair                    # noqa: E402
from sa_csl_prep import _emit_groups, rebuild_text        # noqa: E402


# the traditional upasargas (verb preverbs); bound to their verb, so joined like a compound
UPASARGA = {"ā", "pra", "parā", "apa", "sam", "saṃ", "anu", "ava", "nis", "nir", "niḥ", "dus",
            "dur", "duḥ", "vi", "ud", "ut", "ni", "adhi", "api", "ati", "su", "abhi", "prati",
            "pari", "upa", "abhī"}


def is_privative(t):
    return t.upos == "PART" and t.lemma in ("a", "an")


def is_preverb(t, nxt):
    # a preverb bound to the verb it immediately precedes (head == that verb); the head
    # check excludes Vedic tmesis (preverb separated from its verb) and adverbial particles.
    return (nxt is not None and t.lemma in UPASARGA and t.upos in ("ADV", "ADP")
            and nxt.upos == "VERB" and t.head == nxt.id)


def generate(words, feats, internal):
    """Forward-sandhi a sentence's pausa words -> CSL surface pieces (one per word).
    `internal[j]` marks a bound junction (compound / preverb / a-/an- prefix).

    Applies sandhi **sequentially left-to-right**, carrying each word's evolving surface into the
    next junction. This matters for single-character words (notably the particle `u`): a word that
    coalesces on its left has no original material left for its right junction, so computing the two
    junctions independently and merging them mishandled it (`atha u āhuḥ` came out `ath' u āhuḥ` —
    the left elided but `u` stayed uncoalesced). Sequencing makes `u` coalesce correctly into the
    left word's vowel: `atha u -> ath' ô` (= atho)."""
    surf = []
    for i, w in enumerate(words):
        if i == 0:
            surf.append(w)
            continue
        L, R = surf[-1], w
        if L == "_" or R == "_":                           # elided token: no surface, blocks sandhi
            surf.append(R)
            continue
        fl = "" if internal[i - 1] else feats[i - 1]       # pragṛhya only blocks *external* sandhi
        lout, rout = join_pair(L, R, fl, internal[i - 1])
        surf[-1] = lout
        surf.append(rout)
    return surf


def process(in_path, out_path):
    n_sent = n_comp = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for comments, toks, mwts in read_conllu(in_path):
            n_sent += 1
            by_id = {int(t.id): t for t in toks}
            ids = [int(t.id) for t in toks]
            words = [t.form for t in toks]
            feats = [t.feats for t in toks]
            comp = ["Compound=Yes" in t.feats.split("|") for t in toks]
            n_comp += sum(comp)
            # a junction is *bound* (sandhi + hyphen) if the left word is a compound member,
            # a preverb of the following verb, or the privative prefix a-/an-.
            bound = [comp[k] or is_preverb(toks[k], toks[k + 1]) or is_privative(toks[k])
                     for k in range(len(toks) - 1)]
            pieces = generate(words, feats, bound)
            dividers = ["-" if (bound[k] and words[k] != "_" and words[k + 1] != "_") else " "
                        for k in range(len(ids) - 1)]    # never hyphenate an elided _ token
            out_ranges = []
            _emit_groups(ids, pieces, dividers, by_id, "_", out_ranges, flagged=False)
            range_at = {}
            for a, b, form, misc in out_ranges:
                range_at.setdefault(a, []).append((b, form, misc))
            for c in comments:
                out.write(("# text = " + rebuild_text(toks)) if c.startswith("# text =") else c)
                out.write("\n")
            for t in toks:
                tid = int(t.id)
                for b, form, misc in range_at.get(tid, []):
                    out.write("\t".join([f"{tid}-{b}", form, "_", "_", "_", "_",
                                         "_", "_", "_", misc]) + "\n")
                out.write("\t".join([t.id, t.form, t.lemma, t.upos, t.xpos, t.feats,
                                     t.head, t.deprel, t.deps, t.misc]) + "\n")
            out.write("\n")
    print(f"{in_path}: {n_sent} sentences, {n_comp} compound members -> {out_path}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    process(a.inp, a.out)


if __name__ == "__main__":
    main()
