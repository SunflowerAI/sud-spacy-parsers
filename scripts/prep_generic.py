#!/usr/bin/env python3
"""Build the GENERIC parser's corpus: thirteen SUD treebanks, one label set, balanced by TYPE.

WHAT THE GENERIC ARM IS. A parser that sees only UPOS, FEATS and a cross-lingually aligned vector
per token -- no wordform, no affix, no shape, no language identifier -- and predicts heads and SUD
deprels. Every other arm in this repo is monolingual and reads the token string; this one cannot,
by construction, which is the whole point. See `docs/generic-parser.md`.

This script prepares its data. Four normalisations, each of which is a decision rather than a
cleanup, and each of which would silently change the experiment if it were skipped:

1. **`@`-SUBTYPES ARE STRIPPED.** The thirteen treebanks together use 120 deprels, and roughly 60 of
   them occur in exactly ONE language (`mod@neg` is fa's alone at 7 818 tokens, `flat@vv` ko's at
   4 868, `udep@instr` sa's at 1 982). Predicting those from a shared model means predicting a class
   whose only evidence is one treebank's annotation convention -- and at this corpus size
   `min_action_freq` would delete most of them anyway, pinning their recall to zero without saying
   so (`docs/dravidian.md`). Stripping `@` leaves **27** relations, every one attested in at least
   four languages and twenty in ten or more. The `:` subtypes are KEPT: `comp:obj`/`comp:obl`/
   `comp:pred`/`comp:aux` and `conj:coord`/`conj:appos` are core SUD, defined identically across
   treebanks, and collapsing them would throw away the distinction this project exists to study.

2. **`Shared` IS REMOVED FROM FEATS, AND THIS ONE IS LEAKAGE.** `Shared=Yes/No` is native SUD
   annotation -- it is in the pristine treebanks, not something this repo hoisted -- and it records
   whether a dependent is shared across the conjuncts of a coordination. That is a fact about the
   TREE. Our own `sud_shared` pipe predicts it FROM a finished parse (`docs/sud-misc-layer.md`), so
   handing it to the parser as an input inverts the dependency and leaks coordination structure into
   a model whose job is to recover it. It is on 10 178 en tokens alone and in all thirteen
   treebanks, so leaving it in would have been quiet and expensive.

3. **XPOS IS BLANKED.** One tagset per arm is the rule (`docs/xpos.md`) -- la's composite codes,
   en's `,`, te's verbatim copy of UPOS -- so XPOS is the least commensurable column in the file.
   UPOS is the universal one and is what the generic arm reads.

4. **AN EMPTY LEMMA FALLS BACK TO THE FORM.** te's LEMMA column is `_` on every one of its tokens
   and spaCy keeps `_` as a LITERAL string, not as missing (CLAUDE.md; it once taught a Sanskrit
   transducer `FORM -> "_"` on 5 043 tokens). sa's aligned vectors are keyed by LEMMA, so a literal
   `_` reaching the lookup would be a silent all-OOV language. Identity is the fallback
   `scripts/prep_te.py` already uses.

THE BALANCE IS TYPOLOGICAL, NOT PER-LANGUAGE. Equal shares per language would still be a
genealogical accident: four of the thirteen treebanks are Indo-European and three are Sinitic, so
"one share each" hands 54 % of the corpus to two families. Each GROUP gets an equal token budget,
split equally within it, and capacity a small treebank cannot fill is redistributed to its own
group's siblings -- never across groups, which would undo the balance being bought.

Two groups cannot reach parity at any budget and the manifest says so rather than hiding it:
Dravidian has 13 506 training tokens in total (ta 8 409 + te 5 097) and Koreanic 56 687. At the
default budget those are 3.6 % and 15.3 % of the sample against an even seventh's 14.3 %.

SAMPLING IS BY 10-SENTENCE BLOCK, NOT BY SENTENCE. The corpora are converted with `spacy convert
-n 10` and read by `sud.GenericCorpus.v1`, which inherits `GoldTokCorpus`'s contract: a doc is ten
sentences, so the parser learns to START one instead of scoring a cosmetic `SENTS_F` of 100
(CLAUDE.md hazard 4). Sampling whole blocks keeps each doc ten CONSECUTIVE sentences of one text;
sampling loose sentences would build docs out of ten unrelated fragments and quietly make sentence
segmentation easier than it is.

Dev is capped the same way -- thirteen full dev sets are 253 k tokens and every eval would pay for
them. **Test is never sampled**: the arm is scored on each language's complete test set.

    .venv/bin/python scripts/prep_generic.py                       # default budget, all 13
    .venv/bin/python scripts/prep_generic.py --budget 30000        # a tighter, better-balanced mix
    .venv/bin/python scripts/prep_generic.py --hold-out ja         # for the zero-shot arms
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import zlib

# The released generation of each treebank -- kept in step with `src_conllu()` in
# scripts/train_sud.sh, which is the authority. Deliberately the SAME files the shipped monolingual
# arms train on, so the generic arm's per-language numbers can be read against theirs.
#
# ⚠ ta and te are NOT udep-relabelled and the other eleven are (extended scope). That is visible in
# the data -- `udep` is 9.9 % of ta's tokens, 2.4 % of en's and 0.0 % of sa's, whose extended relabel
# committed every one -- and it is a difference in ANNOTATION POLICY, not in the languages. The
# manifest reports the per-language `udep` rate so the gap cannot be mistaken for a finding.
SRC = {
    "en":  "assets/en_ewt-sud-%s.relabeled_ext.conllu",
    "zh":  "assets_zh/SUD_Chinese-GSDBoth/zh_gsdboth-sud-%s.relabeled_ext.conllu",
    "yue": "assets_yue/SUD_Cantonese-HK/yue_hk-sud-%s.relabeled_ext.conllu",
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto/"
           "lzh_kyoto-sud-%s.relabeled_ext.udep_ruled.punct.rulemerged.conllu",
    "fa":  "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-%s.relabeled_ext.conllu",
    "ar":  "assets_ar/SUD_Arabic-PADT/ar_padt-sud-%s.relabeled_ext.conllu",
    "la":  "assets_la/la_ittbproiel-sud-%s.relabeled_ext.conllu",
    "id":  "assets_id/SUD_Indonesian-GSD/id_gsd-sud-%s.relabeled_ext.conllu",
    "ko":  "assets_ko/SUD_Korean-GSD/ko_gsd-sud-%s.relabeled_ext.conllu",
    "ja":  "assets_ja/SUD_Japanese-GSD/ja_gsd-sud-%s.relabeled_ext.udep_ruled.conllu",
    "ta":  "assets_ta/ta_ttb_mwtt-sud-%s.conllu",
    "te":  "assets_te/te_mtg-sud-%s.conllu",
    "sa":  None,
}
#: sa's train is Vedic + UFAL combined; dev/test are Vedic only.
#:
#: ⚠ THIS MUST BE THE `csl_mwt` GENERATION, NOT `csl_rev`. `train_sud.sh`'s `src_conllu()` still
#: names `corpus_sa_csl_rev/`, and CLAUDE.md lists `rebuild_sa_csl_rev.sh` under "Superseded but
#: kept" -- the pausa-normalised representation it builds was replaced by the MWT one. Copying that
#: stale path in cost two things at once and neither announced itself: the corpus came out
#: **unrelabelled** (udep 7.89 % of sa tokens, against 0.00 % in the current generation, where the
#: extended relabel committed every one) while all ten other relabelled languages were on their
#: released generation; and its tokenisation differs from the released sa arm's, so no
#: monolingual comparison on it was valid. A superseded corpus loads, converts and trains exactly
#: like a current one.
SA = {
    "train": "corpus_sa_mwt_rl2/train.csl_mwt.conllu",
    "dev":   "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-dev.relabeled_ext.csl_mwt.conllu",
    "test":  "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-test.relabeled_ext.csl_mwt.conllu",
}

# The balance axis. Genealogical labels, but the grouping is doing typological work: the four
# Indo-European treebanks span fusional-with-articles (en), fusional-caseless (fa), fusional-rich
# (la, sa) and the three Sinitic ones span isolating-classical (lzh) and isolating-modern (zh, yue).
# Splitting a group's budget evenly across those is exactly the diversity the sample is buying.
GROUPS = {
    "Indo-European": ["en", "fa", "la", "sa"],
    "Sinitic":       ["zh", "yue", "lzh"],
    "Dravidian":     ["ta", "te"],
    "Semitic":       ["ar"],
    "Japonic":       ["ja"],
    "Koreanic":      ["ko"],
    "Austronesian":  ["id"],
}
LANGS = [l for g in GROUPS.values() for l in g]

#: Native SUD FEATS keys that are facts about the TREE rather than about the token. See note 2.
LEAKY_FEATS = {"Shared"}


def lang_seed(seed: int, lang: str) -> int:
    """A per-language seed that is STABLE ACROSS PROCESSES.

    `hash(str)` is randomised by PYTHONHASHSEED, so seeding a sampler with it gives a different
    sample every run while looking entirely deterministic in the source. crc32 is stable.
    """
    return seed * 1000003 + zlib.crc32(lang.encode("utf-8"))


def src_path(lang: str, split: str) -> str:
    return SA[split] if lang == "sa" else SRC[lang] % split


# --------------------------------------------------------------------------------------------
# CoNLL-U in, normalised CoNLL-U out


class Sent:
    __slots__ = ("comments", "rows")

    def __init__(self, comments, rows):
        self.comments = comments
        self.rows = rows

    def __len__(self):
        return len(self.rows)


def read_conllu(path):
    """Word rows only -- MWT ranges (`3-4`) and empty nodes (`3.1`) are dropped.

    Both are legitimate CoNLL-U and neither survives into a `.spacy` corpus as a parseable token, so
    keeping them here would only let them reach the token counts and skew the budget.
    """
    sents, comments, rows = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    sents.append(Sent(comments, rows))
                comments, rows = [], []
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            rows.append(f)
    if rows:
        sents.append(Sent(comments, rows))
    return sents


def strip_subtype(deprel: str) -> str:
    """`mod@relcl` -> `mod`, `comp:obl@agent` -> `comp:obl`. The `:` subtype is core SUD; `@` is not."""
    return deprel.split("@", 1)[0]


def clean_feats(feats: str) -> str:
    if feats == "_":
        return "_"
    keep = [kv for kv in feats.split("|")
            if kv.split("=", 1)[0] not in LEAKY_FEATS
            # defensive: `hoist_sud_gold.py` writes Sud-prefixed keys into FEATS for the MISC layer,
            # and those are gold for a pipe that runs downstream of this parser.
            and not kv.startswith("Sud")]
    return "|".join(keep) if keep else "_"


def normalise(sent: Sent) -> Sent:
    out = []
    for f in sent.rows:
        f = list(f)
        while len(f) < 10:
            f.append("_")
        # LEMMA: identity fallback, never the literal `_` (see note 4).
        if f[2] == "_":
            f[2] = f[1]
        f[4] = "_"                                   # XPOS blanked (note 3)
        f[5] = clean_feats(f[5])                     # Shared removed (note 2)
        f[7] = strip_subtype(f[7])                   # @-subtypes stripped (note 1)
        f[8] = "_"                                   # enhanced deps: never used by this project
        # MISC: SpaceAfter=No is the only key `spacy convert` reads, and it is what reconstructs the
        # text. Everything else is annotation for other layers and is dropped.
        misc = [kv for kv in f[9].split("|") if kv == "SpaceAfter=No"] if f[9] != "_" else []
        f[9] = "|".join(misc) if misc else "_"
        out.append(f)
    return Sent(list(sent.comments), out)


def write_conllu(path, sents):
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            for c in s.comments:
                fh.write(c + "\n")
            for f in s.rows:
                fh.write("\t".join(f) + "\n")
            fh.write("\n")


# --------------------------------------------------------------------------------------------
# The typologically-balanced sample


def blocks_of(sents, n):
    return [sents[i:i + n] for i in range(0, len(sents), n)]


def sample_blocks(sents, budget, block, seed):
    """Draw whole `block`-sentence blocks, without replacement, until `budget` tokens are reached.

    Returns the sentences in their ORIGINAL order. Order does not matter to the reader (it shuffles
    docs), but keeping it makes the written CoNLL-U diffable against its source and makes a doc ten
    CONSECUTIVE sentences rather than ten unrelated ones.
    """
    blks = blocks_of(sents, block)
    if budget is None or sum(len(s) for s in sents) <= budget:
        return sents, False                          # nothing to sample: the whole split fits
    rng = random.Random(seed)
    order = list(range(len(blks)))
    rng.shuffle(order)
    taken, got = [], 0
    for i in order:
        if got >= budget:
            break
        taken.append(i)
        got += sum(len(s) for s in blks[i])
    taken.sort()
    return [s for i in taken for s in blks[i]], True


def allocate(group_budget, sizes):
    """Split one group's budget equally among its members, redistributing what the small ones cannot fill.

    Iterative because redistribution can itself overshoot a second member: giving Sinitic's unused
    yue capacity to zh and lzh is fine, but in a group where the second-smallest also caps, one pass
    would leave the budget unspent. Redistribution stays INSIDE the group -- moving it across groups
    would spend the balance this function exists to buy.
    """
    langs = list(sizes)
    alloc = {l: 0 for l in langs}
    remaining, open_langs = group_budget, set(langs)
    while open_langs and remaining > 0:
        share = remaining / len(open_langs)
        capped = {l for l in open_langs if sizes[l] <= share}
        if not capped:
            for l in open_langs:
                alloc[l] = int(share)
            remaining = 0
            break
        for l in capped:
            alloc[l] = sizes[l]
            remaining -= sizes[l]
            open_langs.discard(l)
    return alloc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets_generic",
                    help="directory for the normalised, sampled CoNLL-U")
    ap.add_argument("--budget", type=int, default=60000,
                    help="TRAIN tokens per typological group (default 60000 -> ~370k total)")
    ap.add_argument("--dev-budget", type=int, default=3000,
                    help="dev tokens per LANGUAGE; test is never sampled")
    ap.add_argument("--block", type=int, default=10,
                    help="sentences per sampling block; must match `spacy convert -n`")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out", nargs="*", default=[],
                    help="languages to EXCLUDE from train and dev (the zero-shot arms). Their test "
                         "sets are still written, which is the point.")
    a = ap.parse_args()

    for l in a.hold_out:
        if l not in LANGS:
            sys.exit(f"--hold-out: unknown language {l!r}; known: {' '.join(LANGS)}")
    os.makedirs(a.out, exist_ok=True)

    # ---- read and normalise everything first, so the budget is computed on the real token counts
    data = {}
    for lang in LANGS:
        data[lang] = {}
        for split in ("train", "dev", "test"):
            p = src_path(lang, split)
            if not os.path.exists(p):
                sys.exit(f"missing source treebank: {p}")
            data[lang][split] = [normalise(s) for s in read_conllu(p)]

    train_sizes = {l: sum(len(s) for s in data[l]["train"]) for l in LANGS}

    # ---- allocate
    alloc, group_of = {}, {}
    for gname, members in GROUPS.items():
        active = [l for l in members if l not in a.hold_out]
        for l in members:
            group_of[l] = gname
        if not active:
            continue
        alloc.update(allocate(a.budget, {l: train_sizes[l] for l in active}))

    manifest = {"budget_per_group": a.budget, "dev_budget_per_language": a.dev_budget,
                "block": a.block, "seed": a.seed, "held_out": list(a.hold_out),
                "leaky_feats_removed": sorted(LEAKY_FEATS), "languages": {}}
    labels = collections.Counter()
    per_lang_labels = {}

    print(f"{'lang':5} {'group':14} {'available':>10} {'sampled':>9} {'%mix':>6} "
          f"{'sents':>7} {'dev':>7} {'test':>7} {'udep%':>6}")
    total_sampled = 0
    for lang in LANGS:
        held = lang in a.hold_out
        tr, sampled_tr = ([], False) if held else sample_blocks(
            data[lang]["train"], alloc.get(lang), a.block, lang_seed(a.seed, lang))
        dv, sampled_dv = ([], False) if held else sample_blocks(
            data[lang]["dev"], a.dev_budget, a.block, lang_seed(a.seed, lang))
        te = data[lang]["test"]

        for split, sents in (("train", tr), ("dev", dv), ("test", te)):
            if sents:
                write_conllu(os.path.join(a.out, f"{lang}-{split}.conllu"), sents)

        n_tr = sum(len(s) for s in tr)
        total_sampled += n_tr
        c = collections.Counter(f[7] for s in tr for f in s.rows)
        labels.update(c)
        per_lang_labels[lang] = dict(c)
        udep = 100 * c.get("udep", 0) / n_tr if n_tr else 0.0
        manifest["languages"][lang] = {
            "group": group_of[lang], "held_out": held,
            "train_available": train_sizes[lang], "train_sampled": n_tr,
            "train_sents": len(tr), "train_subsampled": sampled_tr,
            "dev_tokens": sum(len(s) for s in dv), "dev_subsampled": sampled_dv,
            "test_tokens": sum(len(s) for s in te), "test_sents": len(te),
            "udep_rate": round(udep, 2), "source": src_path(lang, "train"),
        }
        print(f"{lang:5} {group_of[lang]:14} {train_sizes[lang]:>10} {n_tr:>9} "
              f"{'':>6} {len(tr):>7} {sum(len(s) for s in dv):>7} "
              f"{sum(len(s) for s in te):>7} {udep:>6.2f}")

    for lang in LANGS:
        n = manifest["languages"][lang]["train_sampled"]
        manifest["languages"][lang]["mix_pct"] = round(100 * n / total_sampled, 2) if total_sampled else 0.0

    print()
    print(f"{'group':16} {'tokens':>9} {'% of mix':>9}   (an even seventh is {100/len(GROUPS):.1f} %)")
    for gname, members in GROUPS.items():
        n = sum(manifest["languages"][l]["train_sampled"] for l in members)
        print(f"{gname:16} {n:>9} {100*n/total_sampled if total_sampled else 0:>8.1f} %")
    manifest["groups"] = {g: sum(manifest["languages"][l]["train_sampled"] for l in m)
                          for g, m in GROUPS.items()}
    manifest["total_train_tokens"] = total_sampled
    manifest["labels"] = dict(labels.most_common())
    manifest["labels_per_language"] = per_lang_labels

    print(f"\ntotal train tokens {total_sampled}   labels {len(labels)}")
    print("  " + "  ".join(f"{k}:{v}" for k, v in labels.most_common()))

    # A label the sample never shows the parser cannot be predicted, and `min_action_freq = 1` means
    # a label shown ONCE becomes an action anyway. Say which are on that edge rather than letting a
    # per-label recall of 0.0 turn up in the eval unexplained.
    rare = [k for k, v in labels.items() if v < 20]
    if rare:
        print(f"\n⚠ {len(rare)} labels under 20 tokens in the sample: {' '.join(sorted(rare))}")
        print("  They are learnable actions at min_action_freq = 1 but their test recall will be "
              "noise. Read per-label scores for these against their support, not on their own.")

    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}/ and {a.out}/manifest.json")


if __name__ == "__main__":
    main()
