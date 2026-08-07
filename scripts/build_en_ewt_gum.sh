#!/usr/bin/env bash
# A SECOND English wheel: EWT + the non-NonCommercial part of GUM. NOT RUN YET -- deliberately.
#
# LICENSING, and why this is a separate wheel rather than a change to en_sud_ewt.
# GUM's LICENSE says two things that are in tension for a filtered subset. Its first line: "The
# treebank is licensed under CC BY-NC-SA 4.0." Its explanation: the NC comes specifically from the
# wikiHow and fiction texts, and GUM "is made available under a Creative Commons license in keeping
# with the underlying texts". The second supports filtering; the first, read strictly, offers the
# ANNOTATIONS under NC whatever the document -- and annotations are what a trained model absorbs.
# Pending word from the GUM maintainers, this wheel therefore ships CC BY-NC-SA 4.0 regardless of the
# filter, and `en_sud_ewt` stays CC BY-SA 4.0 and EWT-only. Two wheels, two licences, no ambiguity
# inherited by anyone downstream.
#
# WHAT THE FILTER KEEPS. 15 genres at ~16-18k tokens each; exactly two are NC:
#     drop  fiction 17,501 + whow 17,075 = 34,576 tokens (13.5 %)
#     keep  222,163 tokens across bio interview conversation news academic essay court vlog
#           speech textbook voyage letter podcast   -> +87 % on EWT's 254,820
#
# THE PERSEUS TRAP DOES NOT APPLY. Latin needed `blank_perseus_xpos.py` because three treebanks used
# mutually incompatible XPOS tagsets and the sparse one tanked combined TAG and LAS. Measured here:
# GUM's 46 XPOS tags are a strict SUBSET of EWT's 49 (the extras -- ADD, AFX, NFP -- are web-text
# artefacts absent from GUM by nature), and GUM adds exactly one DEPREL, udep@desc. Nothing to blank.
#
# THE COST IS THE RELABEL. 17,406 GUM tokens carry `udep` (8.69 %), and every derived English
# annotation was built on EWT alone: the udep relabel, the comp:pred fix for appearance verbs, and
# the SUD MISC layers. All of it has to be re-run over GUM before the two can be trained together.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G=assets/SUD_English-GUM
OUT=assets_gum_ok
NC_GENRES="fiction whow"

echo "=== 1. filter the NonCommercial genres ======================================"
mkdir -p "$OUT"
for s in train dev test; do
  $PY - "$G/en_gum-sud-$s.conllu" "$OUT/en_gum_ok-sud-$s.conllu" "$NC_GENRES" <<'PYEOF'
import sys, re, pathlib
src, dst, nc = sys.argv[1], sys.argv[2], set(sys.argv[3].split())
keep, cur, out, dropped = True, None, [], 0
for ln in pathlib.Path(src).open(encoding="utf-8"):
    m = re.match(r"# newdoc id = GUM_([a-z]+)_", ln)
    if m:
        cur = m.group(1); keep = cur not in nc
        if not keep: dropped += 1
    if keep: out.append(ln)
pathlib.Path(dst).write_text("".join(out), encoding="utf-8")
print(f"  {dst}: dropped {dropped} NC documents")
PYEOF
done

echo "=== 2. re-run every EN-derived annotation over GUM =========================="
# Order matters and mirrors how EWT was built. Each step is a separate pass because each has its own
# failure mode; do not collapse them.
#   a. normalise_reparandum.py   UD `reparandum` -> SUD `conj:dicto`, DEPREL column only
#   b. relabel_ext.py            the udep -> comp:obl / mod decision, extended scope. Rule first,
#                                LLM only on the genuinely ambiguous remainder, resumable through
#                                relabel_cache_ext_en.jsonl -- so a crash costs no queries.
#   c. fix_seem_like_comppred.py predicative like/as-if under appearance verbs -> comp:pred. This is
#                                ENGLISH-SPECIFIC and was checked against ten other languages before
#                                being applied; GUM needs it for the same reason EWT did.
#   d. apply_udep_rules.py       commit whatever the residue audit finds dominated past 90 %
echo "  (steps a-d: see the commands in this script's comments; each writes a new suffix)"

echo "=== 3. merge, train, and evaluate APPLES TO APPLES =========================="
# Merge train->train, dev->dev, test->test, as add_perseus_la.sh does.
#
# THE EVALUATION THAT MATTERS is the EWT-ONLY test set, not the combined one. Latin's combined
# headline (LAS 73.9) looked like a regression purely because the test set had changed; on the
# original ITTB+PROIEL test the addition IMPROVED things (77.7 -> 78.3). Report both, lead with the
# EWT-only figure, and never compare a combined-test number against an EWT-test number.
echo "  then: retrain_seg.sh -> train_morph.sh -> train_lemma.sh -> train_sud.sh"

echo "=== 4. package as a SEPARATE wheel =========================================="
# --name sud_ewt_gum, so `en_sud_ewt` is untouched and users choose deliberately. stamp_model_meta.py
# must carry CC BY-NC-SA 4.0 and cite GUM's own attribution requirement: the LICENSE asks that the
# sources of the texts be cited and the GUM annotators credited.
echo "  VERSION=0.3.0 LICENSE='CC BY-NC-SA 4.0' bash scripts/package_sud.sh en   # with --name sud_ewt_gum"
