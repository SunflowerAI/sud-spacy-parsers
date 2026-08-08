#!/usr/bin/env bash
# A SECOND English wheel: EWT + the non-NonCommercial part of GUM.
#
# LICENSING, and why this is a separate wheel rather than a change to en_sud_ewt.
# GUM's LICENSE says two things that are in tension for a filtered subset. Its first line: "The
# treebank is licensed under CC BY-NC-SA 4.0." Its explanation: the NC comes from the individual
# text sources, and GUM "is made available under a Creative Commons license in keeping with the
# underlying texts". The second supports filtering; the first, read strictly, offers the
# ANNOTATIONS under NC whatever the document -- and annotations are what a trained model absorbs.
# Pending word from the GUM maintainers, this wheel therefore ships CC BY-NC-SA 4.0 regardless of
# the filter, and `en_sud_ewt` stays CC BY-SA 4.0 and EWT-only. Two wheels, two licences, no
# ambiguity inherited by anyone downstream.
#
# WHAT THE FILTER KEEPS. 15 genres (`reddit` is not here -- SUD ships it separately as GUMReddit,
# text-free). GUM's LICENSE.md puts FIVE of the 15 under CC BY-NC-SA, not two:
#     drop  essay (4.0) fiction (3.0) letter (4.0) podcast (4.0) whow (3.0)
#             -> train 64,477 tokens (32.2 %), dev 9,687, test 10,088
#     keep  academic bio conversation court interview news speech textbook vlog voyage
#             -> train 135,746 -> combined 340,324, i.e. +66 % on EWT's 204,578
# NB two of the kept genres are ShareAlike (bio, voyage -- Wikipedia/Wikivoyage CC BY-SA 3.0) and
# the political speeches carry no explicit licence beyond being government/UN public-domain
# material. That affects ATTRIBUTION, not the filter, since the wheel ships NC either way.
#
# THE RELABEL IS FREE, IF THE ORDER IS RIGHT. An earlier draft of this script claimed "THE COST IS
# THE RELABEL ... all of it has to be re-run over GUM", and filtered the NC genres in step 1. Both
# were wrong, and the second one is what made the first true. The original development corpus WAS
# EWT+GUM concatenated, so relabel_cache.jsonl and relabel_cache_ext_en.jsonl -- both tracked --
# already hold every GUM decision. Their keys are POSITIONAL (`path|sentence_index|token_id`), and
# the highest index in each split is exactly EWT+GUM minus one: train 23,855 = 12,544 + 11,314 - 1,
# dev 3,575 = 2,001 + 1,575 - 1, test 3,540 = 2,077 + 1,464 - 1.
# So: relabel the UNFILTERED EWT-then-GUM concatenation first, and filter LAST. Dropping documents
# before the relabel shifts every later sentence index and discards roughly half the GUM cache,
# turning a free step into ~12-17k LLM queries. Step 2 refuses to proceed if that has happened.
#
# THE PERSEUS TRAP DOES NOT APPLY. Latin needed `blank_perseus_xpos.py` because three treebanks
# used mutually incompatible XPOS tagsets and the sparse one tanked combined TAG and LAS. Measured
# here: GUM's 46 XPOS tags are a strict SUBSET of EWT's 49 (the extras -- ADD, AFX, NFP -- are
# web-text artefacts absent from GUM by nature), and GUM adds exactly one DEPREL, udep@desc, on one
# token of train. Nothing to blank.
#
# Usage:  bash scripts/build_en_ewt_gum.sh [merge|relabel|fix|verify|filter|corpus|all]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=.venv/bin/python
E=assets/SUD_English-EWT
G=assets/SUD_English-GUM
SPLITS="train dev test"
NC_GENRES="essay fiction letter podcast whow"
# The merged, UNFILTERED concatenation. These paths are load-bearing: they are the literal keys in
# relabel_cache*.jsonl, so relabel_ext.py's EN_FILES must name exactly these and nothing else.
MERGED=assets/en_sud            # -> assets/en_sud-<split>.conllu
FILTERED=assets/en_ewtgum-sud   # -> assets/en_ewtgum-sud-<split>.relabeled_ext.conllu
CORPUS=corpus_en_gum_ext

step="${1:-all}"
run () { [ "$step" = all ] || [ "$step" = "$1" ]; }
die () { echo "FATAL: $*" >&2; exit 1; }

# --- 1. merge, UNFILTERED, EWT first ------------------------------------------------------------
if run merge; then
  echo "=== 1. merge EWT + GUM (unfiltered, EWT first -- do not reorder) ==============="
  for s in $SPLITS; do
    [ -s "$E/en_ewt-sud-$s.conllu" ] || die "missing $E/en_ewt-sud-$s.conllu (extract SUD_English-EWT.tgz)"
    [ -s "$G/en_gum-sud-$s.conllu" ] || die "missing $G/en_gum-sud-$s.conllu (extract SUD_English-GUM.tgz)"
    cat "$E/en_ewt-sud-$s.conllu" "$G/en_gum-sud-$s.conllu" > "$MERGED-$s.conllu"
    n=$(grep -c '^# sent_id' "$MERGED-$s.conllu")
    t=$(grep -cE '^[0-9]+\s' "$MERGED-$s.conllu")
    printf "  %-28s %6d sents  %7d tokens\n" "$MERGED-$s.conllu" "$n" "$t"
  done
  # The cache keys are sentence INDICES, so the counts are the contract, not a nicety.
  for pair in "train 23858" "dev 3576" "test 3541"; do
    set -- $pair
    got=$(grep -c '^# sent_id' "$MERGED-$1.conllu")
    [ "$got" = "$2" ] || die "$1 has $got sentences, expected $2 -- the cache keys will not line up"
  done
  echo "  sentence counts match the cache's index range."
fi

# --- 2. relabel: dry run as a hard gate, then the real pass ------------------------------------
if run relabel; then
  echo "=== 2. udep -> comp:obl / mod, extended scope (expect ZERO model calls) ========"
  # --rules-only makes no calls and seeds the cache exactly as the real pass does, so its
  # needs_model column IS the query bill the real pass would run up. Sum the per-bucket rows.
  $PY scripts/relabel_ext.py --lang en --rules-only 2>&1 | tee /tmp/en_gum_relabel_dryrun.log
  need=$(awk '/^[a-z_]+ +[0-9]+ +[0-9]+ +[0-9]+ +[0-9]+ +[0-9]+$/ {s+=$6} END {print s+0}' \
           /tmp/en_gum_relabel_dryrun.log)
  if [ "$need" != 0 ]; then
    die "$need targets would need the LLM. The cache is not lining up -- check that step 1 ran
     unfiltered and EWT-first, and that relabel_ext.EN_FILES names $MERGED-<split>.conllu.
     Do NOT let this pass run; it would spend hours re-deciding what is already decided."
  fi
  echo "  dry run: 0 targets need the model. Running for real."
  $PY scripts/relabel_ext.py --lang en || die "relabel_ext failed"
fi

# --- 3. the two English-specific annotation fixes ----------------------------------------------
if run fix; then
  echo "=== 3. comp:pred for appearance verbs, and reparandum -> conj:dicto ============"
  # Both key on invariants (FORM/UPOS/head-lemma/children; the DEPREL cell alone), so they are
  # order-independent and idempotent -- which is why they can run after the relabel.
  $PY scripts/fix_seem_like_comppred.py $(for s in $SPLITS; do echo "$MERGED-$s.relabeled_ext.conllu"; done)
  $PY scripts/normalise_reparandum.py   $(for s in $SPLITS; do echo "$MERGED-$s.relabeled_ext.conllu"; done)
fi

# --- 4. verify the EWT half against the PUBLISHED EWT-only artefact -----------------------------
if run verify; then
  echo "=== 4. the EWT half must reproduce the published en_ewt-sud-*.relabeled_ext ===="
  # The strongest available check that the merged rebuild is faithful: the first N sentences of the
  # merged relabelled file ARE EWT, and EWT was published on its own after exactly these steps. So
  # the two must agree cell for cell -- with ONE intended exception.
  #
  # `reparandum` -> `conj:dicto` is EXPECTED here and is a small correction to the released English
  # model, not a regression. normalise_reparandum.py was run over la/yue/zh but never over en, so
  # the published en_ewt files still carry UD's bare `reparandum` on 36/9/4 tokens and the shipped
  # en_sud_ewt can emit it. GUM makes the case 20x larger (680 in GUM train against EWT's 36), so
  # the new arm normalises. Any OTHER differing cell means the rebuild is not faithful -- stop.
  for s in $SPLITS; do
    ref="assets/en_ewt-sud-$s.relabeled_ext.conllu"
    [ -s "$ref" ] || { echo "  (no $ref -- skipping)"; continue; }
    nref=$(grep -c '^# sent_id' "$ref")
    $PY - "$MERGED-$s.relabeled_ext.conllu" "$ref" "$nref" <<'PYEOF' || die "unexpected divergence"
import sys
from collections import Counter
new, ref, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
EXPECTED = "reparandum->conj:dicto"

def blocks(p):
    buf = []
    for ln in open(p, encoding="utf-8"):
        if ln.strip() == "":
            if buf: yield "".join(buf); buf = []
        else: buf.append(ln)
    if buf: yield "".join(buf)

head = [b for _, b in zip(range(n), blocks(new))]
gold = list(blocks(ref))
cells, ndiff = Counter(), 0
for a, b in zip(head, gold):
    if a == b:
        continue
    ndiff += 1
    for la, lb in zip(a.split("\n"), b.split("\n")):
        if la == lb:
            continue
        ca, cb = la.split("\t"), lb.split("\t")
        if len(ca) != 10 or len(cb) != 10:
            cells["non-token line"] += 1
            continue
        cols = [i + 1 for i in range(10) if ca[i] != cb[i]]
        cells[EXPECTED if cols == [8] and cb[7] == "reparandum" and ca[7] == "conj:dicto"
              else f"col{cols} {cb[7]}->{ca[7]}"] += 1
unexpected = {k: v for k, v in cells.items() if k != EXPECTED}
print(f"  {new}: EWT half {len(gold)-ndiff}/{len(gold)} blocks identical; "
      f"{cells[EXPECTED]} expected {EXPECTED}"
      + (f"; UNEXPECTED {unexpected}" if unexpected else ""))
sys.exit(1 if unexpected else 0)
PYEOF
  done
  echo "  the EWT half reproduces the published artefact (bar the intended reparandum fix)."
fi

# --- 5. filter the NonCommercial genres, LAST ---------------------------------------------------
if run filter; then
  echo "=== 5. drop the NonCommercial GUM genres ($NC_GENRES) ==========================="
  for s in $SPLITS; do
    $PY - "$MERGED-$s.relabeled_ext.conllu" "$FILTERED-$s.relabeled_ext.conllu" "$NC_GENRES" <<'PYEOF'
import sys, re, pathlib
from collections import Counter
src, dst, nc = sys.argv[1], sys.argv[2], set(sys.argv[3].split())
keep, out, kept, dropped = True, [], Counter(), Counter()
genre = "EWT"
for ln in pathlib.Path(src).open(encoding="utf-8"):
    m = re.match(r"# newdoc id = GUM_([a-z]+)_", ln)
    if m:
        genre = m.group(1); keep = genre not in nc
    elif ln.startswith("# newdoc"):
        genre, keep = "EWT", True
    if keep:
        out.append(ln)
        if re.match(r"^\d+\t", ln): kept[genre] += 1
    elif re.match(r"^\d+\t", ln):
        dropped[genre] += 1
pathlib.Path(dst).write_text("".join(out), encoding="utf-8")
print(f"  {dst}")
print(f"    kept    {sum(kept.values()):7d} tokens  " + " ".join(f"{g}:{n}" for g, n in sorted(kept.items())))
print(f"    dropped {sum(dropped.values()):7d} tokens  " + " ".join(f"{g}:{n}" for g, n in sorted(dropped.items())))
PYEOF
  done
  echo "  expected kept tokens: train 340324, dev 43580, test 43403"
fi

# --- 6. convert to .spacy -----------------------------------------------------------------------
if run corpus; then
  echo "=== 6. convert to $CORPUS/ ====================================================="
  mkdir -p "$CORPUS"
  for s in $SPLITS; do
    $PY -m spacy convert "$FILTERED-$s.relabeled_ext.conllu" "$CORPUS/" --converter conllu -n 10 \
      || die "convert failed on $s"
  done
  ls -la "$CORPUS/"
fi

echo
echo "next: 200-step GPU-vs-CPU probe, then base -> train_morph.sh -> train_lemma.sh -> train_sud.sh"
echo "      (all under the en_gum arm names), then package_sud.sh en_gum at CC BY-NC-SA 4.0."
