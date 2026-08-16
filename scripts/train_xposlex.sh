#!/usr/bin/env bash
# Train and score the lzh XPOS-lexicon parser channel against its capacity control.
#
# THE QUESTION. lzh's XPOS is a four-level comma-separated ontology and the parser cannot see it:
# in the arm that ships, the parser runs FIRST and the grafted tagger sits at the end (and
# `package_sud.sh` refuses any pipeline ordered the other way). `sud.LexFieldEmbed.v1` supplies the
# fields from a per-form lexicon that travels inside the model, injected ABOVE the parser's encoder
# by `sud.Tok2VecPlusFeats.v1` -- the injection point `docs/xpos.md` resolved in favour of.
#
# ANSWERED (2026-08-16), NEGATIVELY: fields -0.19 LAS and the whole tag +0.24 against the control,
# on one seed, inside a ~0.5 seed band -- and the lexicon provably carries 0.0000 bits beyond NORM.
# Kept so the measurement is reproducible, not because the arm is worth building. See
# NEGATIVE-RESULTS.md, "XPOS as a parser input, and kanripo vectors, for lzh".
#
# THREE ARMS, and the control is the one that makes the result readable:
#
#     baseline    training_lzh_trad             the released generation's base arm, already built
#     fields      training_lzh_xposlex          one table per XPOS field (品詞 12, semantic 46)
#     control     training_lzh_xposlex_ctl      same columns, rows and parameters, no information
#     whole       training_lzh_xposlex_whole    the undecomposed 118-way tag through one table
#
# ⚠ ONE SEED PROVES NOTHING HERE. lzh's own measured spread between architecturally identical arms
# is ~0.5 LAS (NEGATIVE-RESULTS.md, the la control), which is larger than any delta this channel is
# expected to produce. Run SEEDS="0 1 2" before reading a sign off the table.
#
# Usage:  bash scripts/train_xposlex.sh              # seed 0, all three arms
#         SEEDS="0 1 2" bash scripts/train_xposlex.sh
#         ARMS="fields control" bash scripts/train_xposlex.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_lzh/SUD_Classical_Chinese-Kyoto
S=relabeled_ext.udep_ruled.punct.rulemerged
C=corpus_lzh_trad/lzh_kyoto-sud
SEEDS="${SEEDS:-0}"
ARMS="${ARMS:-fields control whole}"
STAMP () { echo; echo "===== $* ===== $(date '+%H:%M')"; }

STAMP "0/3  lexicons"
# Harvested from TRAIN only, jackknifed K=5. check_lex_embed.py is not optional: a table that fails
# to serialise does not crash, it makes every token <OOV> and the arm scores like its own control.
$PY scripts/build_xpos_lexicon.py "$A/lzh_kyoto-sud-train.$S.conllu" \
    --out models/lzh_xpos_lex.json --k 5 || exit 1
$PY scripts/build_xpos_lexicon.py "$A/lzh_kyoto-sud-train.$S.conllu" \
    --out models/lzh_xpos_whole.json --k 5 --whole || exit 1
$PY scripts/check_lex_embed.py --table models/lzh_xpos_lex.json \
    --train "$A/lzh_kyoto-sud-train.$S.conllu" --test "$A/lzh_kyoto-sud-test.$S.conllu" || exit 1

STAMP "1/3  configs"
$PY scripts/make_xposlex_config.py --variant fields  --out configs/config_lzh_xposlex.cfg
$PY scripts/make_xposlex_config.py --variant control --out configs/config_lzh_xposlex_ctl.cfg
$PY scripts/make_xposlex_config.py --variant whole   --out configs/config_lzh_xposlex_whole.cfg

for seed in $SEEDS; do
  sfx=""; [ "$seed" != "0" ] && sfx="_s$seed"
  for arm in $ARMS; do
    case $arm in
      fields)  cfg=configs/config_lzh_xposlex.cfg;       out=training_lzh_xposlex$sfx ;;
      control) cfg=configs/config_lzh_xposlex_ctl.cfg;   out=training_lzh_xposlex_ctl$sfx ;;
      whole)   cfg=configs/config_lzh_xposlex_whole.cfg; out=training_lzh_xposlex_whole$sfx ;;
      *) echo "unknown arm $arm"; continue ;;
    esac
    # ⚠ THE LOG NAME IS COUPLED TO --output, and not by preference. `.claude/statusline.sh` finds a
    # running job by scanning ps for `spacy train`, taking its --output, stripping `training_`, and
    # opening `train_<that>.log`. Name the log after the ARM instead (train_lzh_fields.log for
    # training_lzh_xposlex) and the lookup misses: the status line falls through to a bare
    # "<name> running" segment with no bar, no ETA and no best score, for the whole run. Derive it.
    log="train_${out#training_}.log"
    STAMP "2/3  $arm, seed $seed -> $out  (log $log)"
    # `python -u`, and NOT piped through tail: a piped `spacy train` shows nothing until it exits,
    # and two runs have looked stalled for hours because of it. model-last's mtime is the live
    # progress signal.
    $PY -u -m spacy train "$cfg" --code scripts/seg_code.py \
        --output "$out/" --system.seed "$seed" \
        --paths.train "$C-train.$S.spacy" --paths.dev "$C-dev.$S.spacy" \
        > "$log" 2>&1
    [ -d "$out/model-best" ] || { echo "  $arm seed $seed FAILED"; tail -12 "$log"; continue; }
    # --gold-preproc is compulsory for every language but en: without it `spacy evaluate`
    # re-tokenises and the alignment collapses (Korean LAS once fell to ~30).
    $PY -m spacy evaluate "$out/model-best" "$C-test.$S.spacy" --gold-preproc \
        --code scripts/seg_code.py --output "metrics_lzh_${arm}${sfx}_gp.json" 2>&1 | tail -3
  done
done

STAMP "3/3  summary"
$PY - <<'PYEOF'
import glob, json, pathlib
rows = []
for p in sorted(glob.glob("metrics_lzh_*_gp.json")) + ["metrics_lzh_trad_base_gp.json"]:
    f = pathlib.Path(p)
    if not f.exists():
        continue
    m = json.loads(f.read_text())
    rows.append((f.stem, m.get("tag_acc"), m.get("dep_uas"), m.get("dep_las")))
w = max((len(r[0]) for r in rows), default=10)
print(f"  {'arm'.ljust(w)}   TAG      UAS      LAS")
for n, t, u, l in rows:
    fmt = lambda x: f"{x * 100:6.2f}" if isinstance(x, float) else "     -"
    print(f"  {n.ljust(w)}  {fmt(t)}  {fmt(u)}  {fmt(l)}")
PYEOF
