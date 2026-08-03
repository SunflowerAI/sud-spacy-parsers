#!/bin/bash
# Package the SUD-MISC-equipped wheels: the lemma arms plus SUD's own MISC layer
# (Idiom / InIdiom / Subject / Reported) on Token._.sud_misc.
#
# WHICH ARM PER LANGUAGE IS AN EMPIRICAL CHOICE, not a uniform recipe. Measured end-to-end on
# test (scripts/eval_sud_subject.py; gold tokens, everything else predicted), `Subject` F is:
#
#     lang   trained   rule     ships      n(test)
#     en      80.0     63.9     trained      266
#     fa      89.5     71.6     trained       38
#     la      66.3     53.0     trained      674
#     yue     66.7     36.4     trained        6   (n=6 -- not meaningful either way)
#     lzh     59.0     80.7     RULE         174
#     zh      27.7     31.6     rule          302  (both weak; see CLAUDE.md)
#     sa      10.5     12.5     NEITHER        14  (142 train instances -- too sparse to ship)
#
# The split is not arbitrary: Classical Chinese raising rides on a handful of verbs (可/能/欲), which
# a 7-entry frame table captures and a small neural encoder cannot beat; English, Persian and Latin
# raising has a long lexical tail, where the table's recall is fine but its precision collapses.
#
# The idiom layer is deterministic everywhere and needs no training -- it is added to the seven arms
# whose treebanks annotate idioms (en/lzh/ja/fa/ar/la/sa). zh/yue/ko/id carry none.
#
# ORDERING: sud_* pipes go LAST, after clause_parser on lzh/sa. clause_parser reassigns every head
# and deprel, so sud_idiom (which reads `unk`) has to see the tree it leaves behind.
#
# Usage: bash scripts/package_sud.sh en ar fa ja id ko la zh yue lzh sa
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python
CODE_BASE="scripts/sud_misc.py,scripts/sud_idiom.py"
# arms that also ship the Reported rule (en/ar/sa -- see the table below)
CODE_REP="$CODE_BASE,scripts/sud_reported_data.py,scripts/sud_reported_rule.py"

pkg() {  # $1=lang  $2=src model dir  $3=--name value  $4=comma-separated --code files (no flag)
  local lang=$1 src=$2 name=$3 code=""
  [ -n "$4" ] && code="--code $4"
  if [ ! -d "$src" ]; then echo "  $lang: SRC $src missing — skip"; return; fi
  rm -rf build_sud/$lang && mkdir -p build_sud/$lang
  $PY -m spacy package "$src" build_sud/$lang --name "$name" --version 0.1.0 $code \
    --build wheel --force >build_sud/$lang.log 2>&1
  local whl=$(find build_sud/$lang -name '*.whl')
  echo "  $lang -> ${whl:-FAILED}"
  [ -z "$whl" ] && tail -8 build_sud/$lang.log
}

# add_idiom <in> <out> -- deterministic Idiom=Yes / InIdiom=Yes, last in the pipeline
add_idiom() { $PY scripts/add_sud_idiom.py "$1" "$2" >/dev/null 2>&1; }

for lang in "$@"; do
  # Base arm: the trained SUD arm where it won, else the released lemma arm.
  case $lang in
    en|fa|la|yue) base=training_${lang}_sud/model-best ;;
    # sa ships the JOINT MULTI-TASK arm: ONE shared encoder for tagger + parser + morphologizer +
    # lemmatizer, instead of the three-encoder freeze recipe every other arm uses. 25.85 -> 19.16 MB
    # (-25.9 %), tag/pos/morph/lemma each +0.3 to +0.7, and on HELD-OUT UFAL (classical prose, the
    # actual use case) LAS 0.3873 -> 0.4163 / UAS 0.5685 -> 0.6199. It costs Vedic LAS 0.5470 ->
    # 0.5140, accepted by user decision because the target is classical, not Vedic. NB the UFAL
    # figure rests on 416 tokens; the Vedic one on 18 k, so the cost is better measured than the gain.
    sa)           base=training_sa_multitask/model-best ;;
    # ko ships the EOJEOL arm, trained on the ORIGINAL SUD_Korean-GSD with spaCy's rule tokeniser
    # instead of mecab morphemes. The point is tokenisation fidelity: against that treebank the
    # shipped tokeniser now scores TOK 99.77, where the morpheme arm scored 0.3070 (strict span
    # match: rule 0.9522 vs morphemes 0.3070). Trained with `sud.GoldTokCorpus.v1`, so sentence
    # boundaries are LEARNED — raw SENT F 83.80 and raw LAS 55.00, against SENT F 0.00 / LAS 47.15
    # for the same arm under plain gold_preproc.
    # COST, accepted by user decision: this discards the Korean case-particle relabel result
    # (`comp:obl` F 0.169 -> 0.386), which depended on the morpheme split putting relations on the
    # particle. Eojeol tokens fuse noun+particle, so that signal has nowhere to live.
    # CAVEAT: the original treebank populates FEATS on only 4.7 % of tokens, so this arm's
    # `morph_acc` 95.36 is ~the base rate for predicting empty and says nothing. POS 83.05 and
    # lemma 78.30 are real.
    ko)           base=training_ko_eojeol_lemma/model-best ;;
    *)            base=training_${lang}_lemma/model-best ;;
  esac
  work=build_sud/work_$lang
  rm -rf "$work" && mkdir -p build_sud

case $lang in
  en)  $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang en >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg en  "$work" sud_ewt   "$CODE_REP,scripts/sud_tagger.py" ;;
       # fa/la ship NO Reported layer. fa's structural arm does beat its rule (F 40.0 vs 23.5,
       # because fa's gold is 87% LLM-decided and the rule can only reach recall 0.13) -- but at
       # P 0.50 half of what it emits is wrong, which is not worth shipping. la is worse still:
       # F 17.7 by rule, 0.0 trained, a four-deep chain of predicted lemma/deprel/VerbForm/Mood.
       # Both keep the Subject layer and the idiom layer.
  fa)  $PY scripts/add_sud_idiom.py "$base" "$work" --drop sud_reported >/dev/null 2>&1
       pkg fa  "$work" sud_perdt "$CODE_BASE,scripts/sud_tagger.py" ;;
       # la additionally ships `la_macronise` IN THE PIPELINE, with NO lookup table (--no-lut): the
       # vowel lengths are Morpheus-derived (CC BY-SA 3.0 US) and this wheel is CC BY-NC-SA, so the
       # data cannot travel with it -- but the COMPONENT can, and it starts macronising the moment
       # the user runs `fetch_morpheus()`. Until then it passes every token through unchanged and
       # warns once (require_data=False; see la_macronise.py on why a default-pipeline component
       # must not raise). Added AFTER sud_idiom so it lands last: it only writes `token._.macron`,
       # reads nothing sud_idiom writes, and nothing downstream reads it -- so last is where it can
       # neither disturb nor be disturbed. See NOTICE.md.
  la)  $PY scripts/add_sud_idiom.py "$base" "$work.idiom" --drop sud_reported >/dev/null 2>&1
       $PY scripts/add_la_macronise.py "$work.idiom" "$work" --no-lut \
            --code sud_tagger.py,sud_misc.py,sud_idiom.py,sud_subject_frames.py,sud_subject_rule.py \
            >/dev/null 2>&1
       pkg la  "$work" sud_ittb_proiel_perseus \
            "$CODE_BASE,scripts/sud_tagger.py,scripts/la_macronise.py" ;;
  ar)  $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang ar >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg ar  "$work" sud_padt  "$CODE_REP,scripts/ar_tokenizer.py" ;;
  ja)  add_idiom "$base" "$work"
       pkg ja  "$work" sud_gsd   "$CODE_BASE" ;;
       # zh ships NO Subject layer: trained F 27.7 / rule 31.6 on test, both too weak to be worth
       # emitting -- an annotation wrong two times in three is worse than none. Chinese raising is
       # marked by 是/被/了 constructions the frame table cannot separate from ordinary comp:obj.
       # (zh also annotates no idioms, so it gets no SUD MISC layer at all.)
       # zh packages from build_zh_charseg, NOT from the lemma arm directly: the raw-text tokeniser
       # is the treebank-trained character segmenter, swapped in post-hoc (training reads through
       # sud.GoldTokCorpus.v1, so it is segmenter-agnostic and needs no retrain). The segmenter has
       # two channels -- the jackknifed corpus word list and jieba's own segmentation decision --
       # so the wheel REQUIRES jieba, declared in meta.json by bundle_zh_charseg.py.
  zh)  $PY scripts/bundle_zh_charseg.py --out "$work" >/dev/null 2>&1
       pkg zh  "$work" sud_gsd_simp_trad \
            "scripts/char_seg_tokenizer.py,scripts/sa_presegment.py,scripts/sa_presegment_lex.py,scripts/zh_jieba_feature.py" ;;
       # lzh DOES ship the frame rule (F 80.7 vs 59.0 trained -- 可/能/欲 carry it).
  lzh) $PY scripts/add_clause_parser.py "$base" "$work.seg" >/dev/null 2>&1
       $PY scripts/add_sud_subject_rule.py "$work.seg" "$work.rule" --lang lzh >/dev/null 2>&1
       add_idiom "$work.rule" "$work"
       pkg lzh "$work" sud_kyoto \
            "$CODE_BASE,scripts/lzh_tokenizer.py,scripts/clause_parser.py,scripts/sud_subject_rule.py,scripts/sud_subject_frames.py" ;;
       # sa: Subject is too sparse to ship (142 train / 14 test); the idiom layer still applies.
       # sa_compound must stay FIRST (the encoder reads MORPH); clause_parser before sud_idiom.
       # sa: the whole front end (CSLiser + de-CSLizer + de-sandhifier + Devanagari rendering)
       # is assembled by add_sa_frontend.py, which also inserts sa_compound / clause_parser /
       # sa_deva in their required positions. CSL is an INTERNAL representation only — the wheel
       # takes raw IAST or Devanagari.
  sa)  $PY scripts/add_sa_frontend.py "$base" "$work.front" \
            --csliser models/sa_presegment_ortho \
            --unsandhi training_sa_mwt_unsandhi/model-best >/dev/null 2>&1
       $PY scripts/add_sud_reported_rule.py "$work.front" "$work.rep" --lang sa >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg sa  "$work" sud_vedic_ufal_dcs \
            "$CODE_REP,scripts/sa_tokenizer.py,scripts/clause_parser.py,scripts/sa_presegment.py,scripts/sud_unsandhi.py,scripts/sud_affix_embed.py,scripts/sa_devanagari.py" ;;
  yue) $PY scripts/bundle_yue_pkuseg.py --src "$base" --out "$work.pkuseg" >/dev/null 2>&1
       pkg yue "$work.pkuseg" sud_hk \
            "$CODE_BASE,scripts/yue_tokenizer.py,scripts/sud_tagger.py" ;;
       # id/ko annotate none of the four keys, so they are unchanged from package_lemma.sh.
       # id packages from build_id_charseg, NOT from the generic training_id_lemma fallback. The
       # released tokeniser is the treebank-trained character segmenter with the enclitics SPLIT
       # (`-nya` gets its own mod@poss), which lives in the training_id_split_* chain; the plain
       # `base` above still points at the older COARSENED arm, whose tokenizer is spacy.Tokenizer.v1.
       # Pointing at the wrong dir is exactly how the v0.1.0 id wheel shipped a generation stale
       # while CLAUDE.md described the split arm as released -- audited 2026-08-04.
  id)  $PY scripts/add_id_lemma_case_fix.py build_id_charseg "$work" >/dev/null 2>&1
       pkg id  "$work" sud_gsd \
            "scripts/id_lemma_case_fix.py,scripts/char_seg_tokenizer.py,scripts/sa_presegment.py" ;;
  # (ko takes no --code at all)
  ko)  pkg ko  "$base" sud_gsd "" ;;
  *) echo "  unknown lang: $lang" ;;
esac
done
echo "Wheels in build_sud/*/dist/. Upload with:"
echo "  gh release upload v0.1.0 \$(find build_sud -name '*.whl') --clobber"
