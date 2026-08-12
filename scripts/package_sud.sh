#!/bin/bash
# Package the SUD-annotation-equipped wheels: the lemma arms plus SUD's own layer
# (Idiom / InIdiom / Subject / Reported / Shared) on Token._.sud_misc.
#
# `Shared` is the one key the treebanks put in FEATS rather than MISC, so the released
# morphologisers have been predicting it all along inside their FEATS bundles -- badly, because a
# local encoder over word forms cannot see the coordination the feature is about. A pipe that takes
# it over therefore has to BEAT that, not merely exist, and it deletes the morphologiser's value
# from `token.morph` when it ships (clear_morph) so the wheel has one answer rather than two.
#
# Test, end to end over gold tokens (scripts/eval_sud_shared.py). "mask" is the share of gold that
# the coordination candidate mask reaches on a PREDICTED parse -- a ceiling on rule and trained
# alike, and the single number that predicts the whole table:
#
#     lang   mask   morph   rule   trained   ships
#     fa     80.2    27.1   58.3     67.7    trained
#     en     70.6    24.7   55.1     62.6    trained
#     lzh    65.5    41.3   52.7     58.8    trained
#     ar     60.2    37.8   52.6     54.6    trained
#     id     57.1    36.1   49.1     53.6    trained
#     la     48.8    10.2   36.8     38.1    trained  (on the AUGMENTED base, which is what ships;
#                                                     the union base preferred the rule, 35.9 v 35.1)
#     ko     37.6    11.3   28.6     32.5    NEITHER  (P 40.1)
#     zh     32.7    37.5   29.1     31.5    NEITHER -- the MORPHOLOGISER wins, uniquely
#     yue    28.4     6.7   16.0     21.5    NEITHER  (P 27.7, n=74)
#     sa     17.3     8.6    9.4      3.8    NEITHER
#
# TWO DIFFERENT TESTS, and conflating them is a mistake worth not repeating. Whether to ship
# ANYTHING is a precision question -- an annotation wrong more often than right is worse than none,
# which is what kept `Subject` out of the zh wheel. WHICH ARM to ship, once both clear that, is
# decided on F, as every other choice in this layer is (lzh's Subject rule at 75.8 over 68.8;
# ar/sa/en's Reported rules). An earlier draft applied the precision floor as a tiebreaker and
# shipped la's trained pipe over its higher-F rule; that was wrong on the union base. On the
# AUGMENTED base la now ships the trained pipe on F, which is the same criterion reaching the
# other answer because the base underneath it changed -- re-measure after ANY base retrain.
# Where nothing ships, the morphologiser's FEATS value is LEFT ALONE -- for zh that is the best arm
# available, and for ko/yue/sa it is merely the status quo.
#
# The mask column also explains the failures, and it is a fact about the PARSER, not the language:
# the mask is defined over the coordination, so a treebank whose conjuncts are recovered poorly
# (sa at LAS ~0.51 reaches 17 % of its own gold) starves both arms of the cases they exist for.
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
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}
PY=.venv/bin/python
# Wheel version. This was hardcoded to 0.1.0, so every release that ADDED a layer had to be packaged
# by hand -- and the hand-built path is where the v0.2.0 near-miss came from (a wheel built straight
# from the base arm, shipping the trained `sud_reported` at F 35.0 instead of the rule at 66.7 and
# omitting `sud_idiom` entirely). Overridable so the driver can do those releases itself:
#   VERSION=0.2.1 bash scripts/package_sud.sh en
VERSION="${VERSION:-0.1.0}"
CODE_BASE="scripts/sud_misc.py,scripts/sud_idiom.py"
# arms that also ship the Reported rule (en/ar/sa -- see the table below)
CODE_REP="$CODE_BASE,scripts/sud_reported_data.py,scripts/sud_reported_rule.py"
# every arm carrying a trained sud_shared pipe needs the candidate mask it looks up by name
CODE_SHARED="scripts/sud_shared_data.py"
# Treebank the lzh lemma lookup table is harvested from. It MUST be the same generation of the data
# the lzh arm was trained on -- override when packaging a differently-trained arm, e.g. the
# punctuation-restored chain (.relabeled_ext.udep_ruled.punct.rulemerged.conllu).
LZH_TRAIN_CONLLU="${LZH_TRAIN_CONLLU:-assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu}"

pkg() {  # $1=arm  $2=src model dir  $3=--name value  $4=comma-separated --code files (no flag)
         # $5=the model's own language code, if it differs from the arm name (en_gum -> en)
  local arm=$1 src=$2 name=$3 code="" lang=${5:-$1}
  # Every arm's tagger now reads UPOS+FEATS through `sud.Tok2VecPlusFeats.v1` /
  # `sud.MultiHashEmbedFeats.v1`, so that layer must travel in EVERY wheel or the model will not
  # load. Appended HERE rather than to each per-language list on purpose: ko passes no --code at
  # all, and a list that has to be remembered is a list that gets missed.
  local always="scripts/sud_feats_embed.py"
  if [ -n "$4" ]; then code="--code $4,$always"; else code="--code $always"; fi
  if [ ! -d "$src" ]; then echo "  $arm: SRC $src missing — skip"; return; fi
  # An arm straight out of `spacy train` has an EMPTY license field, and `spacy package` copies it
  # through without complaint -- so a rebuilt arm ships unlicensed unless this runs. Every model
  # here derives from CC BY-SA treebanks (la, and en_gum, from NonCommercial ones), so this is an
  # obligation. --arm keys the licence/sources tables, --lang goes into the meta: en ships TWO
  # wheels at two licences, so keying on the language code alone would flip both.
  $PY scripts/stamp_model_meta.py "$src" --lang "$lang" --arm "$arm" \
    ${DESCRIPTION:+--description "$DESCRIPTION"} \
    >/dev/null || { echo "  $arm: meta stamp FAILED"; return; }
  rm -rf build_sud/$arm && mkdir -p build_sud/$arm
  $PY -m spacy package "$src" build_sud/$arm --name "$name" --version "$VERSION" $code \
    --build wheel --force >build_sud/$arm.log 2>&1
  local whl=$(find build_sud/$arm -name '*.whl')
  echo "  $arm -> ${whl:-FAILED}"
  [ -z "$whl" ] && tail -8 build_sud/$arm.log
}

# add_idiom <in> <out> -- deterministic Idiom=Yes / InIdiom=Yes, last in the pipeline
add_idiom() { $PY scripts/add_sud_idiom.py "$1" "$2" >/dev/null 2>&1; }

for lang in "$@"; do
  # Base arm: the trained SUD arm where it won, else the released lemma arm.
  case $lang in
    # ar/lzh/id joined this list when `Shared` did: their Subject/Reported layers ship as RULES,
    # so before that they had no reason to take the trained arm at all. The unwanted trained pipes
    # are dropped below, so no dead weights travel.
    en|fa|yue|ar|id) base=training_${lang}_sud/model-best ;;
    # en_gum takes the XPOS-NORMALISED arm: EWT and GUM disagreed on punctuation XPOS (EWT tags
    # `;` `,` 101 times of 101, GUM the PTB-standard `:`), so the same token in the same context
    # carried different gold depending on which treebank the sentence came from. EWT's half was
    # converted to GUM's convention and the tagger retrained on the frozen arm -- every other
    # component byte-identical. Headline TAG is flat (0.3 % of the corpus) but accuracy on the
    # affected punctuation goes 72.47 -> 82.98. `en` (EWT-only) is deliberately NOT in this arm:
    # on its own, EWT's convention is internally consistent.
    en_gum)       base=training_en_gum_sud_xpos/model-best ;;
    # la ships the ORTHOGRAPHICALLY AUGMENTED chain, not the plain-plus-macron union: one copy of
    # the macronised treebank resampled into a fresh edition style every epoch (macrons, breves,
    # u/v, i/j, æ/œ, sentence-initial capitals). It costs ~0.5 LAS and 2.7 TAG on ordinary input
    # and collapses the LAS spread ACROSS orthographies from 54.4 to 7.0 -- printed Latin varies on
    # every one of those axes, so the spread is the number that matters. The SUD layer is trained
    # through the SAME augmenter (configs/config_la_aug_sud.cfg), because a Subject pipe reading
    # NORM/PREFIX/SUFFIX/SHAPE off a spelling it never met is the arm's own weak point.
    # It also ships the XPOS-NORMALISED tagger: PROIEL's 23-value and Perseus's (blanked)
    # tagsets are re-rendered as Index Thomisticus codes, so the arm predicts ONE tagset instead
    # of two-and-a-hole. ITTB's own rows are untouched, and on the ITTB test slice -- the one span
    # whose gold never moved -- TAG goes 90.68 -> 92.92, combined 77.61 -> 86.16 with LAS/UAS/POS/
    # LEMMA identical to the decimal. LA_BASE gets back the pre-normalisation arm.
    la)           base="${LA_BASE:-training_la_aug_sud_xpos/model-best}" ;;
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
    # lzh ships the punctuation-restored chain and takes its lemmas from a lookup table, so its
    # base is the MORPH arm (there is no trained lemmatizer above it) and it must be packaged
    # against the treebank generation it was trained on. Both are overridable so a differently
    # trained lzh arm can be packaged without editing this file:
    #   LZH_BASE=training_lzh_lemma/model-best LZH_TRAIN_CONLLU=<...>.relabeled_ext.conllu \
    #     bash scripts/package_sud.sh lzh      # the pre-punctuation arm
    # lzh: the TRADITIONAL-ONLY rule-merged, punctuation-restored arm (no trained lemmatizer --
    # han_lemma_lut replaces it below) PLUS the Shared pipe trained on top of it.
    # ⚠ This default was left at `training_lzh_rm_sud` after lzh went traditional-only END TO END,
    # so `bash scripts/package_sud.sh lzh` silently rebuilt the superseded BOTH-SCRIPTS generation:
    # tok2vec/tagger/parser/morphologizer/sud_shared all differed from the live 0.2.0 asset. Caught
    # in the 2026-08-09 code-only re-release by diffing the rebuilt wheel against the DOWNLOADED
    # asset file by file -- the wheel built, loaded and ran, and nothing else would have said so.
    # Third time this repo has shipped lzh a generation backwards; a default that names the arm is
    # the fix, not a note telling the next person to remember.
    lzh)          base="${LZH_BASE:-training_lzh_trad_sud/model-best}" ;;
    # zh is TRADITIONAL-ONLY end to end, like lzh, and for the same reason -- a both-scripts
    # inventory never pools 個 with 个. Naming the arm here rather than falling through to
    # `training_zh_lemma` is not tidiness: the fall-through is the both-scripts generation, and it
    # is how the id wheel once shipped a generation stale. zh carries no SUD MISC layer, so the
    # lemma arm is the top of its chain.
    zh)           base="${ZH_BASE:-training_zh_trad_lemma/model-best}" ;;
    *)            base=training_${lang}_lemma/model-best ;;
  esac
  # SUD_BASE overrides the arm for THIS run, whatever the language -- used to package the
  # XPOS-downstream arms (scripts/graft_xpos_tagger.py), which are the shipping arms with the
  # tagger regrafted behind the morphologiser. Per-language, so it is set on a one-language call.
  [ -n "$SUD_BASE" ] && base="$SUD_BASE"
  work=build_sud/work_$lang
  # Clear the INTERMEDIATES too ("$work".rep/.idiom/.mac/...), not just $work. `nlp.to_disk` writes
  # the pipes it has and leaves any other subdirectory alone, so a stale `sud_shared/` from a run
  # when that pipe still shipped survives into the next one -- dead weights in the wheel, and a
  # spurious "WEIGHTS CHANGED" from la's --verify, which walks the source tree rather than the
  # pipeline. Found when la switched from the trained pipe to the rule.
  rm -rf "$work" "$work".* && mkdir -p build_sud

case $lang in
  en)  $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang en >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg en  "$work" sud_ewt   "$CODE_REP,$CODE_SHARED,scripts/sud_tagger.py" ;;
       # en_gum: the SECOND English wheel, EWT + the ten non-NonCommercial GUM genres. Same pipe
       # surgery as en, different corpus and a different licence -- CC BY-NC-SA 4.0, keyed off the
       # ARM name so plain `en_sud_ewt` stays CC BY-SA and commercially usable. The two wheels
       # coexist deliberately; users choose. See scripts/build_en_ewt_gum.sh for the data build.
       # ⚠ Which Reported arm ships here is NOT inherited from en: the MISC layer reads the base
       # arm's own predictions, so the rule-vs-trained comparison must be re-run on this arm
       # (eval_sud_reported.py) before this line is trusted.
  en_gum) $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang en >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg en_gum "$work" sud_ewt_gum "$CODE_REP,$CODE_SHARED,scripts/sud_tagger.py" en ;;
       # fa/la ship NO Reported layer. fa's structural arm does beat its rule (F 40.0 vs 23.5,
       # because fa's gold is 87% LLM-decided and the rule can only reach recall 0.13) -- but at
       # P 0.50 half of what it emits is wrong, which is not worth shipping. la is worse still:
       # F 17.7 by rule, 0.0 trained, a four-deep chain of predicted lemma/deprel/VerbForm/Mood.
       # Both keep the Subject layer and the idiom layer.
       # fa NOW SHIPS its Reported layer (the STRUCTURAL trained pipe), reversing the earlier
       # decision. That decision rested on P 0.50 -- "half of what it emits is wrong" -- measured
       # before the annotating_components fix, when tok2vec was missing and every structural pipe
       # trained on a degenerate parse. Retrained: F 46.15 at P 54.55, against its own rule's
       # F 23.53. It stays the one arm where the trained pipe beats the rule for this feature,
       # because fa's Reported gold is 87 % LLM-decided and a rule can only reach the 13 % it
       # committed itself (rule P 1.00, R 0.13).
  fa)  add_idiom "$base" "$work"
       pkg fa  "$work" sud_perdt "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py" ;;
       # la additionally ships `la_macronise` IN THE PIPELINE, with NO lookup table (--no-lut): the
       # vowel lengths are Morpheus-derived (CC BY-SA 3.0 US) and this wheel is CC BY-NC-SA, so the
       # data cannot travel with it -- but the COMPONENT can, and it starts macronising the moment
       # the user runs `fetch_morpheus()`. Until then it passes every token through unchanged and
       # warns once (require_data=False; see la_macronise.py on why a default-pipeline component
       # must not raise). Added AFTER sud_idiom so it lands last: it only writes `token._.macron`,
       # reads nothing sud_idiom writes, and nothing downstream reads it -- so last is where it can
       # neither disturb nor be disturbed. See NOTICE.md.
       # la also swaps in the `-que`-splitting tokeniser (sud.LatinEncliticTokenizer.v1). spaCy's
       # stock la rules split nothing ending in -que, but ITTB and Perseus write `Animosque` fused
       # and analyse it as `Animos` + `que` -- so real classical orthography reached the model with
       # a token boundary missing. Swapped in post hoc, NOT retrained: training reads through
       # sud.GoldTokCorpus.v1 under gold_preproc, so the parser is segmenter-agnostic and every
       # component's weights come out byte-identical (--verify checks). Perseus test, raw
       # end-to-end: TOK 98.25 -> 99.70, UAS 62.97 -> 65.19, LAS 51.31 -> 53.35; ITTB+PROIEL
       # unchanged. It goes LAST because it is the one step that rewrites [nlp.tokenizer] in the
       # config, and it re-verifies the reload rather than trusting `to_disk`.
       # la now ships the TRAINED pipe for Shared, reversing the union arm's decision. On that
       # base the table won narrowly (rule 35.85 v trained 35.10); on the augmented base the
       # trained pipe wins, 38.11 v 36.78, with the morphologiser at 10.23. The reversal is not
       # noise-sized in the direction that matters: this IS the three-feature arm, whose
       # `model-best` is picked on the mean of Subject/Reported/Shared and which therefore
       # handicapped Shared by ~5 points on the union base -- so the handicap runs AGAINST the
       # winner here, and a solo retrain could only widen the margin. The candidate mask also
       # reaches 48.84 % of gold against the union base's 45.0 %, i.e. the augmented parser
       # recovers the coordination this layer is defined over slightly better.
       # Subject stays trained (67.02 v the rule's 52.41). Reported still ships NOWHERE: rule
       # 17.65, trained 8.00, on 24 test instances.
  la)  $PY scripts/add_sud_idiom.py "$base" "$work.idiom" --drop sud_reported >/dev/null 2>&1
       $PY scripts/add_la_macronise.py "$work.idiom" "$work.mac" --no-lut \
            --code sud_tagger.py,sud_misc.py,sud_shared_data.py,sud_shared_frames.py,sud_shared_rule.py,sud_idiom.py,sud_subject_frames.py,sud_subject_rule.py,sud_feats_embed.py \
            >/dev/null 2>&1
       $PY scripts/add_la_enclitic_tokenizer.py "$work.mac" "$work" --verify \
            --code sud_tagger.py,sud_misc.py,sud_shared_data.py,sud_shared_frames.py,sud_shared_rule.py,sud_idiom.py,sud_subject_frames.py,sud_subject_rule.py,la_macronise.py,sud_feats_embed.py \
            || { echo "  la: enclitic tokeniser swap FAILED — skip"; continue; }
       pkg la  "$work" sud_ittb_proiel_perseus \
            "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py,scripts/la_macronise.py,scripts/la_tokenizer.py,scripts/la_enclitics.py" ;;
       # ar now takes the TRAINED arm as its base (for sud_shared); add_sud_reported_rule drops
       # the trained sud_reported it also carries, since ar ships the Reported RULE (73.5 v 46.0).
  ar)  $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang ar >/dev/null 2>&1
       add_idiom "$work.rep" "$work"
       pkg ar  "$work" sud_padt  "$CODE_REP,$CODE_SHARED,scripts/sud_tagger.py,scripts/ar_tokenizer.py" ;;
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
       # add_zh_script LOADS the segmenter from named paths and refuses to write a model that
       # cannot segment. It used to carry the segmenter over from the input tokenizer by guessing
       # attribute names and missed `seg`, so zh_sud_gsd 0.2.0 shipped returning each input string
       # as ONE TOKEN -- it loaded, it parsed, and only the wheel's file list said otherwise.
       # The segmenter is the TRADITIONAL one (models/zh_seg_jbdec_trad, strict token F 0.9242
       # against the simplified arm's 0.9210 on its own test); its jieba channel is asked about the
       # t2s rendering, since jieba's dictionary is simplified and asking it directly costs the
       # channel F 0.9223 -> 0.8920. `bundle_zh_charseg.py` builds the superseded both-scripts
       # CharSegTokenizer wheel and is kept for that.
  zh)  $PY scripts/add_zh_script.py "$base" "$work" \
            --seg models/zh_seg_jbdec_trad --lexicon models/zh_lex_corpus_trad.txt \
            || { echo "  zh: script/segmenter wiring FAILED — skip"; continue; }
       # Vendor the ~6 MB of jieba the BMES channel actually loads and drop the pip requirement.
       # `spacy package` copies the model dir wholesale and setup.py's list_files walks it, so a
       # tree dropped here ships as package_data with no change to the generated setup.py. Must run
       # AFTER add_zh_script (which writes the requirements) and BEFORE pkg (which packages them).
       # Saves 36 MB per install; the wheel grows 6 MB. See scripts/vendor_jieba.py.
       $PY scripts/vendor_jieba.py "$work" \
            || { echo "  zh: jieba vendoring FAILED — skip"; continue; }
       pkg zh  "$work" sud_gsd \
            "scripts/zh_script.py,scripts/char_seg_tokenizer.py,scripts/sa_presegment.py,scripts/sa_presegment_lex.py,scripts/zh_jieba_feature.py" ;;
       # lzh DOES ship the frame rule for Subject (F 80.0 vs 66.2 trained -- 可/能/欲 carry it;
       # measured on the rule-merged arm, with the lemma layer the wheel ships -- see
       # eval_sud_subject.lzh_rule_arm, without which the rule scores a spurious 0.00),
       # and the TRAINED pipe for Shared (58.8 v rule 52.7 v morphologiser 41.3). Both ride on the
       # punctuation-restored, rule-merged arm,
       # so `training_lzh_rm_sud` is that arm plus the Shared pipe -- NOT the pre-punctuation
       # `training_lzh_sud`, whose parse (and therefore whose coordination mask) is a different
       # model's.
       # lzh replaces the TRAINED lemmatizer with a lookup table: the lemma is the form on 99.0 %
       # of tokens and the exceptions are variant characters (異體字), not morphology. Test lemma
       # accuracy 99.733 by table vs 99.649 trained -- the trained layer's errors ARE the variants,
       # which it leaves untouched -- and the arm loses ~1.4 MB. Only lzh: zh (99.900 / 99.904) and
       # yue (99.762 / 99.841) are one token apart either way, so they keep the trained layer.
       # HARVEST FROM THE TREEBANK THE ARM WAS TRAINED ON -- a table built from a different
       # generation of the data silently disagrees with the model's own vocabulary.
  lzh) $PY scripts/han_lemma_lut.py --build "$base" "$work.lut" \
            --conllu "$LZH_TRAIN_CONLLU" >/dev/null 2>&1
       # --keep-marks is COUPLED to the base: worth +2.34 LAS on the punctuation-trained arm and
       # -3.80 on one that has never seen a mark. Drop it if you set LZH_BASE to a pre-punctuation
       # arm. `$LZH_KEEP_MARKS` exists so that can be done without editing this line.
       $PY scripts/add_clause_parser.py "$work.lut" "$work.seg" \
            ${LZH_KEEP_MARKS:---keep-marks} >/dev/null 2>&1
       $PY scripts/add_sud_subject_rule.py "$work.seg" "$work.rule" --lang lzh >/dev/null 2>&1
       $PY scripts/add_sud_idiom.py "$work.rule" "$work" --drop sud_subject >/dev/null 2>&1
       pkg lzh "$work" sud_kyoto \
            "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py,scripts/lzh_tokenizer.py,scripts/clause_parser.py,scripts/han_lemma_lut.py,scripts/sud_subject_rule.py,scripts/sud_subject_frames.py" ;;
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
       # yue ships NO Shared layer: trained F 21.5 at P 27.7 on 74 test tokens, with the candidate
       # mask reaching 28.4 % of gold. The trained pipe is DROPPED rather than left in place, so
       # the wheel carries no weights it never uses -- and so `Shared` keeps coming out of the
       # morphologiser's FEATS, since nothing here is good enough to take it over.
  yue) $PY scripts/drop_pipes.py "$base" "$work.noshared" sud_shared >/dev/null 2>&1
       $PY scripts/bundle_yue_pkuseg.py --src "$work.noshared" --out "$work.pkuseg" >/dev/null 2>&1
       pkg yue "$work.pkuseg" sud_hk \
            "$CODE_BASE,$CODE_SHARED,scripts/yue_tokenizer.py,scripts/sud_tagger.py" ;;
       # id annotates none of Idiom/Subject/Reported, but it DOES annotate Shared, so it now
       # carries a SUD layer for the first time (F 53.6 trained vs 49.1 rule vs 36.1 morph).
       # `base` is training_id_sud, itself sourced from the SPLIT chain (char segmenter, enclitics
       # separated) -- not the older COARSENED training_id_lemma. Pointing at the wrong dir is
       # exactly how the v0.1.0 id wheel shipped a generation stale while CLAUDE.md described the
       # split arm as released (audited 2026-08-04), so the arm is chosen in src_model()/here and
       # not left to a fall-through.
       # The segmenter is NOT in the trained arm: `sud.CharSegTokenizer.v1` builds with no model,
       # so it has to be loaded in again downstream of training (bundle_id_charseg.py, which
       # verifies the RELOAD rather than the in-memory object).
  id)  $PY scripts/bundle_id_charseg.py "$base" "$work.seg" >/dev/null 2>&1 \
            || { echo "  id: charseg bundling FAILED — skip"; continue; }
       $PY scripts/add_id_lemma_case_fix.py "$work.seg" "$work" >/dev/null 2>&1
       pkg id  "$work" sud_gsd \
            "$CODE_SHARED,scripts/sud_misc.py,scripts/sud_tagger.py,scripts/id_lemma_case_fix.py,scripts/char_seg_tokenizer.py,scripts/sa_presegment.py" ;;
       # ko ships NO Shared layer: trained F 32.5 at P 40.1, i.e. wrong three times in five, and
       # the candidate mask reaches only 37.6 % of its own gold. Same call as zh's Subject.
       # (ko takes no --code at all)
  ko)  pkg ko  "$base" sud_gsd "" ;;
  *) echo "  unknown lang: $lang" ;;
esac
done
echo "Wheels in build_sud/*/dist/. Upload BY NAME, one line per wheel:"
echo "  gh release upload v$VERSION build_sud/<arm>/dist/<name>-$VERSION-py3-none-any.whl --clobber"
echo
echo "  Do NOT upload via \$(find build_sud -name '*.whl'): build_sud accumulates a wheel from"
echo "  every arm ever packaged, at every version -- and has twice held two of the SAME name,"
echo "  where --clobber makes the winner whichever one find yielded last. Count them first:"
find build_sud -name "*.whl" 2>/dev/null | sed "s|^|    |"
