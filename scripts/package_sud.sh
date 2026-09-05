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
# The default is the version every wheel on the release CURRENTLY carries. It was left at 0.1.0
# long after all twelve went to 0.2.0, so a bare call built a wheel named a whole generation behind
# and said nothing -- the same shape as a default that names a pre-graft arm, and the same fix: make
# the default right rather than write a note asking the next person to remember.
VERSION="${VERSION:-0.2.0}"
CODE_BASE="scripts/sud_misc.py,scripts/sud_idiom.py"
# arms that also ship the Reported rule (en/ar/sa -- see the table below)
CODE_REP="$CODE_BASE,scripts/sud_reported_data.py,scripts/sud_reported_rule.py"
# every arm carrying a trained sud_shared pipe needs the candidate mask it looks up by name
CODE_SHARED="scripts/sud_shared_data.py"
# Treebank the lzh lemma lookup table is harvested from. It MUST be the same generation of the data
# the lzh arm was trained on. ⚠ THIS DEFAULT NAMED Kyoto-BOTH while lzh is traditional-only end to
# end, so the lookup table would have been harvested from the both-scripts text — the same stale
# default that `train_sud.sh` carried in four places. LZH_TRAIN_CONLLU remains the override for the
# superseded both-scripts arm.
LZH_TRAIN_CONLLU="${LZH_TRAIN_CONLLU:-assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu}"

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
  # ⚠ THE XPOS-DOWNSTREAM GUARD. All twelve v0.2.0 wheels ship the warm-started tagger MOVED behind
  # the morphologiser so it can read UPOS+FEATS (graft_xpos_tagger.py; ar 89.44 -> 89.71, sa +1.52).
  # That release grafted per arm through SUD_BASE and kept no directory, so the defaults named
  # pre-graft arms -- and rebuilding one produces a wheel that builds, loads and parses perfectly
  # while shipping the previous tagger generation. Only a file-by-file diff against the downloaded
  # asset catches that, which is how it was found on ar.
  #
  # STATE OF THE DEFAULTS, re-derived from the arms themselves on 2026-08-17 (do not trust the
  # prose -- run the check): en, yue, id (7eafb16), ar and fa are POST-graft and safe. en_gum was
  # NOT, although an earlier version of this comment exempted it and la on the grounds that a
  # `_sud_xpos` directory had been kept for them; that claim was false for both. en_gum's default
  # now names `training_en_gum_sud_xw`, rebuilt and verified byte-identical to the released asset.
  # la is RESOLVED as of 2026-08-19: `training_la_aug_sud_xpos` is still pre-graft and is no longer
  # the default. `training_la_lemvec_misc` replaces it -- the lemma-vector arm, sealed, with the
  # `training_la_xposwarm` tagger grafted in by `graft_standalone_tagger.py` (parse unchanged
  # 5 527/5 527, donor tags reproduced 5 527/5 527) and the MISC pipes transplanted on top.
  #
  # ⚠ ko IS PRE-GRAFT AND ALWAYS WAS, found on 2026-08-20 when this guard refused a rebuild.
  # `ko_sud_gsd-0.2.0` ships a tagger that is a LISTENER on the shared tok2vec sitting BEFORE the
  # morphologiser — so ko never received the v0.2.0 graft at all, and this script cannot rebuild the
  # wheel that is live. It is not a stale default: there is no post-graft ko arm to name. The fix is
  # a rebuild, not an exemption, and `scripts/build_ko_release.sh graft` is it — the conditioned
  # tagger, warm-started from the arm's OWN tagger rather than the released one (72.51 against
  # 88.60 TAG; starting from the released tagger would throw the analyser channel away). Until that
  # arm is released, KO_BASE still defaults to the pre-graft released chain, and this guard will
  # keep refusing it — which is correct, and is the only reason anyone found out.
  #
  # The pipeline ORDER is the cheap invariant that encodes all of this, so assert it here rather
  # than hope the next person diffs: a wheel whose tagger precedes its morphologiser is pre-graft,
  # whatever else is right about it.
  # sa is the ONE documented exemption, and it is not a loophole: its XPOS is a COPY of UPOS on
  # 100 % of tokens in both halves of its corpus (docs/sanskrit.md), so there is no XPOS tagset to
  # condition on and nothing for graft_xpos_tagger.py to graft. Its tagger sits before the
  # morphologiser because the arm is JOINT MULTI-TASK — one shared encoder feeding all four heads
  # at once, so "before" is a listing order, not a data dependency. Left implicit, this guard
  # refused the RELEASED sa arm too, i.e. sa could not be rebuilt by its own packaging script.
  $PY - "$src" "$arm" <<'GUARD' || { echo "  $arm: PRE-GRAFT arm — refusing to package. Rebuild with scripts/graft_xpos_tagger.py"; return; }
import sys, json, pathlib
cfg = (pathlib.Path(sys.argv[1]) / "meta.json")
pipe = json.loads(cfg.read_text())["pipeline"] if cfg.exists() else []
if sys.argv[2] == "sa":
    sys.exit(0)
if "tagger" in pipe and "morphologizer" in pipe and pipe.index("tagger") < pipe.index("morphologizer"):
    print("    tagger precedes morphologizer: %s" % pipe, file=sys.stderr)
    sys.exit(1)
GUARD
  # ⚠ THE SILENCED-TAGGER GUARD. spaCy's tagger writes only where `token.tag == 0` unless
  # `overwrite` is on, and the stock configs here all carry `overwrite = false`. Harmless for
  # eleven arms -- nothing sets a tag before the tagger runs -- but `spacy.ja.JapaneseTokenizer`
  # assigns `token.tag_` at TOKENISATION, so ja's trained tagger was a NO-OP at inference and users
  # got SudachiPy's raw UniDic tag (0.7673 against gold) where the tagger would have given 0.9457.
  # It shipped in ja_sud_gsd-0.2.0 and gold_preproc could never have caught it: the predicted doc
  # is built from gold words, carries no tag, so the tagger DOES write and tag_acc looked healthy.
  # BEHAVIOURAL, not a list of languages: tokenise an ASCII probe and ask the arm whether its own
  # tokeniser pre-set anything. ja pre-sets 3/3 on "Test 1.", en/ko/zh 0/3 -- so one probe
  # separates them with no per-language table to fall out of date. Fix with
  # scripts/fix_tagger_overwrite.py, which patches config.cfg AND the pipe's serialised cfg (the
  # latter is what from_disk restores, so patching the config alone changes nothing).
  $PY - "$src" <<'GUARD' || { echo "  $arm: SILENCED TAGGER — refusing to package. Fix with scripts/fix_tagger_overwrite.py $src"; return; }
import pathlib, sys
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401  (custom architectures/tokenisers)
import spacy
nlp = spacy.load(sys.argv[1])
if "tagger" not in nlp.pipe_names:
    sys.exit(0)
preset = sum(1 for t in nlp.make_doc("Test 1.") if t.tag != 0)
if preset and not nlp.get_pipe("tagger").cfg.get("overwrite"):
    print("    tokeniser pre-sets tags and tagger.overwrite=false: "
          "the trained tagger is a no-op at inference", file=sys.stderr)
    sys.exit(1)
GUARD
  # ⚠ THE MISSING-`--code` GUARD. A custom registry name in the model's config resolves at LOAD
  # time, so a file left out of `--code` produces a wheel that builds, installs, and raises E893 the
  # first time a user opens it. The build itself cannot catch that -- the training machine has
  # `scripts/` on `sys.path` -- and it has already happened once: `scripts/sud_lemmavec_embed.py`
  # was missing from all three `la` lists, found only by installing into a clean directory.
  # So the config is read back and every `sud.*` name in it is resolved to the module that
  # registers it, which is checked against the list actually being passed. A refusal, not a comment
  # asking the next person to remember (standing hazard 2).
  # ⚠ THE HOST-PATH GUARD. A factory argument naming a file is serialised into the packaged config
  # verbatim, and a relative one resolves against the CWD -- so it works on the build machine and
  # raises FileNotFoundError everywhere else. lzh 0.3.0 shipped exactly that
  # (`tables = "models/lzh_xpos_tables.json"`) and every local verification passed, because they
  # were all run from the repo root. Any config value that looks like a path to a file that is NOT
  # inside the model directory is refused.
  $PY - "$src" <<'PATHGUARD' || { echo "  $arm: config names a HOST PATH — refusing to package"; return; }
import pathlib, sys, re
from thinc.api import Config
src = pathlib.Path(sys.argv[1])
cfg = Config().from_disk(src / "config.cfg", interpolate=False)
bad = []
def walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, f"{path}/{k}")
    elif isinstance(d, str) and v_is_path(d):
        if not (src / d).exists():
            bad.append((path, d))
def v_is_path(v):
    return bool(re.search(r"\.(json|txt|bin|cfg|vec|npz)$", v)) or v.startswith(("/", "./", "../"))
walk(cfg)
for p, v in bad:
    print(f"    {p} = {v!r} is not inside the model directory", file=sys.stderr)
sys.exit(1 if bad else 0)
PATHGUARD

  $PY - "$src" "$code" <<'GUARD' || { echo "  $arm: --code list is INCOMPLETE — refusing to package"; return; }
import pathlib, re, sys
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401  (registers every custom architecture, reader, tokeniser and factory)
from spacy.util import registry

from thinc.api import Config

# ONLY the sections spaCy resolves at LOAD time. `[corpora]`, `[training]` and `[initialize]` name
# readers, augmenters and batchers that exist on the training machine and are never touched by
# `spacy.load`, so requiring them in --code would refuse arms that are perfectly loadable — the
# released ko arm names `sud.GoldTokCorpus.v1` and needs nothing at all.
cfg = Config().from_disk(pathlib.Path(sys.argv[1]) / "config.cfg", interpolate=False)
text = str({k: cfg.get(k) for k in ("nlp", "components")})
listed = {pathlib.Path(p).name for p in sys.argv[2].replace("--code", "").split(",") if p.strip()}
names = set(re.findall(r"'(sud\.[A-Za-z0-9_.]+)'", text))
missing = {}
for name in sorted(names):
    for reg in ("architectures", "tokenizers", "factories", "misc", "layers", "callbacks"):
        try:
            func = getattr(registry, reg).get(name)
        except Exception:
            continue
        # NOT `sys.modules[func.__module__]`: seg_code.py loads each file with `exec_module` and
        # never puts it in sys.modules, so that lookup returns None for every custom name and the
        # guard would pass everything — a check that cannot fail, which is worse than no check.
        f = getattr(func, "__globals__", {}).get("__file__")
        if f:
            missing.setdefault(pathlib.Path(f).name, set()).add(name)
        break
gaps = {m: n for m, n in missing.items() if m not in listed and m != "seg_code.py"}
for m, n in sorted(gaps.items()):
    print(f"    config names {sorted(n)} but scripts/{m} is not in --code", file=sys.stderr)
sys.exit(1 if gaps else 0)
GUARD
  # An arm straight out of `spacy train` has an EMPTY license field, and `spacy package` copies it
  # through without complaint -- so a rebuilt arm ships unlicensed unless this runs. Every model
  # here derives from CC BY-SA treebanks (la and ar, from NonCommercial ones), so this is an
  # obligation. --arm keys the licence/sources tables, --lang goes into the meta: en ships TWO
  # wheels and only en_gum owes GUM's CC BY attribution, so keying on the language code alone
  # would give it to both.
  $PY scripts/stamp_model_meta.py "$src" --lang "$lang" --arm "$arm" \
    ${DESCRIPTION:+--description "$DESCRIPTION"} \
    >/dev/null || { echo "  $arm: meta stamp FAILED"; return; }
  # `spacy package` appends its OWN `spacy>=x,<y` pin. An arm rebuilt FROM A RELEASED WHEEL is
  # already carrying one (the released meta has it), so the packaged wheel comes out declaring
  # spacy twice -- harmless to pip, but a gratuitous diff against the asset the rebuild is meant to
  # reproduce, and the rebuild-from-release path is the only way to repackage zh, ar, en_gum or ko
  # while their post-graft arms are missing. Drop it and let packaging add exactly one.
  $PY - "$src" <<'PIN'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "meta.json"
m = json.loads(p.read_text(encoding="utf-8"))
was = m.get("requirements") or []
now = [r for r in was if r.split(">=")[0].split("==")[0].split("[")[0].strip() != "spacy"]
if now != was:
    m["requirements"] = now
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    dropped spacy pin from meta (packaging re-adds it): {was} -> {now}")
PIN
  rm -rf build_sud/$arm && mkdir -p build_sud/$arm
  $PY -m spacy package "$src" build_sud/$arm --name "$name" --version "$VERSION" $code \
    --build wheel --force >build_sud/$arm.log 2>&1
  local whl=$(find build_sud/$arm -name '*.whl')
  echo "  $arm -> ${whl:-FAILED}"
  [ -z "$whl" ] && tail -8 build_sud/$arm.log
}

# add_idiom <in> <out> -- deterministic Idiom=Yes / InIdiom=Yes, last in the pipeline
add_idiom() { $PY scripts/add_sud_idiom.py "$1" "$2" >/dev/null || {
  echo "  add_sud_idiom FAILED"; return 1; }; }

for lang in "$@"; do
  # Base arm: the trained SUD arm where it won, else the released lemma arm.
  case $lang in
    # en, yue and id all take a trained SUD arm; ar/lzh/id joined them when `Shared` did, their
    # Subject/Reported layers shipping as RULES, so before that they had no reason to take the
    # trained arm at all. The unwanted trained pipes are dropped below, so no dead weights travel.
    # All three name the GRAFTED arm, as every other language does. The v0.2.0 release
    # grafted these three per arm through SUD_BASE and kept no directory, so the default named a
    # base whose tagger predates the graft and repackaging would have shipped the tagger BACKWARDS
    # -- the same trap ar fell into. Rebuilt (2026-08-15) with, per language,
    #   python scripts/graft_xpos_tagger.py training_<l>_sud/model-best \
    #          training_<l>_xposwarm/model-best training_<l>_sud_xpos/model-best \
    #          --corpus corpus_<l>_sud/test.spacy
    # parse unchanged en 8585/8585, yue 1261/1261, id 11756/11756; tag_acc 0.9287 -> 0.9325,
    # 0.9286 -> 0.9313, 0.9264 -> 0.9288. These reproduce what SHIPPED rather than making a new
    # generation, and that is checked rather than argued: every weight file in the arm -- the five
    # base components AND the sud_* pipes -- is byte-identical to the one hashed out of the
    # DOWNLOADED v0.2.0 wheel. EN_BASE/YUE_BASE/ID_BASE get back the pre-graft arm.
    en)  base="${EN_BASE:-training_en_sud_xpos/model-best}" ;;
    # ⚠ NO `/model-best`: graft_xpos_tagger.py writes a MODEL directory, not a training directory,
    # so the old default pointed at a path that does not exist — and because the yue branch mutes
    # its steps, the only symptom was "SRC build_sud/work_yue.pkuseg missing — skip". lzh's grafted
    # default (training_lzh_seg_sud_xw) has always been correct on this; yue's never was.
    yue) base="${YUE_BASE:-training_yue_sud_xpos}" ;;
    id)  base="${ID_BASE:-training_id_sud_xpos/model-best}" ;;
    # fa names the GRAFTED arm, for the same reason ar does -- see the ar note below. Rebuilt with
    #   python scripts/graft_xpos_tagger.py training_fa_sud/model-best \
    #          training_fa_xposwarm/model-best <out> --corpus corpus_fa/fa_perdt-sud-dev.spacy
    # (parse unchanged 10427/10427, tag_acc 0.9599 -> 0.9613). FA_BASE gets back the pre-graft arm.
    # ADOPTED 2026-08-14 (user decision): the vocalisation-augmented chain, as ar. LAS spread
    # 64.40 -> 2.04, bare LAS +0.10. Its worst released row was not vocalisation at all -- Arabic
    # keyboard letterforms (ی/ي, ک/ك) cost 29.6 LAS and now cost 1.2. Ship decisions re-measured:
    # Shared trained 68.59 v rule 58.51, Reported trained 58.33 v rule 23.53 (both unchanged, and
    # Reported IMPROVED from 46.15), Idiom 72.73 unchanged. FA_BASE gets back the old arm.
    fa)           base="${FA_BASE:-training_fa_vocal_sud_xpos/model-best}" ;;
    # en_gum takes the XPOS-NORMALISED arm: EWT and GUM disagreed on punctuation XPOS (EWT tags
    # `;` `,` 101 times of 101, GUM the PTB-standard `:`), so the same token in the same context
    # carried different gold depending on which treebank the sentence came from. EWT's half was
    # converted to GUM's convention and the tagger retrained on the frozen arm -- every other
    # component byte-identical. Headline TAG is flat (0.3 % of the corpus) but accuracy on the
    # affected punctuation goes 72.47 -> 82.98. `en` (EWT-only) is deliberately NOT in this arm:
    # on its own, EWT's convention is internally consistent.
    #
    # ⚠ THE DEFAULT NAMED `training_en_gum_sud_xpos/model-best` UNTIL 2026-08-17, AND THAT ARM IS
    # PRE-GRAFT -- its tagger sits before the morphologiser, so the guard in pkg() refuses it and
    # the shipped wheel could not be rebuilt from this script at all. The comment in pkg() said
    # v0.2.0 "kept a `_sud_xpos` directory only for en_gum and la", implying those two defaults were
    # safe; they are not, and `training_la_aug_sud_xpos` is pre-graft in exactly the same way.
    # `training_en_gum_sud_xw` is the GRAFTED arm, rebuilt with (and reproducible by):
    #   graft_xpos_tagger.py training_en_gum_sud_xpos/model-best training_en_gum_xposwarm/model-best \
    #     training_en_gum_sud_xw --corpus corpus_en_gum_ext/en_ewtgum-sud-test.relabeled_ext.spacy
    # All three of that script's checks pass, and the wheel it packages is byte-identical to the
    # released v0.2.0 asset in every weight file -- only the licence metadata differs.
    en_gum)       base=training_en_gum_sud_xw ;;
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
    # ⚠ THE ARM CHANGED ON 2026-08-19 and the old default is a generation BEHIND in two ways at
    # once: pre-graft tagger AND pre-lemma-vector parser. `training_la_lemvec_misc` is the whole
    # chain -- morphologiser and lemmatiser in FRONT of the parser and annotating, so the parser
    # reads their predictions through `sud.LemmaVecFeatsEmbed.v1` (one hash table per morphological
    # category, plus PPMI+SVD lemma vectors). +1.51 LAS / +1.28 UAS on the combined test against a
    # CAPACITY CONTROL that is -2.56 below it, so the gain is the information and not the rows.
    # `sud_shared` is RETRAINED on this base rather than inherited (standing hazard 5): it pools
    # over the head and the head's other dependents, so a changed parse is a changed input, and the
    # transplanted pipe lost 0.44 F. Retrained it reaches 38.88 against the released 38.11.
    # The table is SEALED into the model bytes (seal_la_lemvec_model.py); an unsealed arm would
    # load on this machine and nowhere else. The wheel must therefore also ship
    # scripts/sud_lemmavec_embed.py -- it is in all three `la` --code lists below, and without it
    # the installed model raises E893 at load.
    la)           base="${LA_BASE:-training_la_lemvec_sud/model-best}" ;;
    # sa ships the JOINT MULTI-TASK arm: ONE shared encoder for tagger + parser + morphologizer +
    # lemmatizer, instead of the three-encoder freeze recipe every other arm uses. 25.85 -> 19.16 MB
    # (-25.9 %), tag/pos/morph/lemma each +0.3 to +0.7, and on HELD-OUT UFAL (classical prose, the
    # actual use case) LAS 0.3873 -> 0.4163 / UAS 0.5685 -> 0.6199. It costs Vedic LAS 0.5470 ->
    # 0.5140, accepted by user decision because the target is classical, not Vedic. NB the UFAL
    # figure rests on 416 tokens; the Vedic one on 18 k, so the cost is better measured than the gain.
    # ⚠ THIS DEFAULT NAMED THE JOINT MULTI-TASK ARM, which is NOT what ships. The v0.2.0 wheel was
    # built with SA_BASE overridden to the morph-first arm (parser/model 7de6d8d667d0fd3d ==
    # training_sa_mp2_s1, verified out of the DOWNLOADED wheel), so a bare `package_sud.sh sa` built
    # a different model from the released one — the same class of defect as lzh's and yue's defaults.
    # SA_MULTITASK is the escape hatch for the superseded joint arm.
    sa)           base="${SA_BASE:-training_sa_mp2_sub_s1/model-best}" ;;
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
    # KO_BASE is the override for the analyser-channel arm (docs/korean.md): that arm's config names
    # `sud.KoAnalyserEmbed.v1`, so its wheel MUST carry scripts/sud_ko_embed.py and
    # scripts/ko_analyser.py — which the ko --code list below now passes unconditionally, and the
    # missing---code guard in pkg() refuses without.
    ko)           base=${KO_BASE:-training_ko_eojeol_lemma/model-best} ;;
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
      # ⚠ AND THIS DEFAULT WAS PRE-GRAFT TOO. `training_lzh_trad_sud` has the tagger BEFORE the
    # morphologiser, so pkg()'s guard refuses it and the shipped wheel could not be rebuilt from
    # this script -- the same defect recorded for en_gum and la above, in a third place.
    # `training_lzh_trad_sud_xw` is the grafted arm, rebuilt with (and reproducible by):
    #   graft_xpos_tagger.py training_lzh_trad_sud/model-best training_lzh_xposwarm/model-best \
    #     training_lzh_trad_sud_xw --corpus corpus_lzh_trad/lzh_kyoto-sud-test.<suffix>.spacy
    # All three of that script's checks pass (parse unchanged 5628/5628, tags match donor
    # 5628/5628) and TAG goes 0.8469 -> 0.9254, which is the released wheel's generation.
  lzh)          base="${LZH_BASE:-training_lzh_seg_sud_xw}" ;;   # the SENTENCE-SEGMENTING arm
    # zh is TRADITIONAL-ONLY end to end, like lzh, and for the same reason -- a both-scripts
    # inventory never pools 個 with 个. Naming the arm here rather than falling through to
    # `training_zh_lemma` is not tidiness: the fall-through is the both-scripts generation, and it
    # is how the id wheel once shipped a generation stale. zh carries no SUD MISC layer, so the
    # lemma arm is the top of its chain.
    zh)           base="${ZH_BASE:-training_zh_trad_lemma/model-best}" ;;
    # ar names the GRAFTED arm, for the fourth iteration of the same lesson. The v0.2.0 wheel ships
    # the XPOS-downstream tagger -- warm-started, conditioned on UPOS+FEATS, and MOVED behind the
    # morphologiser so it can read them (TAG 89.44 -> 89.71) -- but that release grafted per arm
    # through SUD_BASE and kept no `_sud_xpos` directory for ar, so the default still pointed at
    # `training_ar_sud`, whose tagger predates the graft and whose pipeline has `tagger` second.
    # Packaging ar the routine way therefore rebuilt the PREVIOUS generation: it built, loaded and
    # parsed correctly, and only a file-by-file diff against the downloaded asset said otherwise
    # (tagger/model, tagger/cfg and vocab/strings.json moved). Rebuild the arm with
    #   python scripts/graft_xpos_tagger.py training_ar_sud/model-best \
    #          training_ar_xposwarm/model-best training_ar_sud_xpos --corpus corpus_ar/*-dev.spacy
    # AR_BASE gets back the pre-graft arm. A default that names the right arm is the fix.
    # ADOPTED 2026-08-14 (user decision): the VOCALISATION-AUGMENTED chain. Trained on the fully
    # pointed corpus with marks removed per document per epoch, so the arm reads any level of
    # pointing. LAS spread across orthographies 54.42 -> 1.42 at NO cost on bare text (LAS
    # identical to the decimal, TAG +0.41, LEMMA +2.21). ⚠ It does cost the IDIOM layer: F 67.30
    # -> 61.89 end-to-end, precision up 78.0 -> 80.1 but recall down 59.2 -> 50.4 -- the standing
    # pattern for a rule that is a CONJUNCTION of two of the base's own predictions. Shared and
    # Reported ship decisions both re-measured and both unchanged (Shared trained 52.17 v rule
    # 51.70 -- narrowed from ~2.0, so re-check it after any further base change; Reported rule
    # 73.49 v trained 34.78). AR_BASE gets back the un-augmented arm.
    # ⚠ The base is now the arm carrying the TRAINED Idiom pipes, and `add_idiom` is NOT run for
    # ar below. On the augmented base the rule loses to a trained pipe: Idiom 61.89 -> **64.98**,
    # InIdiom 62.30 -> **66.11** end-to-end on test, measured on this exact arm. The gain is recall
    # (58.75 v 50.42) against ~7 points of precision, and it comes from a signal the rule reads
    # NONE of -- 97.5 % of gold idiom heads have a lemma seen as an idiom head in train, while the
    # rule sees only ExtPos and `unk`, whose conjunction the augmented base supplies on just 50.4 %
    # of them. ar is the ONLY language where this wins: ja has no headroom (95.7 % availability,
    # rule already 96.88) and lzh/sa have half ar's data. See NEGATIVE-RESULTS.md.
    # Rebuild with: train the pipes against the SHIPPING arm (SUD_SRC_MODEL=..._sud_xpos), then
    # graft_pipe.py them in -- graft_pipe REFUSES if the two arms' frozen components differ, which
    # is what caught an earlier attempt to graft pipes trained against the pre-graft tagger.
    ar)           base="${AR_BASE:-training_ar_vocal_sud_idiom/model-best}" ;;
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
       # surgery as en, different corpus. It is CC BY-SA 4.0 like plain `en`, since 2026-08-17 --
       # GUM's maintainer confirmed the annotations are CC BY and the NC belongs to the individual
       # documents, which the genre filter already drops. The ARM key still matters: en_gum owes
       # GUM's CC BY ATTRIBUTION (SOURCES in stamp_model_meta.py) and plain en does not.
       # See scripts/build_en_ewt_gum.sh for the data build.
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
       # fa additionally ships `fa_vocalise`, bare (--no-lut), on the la/ar pattern. ⚠ It is NOT
       # the same proposition as ar's: Persian has no vocalised gold anywhere in this project, so
       # nothing about it is scored -- the lexicon is reconstructed from Tihu by aligning
       # pronunciations onto spellings, and Tihu holds one reading per word, so the morphological
       # disambiguation that makes ar_vocalise worth pipelining has no alternatives to choose
       # between. It answers 72.80 % of vocalisable PerDT test tokens; that is COVERAGE, not
       # accuracy. See fa_vocalise.py before quoting anything about it.
  fa)  add_idiom "$base" "$work.idiom"
       $PY scripts/add_vocalise.py --lang fa "$work.idiom" "$work" --no-lut --verify \
            --code sud_tagger.py,sud_misc.py,sud_idiom.py,sud_shared_data.py,sud_shared_rule.py,sud_reported_data.py,sud_reported_rule.py,sud_feats_embed.py \
            || { echo "  fa: fa_vocalise surgery FAILED — skip"; continue; }
       # fa also swaps in the NORMALISING tokeniser (sud.FaNormTokenizer.v1), which maps the
       # Arabic codepoints an Arabic keyboard produces (ي/ك -> ی/ک) and strips diacritics before
       # tokenising. This is the zh move -- normalise at the boundary rather than train for every
       # spelling -- and it is worth having ON TOP of the augmentation: 86.09 -> 87.02 LAS on
       # Arabic-letterform input, 85.64 -> 86.75 on all axes at once.
       # ⚠ Do NOT copy this to ar. Stripping diacritics costs the augmented ARABIC arm 0.77 LAS
       # and 4.42 TAG, because there the marks are the case endings. Persian has no case system,
       # so its short vowels carry no syntax and removing them is free. Normalise what is
       # information-free; augment for what is not, and for the ZWNJ, which cannot be reversed.
       # It goes LAST because it rewrites [nlp.tokenizer] in the config, and it re-verifies the
       # RELOADED model -- assigning nlp.tokenizer alone does not update the config.
       $PY scripts/add_fa_normaliser.py "$work" "$work.norm" --verify \
            --code sud_tagger.py,sud_misc.py,sud_idiom.py,sud_shared_data.py,sud_shared_rule.py,sud_reported_data.py,sud_reported_rule.py,sud_feats_embed.py,fa_vocalise.py,fa_align.py \
            || { echo "  fa: normalising tokeniser swap FAILED — skip"; continue; }
       rm -rf "$work" && mv "$work.norm" "$work"
       pkg fa  "$work" sud_perdt "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py,scripts/fa_vocalise.py,scripts/fa_align.py,scripts/fa_normalise.py" ;;
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
            --code sud_tagger.py,sud_misc.py,sud_shared_data.py,sud_shared_frames.py,sud_shared_rule.py,sud_idiom.py,sud_subject_frames.py,sud_subject_rule.py,sud_feats_embed.py,sud_lemmavec_embed.py \
            >/dev/null 2>&1
       $PY scripts/add_la_enclitic_tokenizer.py "$work.mac" "$work" --verify \
            --code sud_tagger.py,sud_misc.py,sud_shared_data.py,sud_shared_frames.py,sud_shared_rule.py,sud_idiom.py,sud_subject_frames.py,sud_subject_rule.py,la_macronise.py,sud_feats_embed.py,sud_lemmavec_embed.py \
            || { echo "  la: enclitic tokeniser swap FAILED — skip"; continue; }
       pkg la  "$work" sud_ittb_proiel_perseus \
            "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py,scripts/la_macronise.py,scripts/la_tokenizer.py,scripts/la_enclitics.py,scripts/sud_lemmavec_embed.py" ;;
       # ar now takes the TRAINED arm as its base (for sud_shared); add_sud_reported_rule drops
       # the trained sud_reported it also carries, since ar ships the Reported RULE (73.5 v 46.0).
       # ar ships `ar_vocalise` WITH its table, which is what separates it from la_macronise.
       # The table is harvested from SUD_Arabic-PADT's own Vform column and is therefore
       # CC BY-NC-SA, exactly like the parser trained on the same annotation -- so once the wheel
       # declares NC (stamp_model_meta.LICENSE["ar"], corrected 2026-08-14) there is nothing to
       # keep the two apart. la's case is different and stays --no-lut: its vowel lengths come from
       # Morpheus, a CC BY-SA source, and bundling THAT into an NC wheel would impose exactly the
       # restriction ShareAlike forbids. Same shape of component, opposite conclusion, because the
       # data has a different provenance from the model.
       # The calima-msa analyser fall-through is still never bundled (GPL v2): it activates when
       # the user runs `camel_data -i morphology-db-msa-r13`, which they need for the tokeniser
       # anyway. Added AFTER add_idiom so it lands last: it only writes `token._.vocalised`, reads
       # nothing sud_idiom writes, and nothing downstream reads it.
       # ⚠ NOT `>/dev/null 2>&1`. Three surgery scripts once failed silently in this driver and
       # surfaced only as a stale wheel; this one is allowed to speak, and to stop the arm.
  ar)  # The table is DERIVED data: gitignored like scripts/la_*_lut.json.gz and rebuilt on demand,
       # so a fresh clone packages correctly instead of failing on a missing file.
       [ -f scripts/ar_vocalise_lut.json.gz ] || $PY scripts/build_ar_vocalise_lut.py
       $PY scripts/add_sud_reported_rule.py "$base" "$work.rep" --lang ar >/dev/null 2>&1
       # NO add_idiom for ar: the base already carries the TRAINED sud_idiom/sud_inidiom pipes,
       # and add_sud_idiom.py would append the RULE on top of them, so the wheel would ship two
       # answers for one key with the rule silently winning by running last.
       $PY scripts/add_vocalise.py --lang ar "$work.rep" "$work" --verify \
            --code sud_tagger.py,sud_misc.py,sud_idiom.py,sud_shared_data.py,sud_shared_rule.py,sud_reported_data.py,sud_reported_rule.py,sud_feats_embed.py \
            || { echo "  ar: ar_vocalise surgery FAILED — skip"; continue; }
       pkg ar  "$work" sud_padt  "$CODE_REP,$CODE_SHARED,scripts/sud_tagger.py,scripts/ar_tokenizer.py,scripts/ar_vocalise.py" ;;
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
       # The segmenter is the TRADITIONAL one (models/zh_seg_jbdict_trad), and so is the jieba
       # dictionary its second channel reads: jieba's own is simplified, which costs that channel
       # boundary F 0.9237 -> 0.8931 if it is asked about traditional text directly.
       # ⚠ THIS DEFAULT NAMED models/zh_seg_jbdec_trad, the arm that instead converted the TEXT
       # (--jieba-t2s) -- equivalent in score (ten runs each, mean token F 0.9203 vs 0.9209) but
       # asking jieba about a string t2s had already collapsed (乾/幹/干 -> 干). The dictionary
       # travels inside the segmenter directory and add_zh_script refuses a model without it.
       # `bundle_zh_charseg.py` builds the superseded both-scripts CharSegTokenizer wheel and is
       # kept for that.
       # ⚠ ZH_BASE IS PRE-GRAFT AND THIS BLOCK CANNOT REBUILD THE RELEASED WHEEL FROM IT.
       # `training_zh_trad_lemma/model-best` runs [tok2vec, tagger, parser, morphologizer,
       # lemmatizer] while zh_sud_gsd-0.2.0 ships [tok2vec, parser, morphologizer, lemmatizer,
       # tagger] -- the v0.2.0 graft again, kept in no directory (same shape as ar, en_gum, la and
       # ko). The XPOS guard above refuses it, which is correct and is how this was found.
       # Until a post-graft zh arm exists, swap the tokeniser into the RELEASED model instead --
       # sound because training reads through sud.GoldTokCorpus.v1 and the parser is therefore
       # segmenter-agnostic, and verifiable because every other weight must come back byte-
       # identical (cmp on tok2vec/parser/morphologizer/lemmatizer/tagger `model`):
       #   gh release download v0.2.0 -R SunflowerAI/sud-spacy-parsers -p 'zh_sud_gsd-*.whl'
       #   unzip -q zh_sud_gsd-0.2.0-py3-none-any.whl -d rel
       #   ZH_BASE=rel/zh_sud_gsd/zh_sud_gsd-0.2.0 bash scripts/package_sud.sh zh
  zh)  $PY scripts/add_zh_script.py "$base" "$work" \
            --seg models/zh_seg_jbdict_trad --lexicon models/zh_lex_corpus_trad.txt \
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
       # THE CHARACTER SEGMENTER. lzh shipped "one Han character = one token" for its whole life,
       # but the Kyoto treebank is not: 852 of the traditional test set's 34,233 tokens are
       # multi-character (孔子 君子 匈奴 五十), so the rule tokeniser scored token F 0.9624 with
       # multi-char recall of exactly ZERO. The trained segmenter scores 0.9825 / 0.7124.
       # NO RETRAIN IS INVOLVED: gold_preproc + sud.GoldTokCorpus.v1 make the parser
       # segmenter-agnostic, and bundle_lzh_charseg.py asserts every component comes out
       # BYTE-IDENTICAL. It also refuses a segmenter whose stamped training corpus is not the
       # traditional one -- the both-scripts model differs by nothing a weight check can see.
  # THE COMBINED MULTI-FIELD TAGGER. Four per-field softmaxes over the comma-separated XPOS code
  # PLUS the joint 121-way head, on listener + UPOS/FEATS + PCA'd SikuBERT, all above the encoder.
  # Costs 0.15 TAG against the shipped tagger (92.72 -> 92.57) and buys per-field confidences and
  # an editing mode: with `upos_mask` on, a hand-corrected UPOS CONSTRAINS the XPOS, worth +3.11
  # TAG on gold UPOS (92.72 -> 95.83) and -0.33 on predicted — so it ships OFF and is a runtime
  # toggle, `nlp.get_pipe("tagger").cfg["upos_mask"] = True`.
  # ⚠ INSTALLED UNDER THE NAME `tagger` so pkg()'s XPOS-order and silenced-tagger guards still fire.
  # ⚠ It carries the SikuBERT vector table into the wheel; without it the channel reads zeros.
  # LZH_MFTAGGER=0 keeps the released single-softmax tagger.
  lzh) if [ "${LZH_MFTAGGER:-1}" != "0" ] && [ -d "${LZH_MFTAGGER_DONOR:-training_lzh_mftagger/model-best}" ]; then
         $PY scripts/swap_lzh_mftagger.py "$base" "$work.mft" \
              --donor "${LZH_MFTAGGER_DONOR:-training_lzh_mftagger/model-best}" >/dev/null 2>&1 \
              && base="$work.mft" || echo "  lzh: multi-field tagger swap FAILED — keeping the released tagger"
       fi
       $PY scripts/bundle_lzh_charseg.py --src "$base" --seg "${LZH_SEG:-models/lzh_seg_char_trad}" \
            --out "$work.tok" --verify >/dev/null 2>&1 \
            || { echo "  lzh: charseg bundling FAILED — skip"; continue; }
       $PY scripts/han_lemma_lut.py --build "$work.tok" "$work.lut" \
            --conllu "$LZH_TRAIN_CONLLU" >/dev/null 2>&1
       # ⚠ NO clause_parser ON lzh ANY MORE. It had three jobs; the punctuation-restored arm took
       # over marks and their morphology, and the SENTENCE-SEGMENTING base (config_lzh_seg.cfg) took
       # over the last one. Measured on the shipped wheel over 50 raw documents, dropping it is
       # 76.11 -> 76.23 LAS, 81.42 -> 81.51 UAS, 82.48 -> 83.24 SENTS_F — small, but consistent in
       # all three, and it removes a pipe that RE-PARSES every sentence.
       # ⚠ IT IS COUPLED TO THE BASE, so restoring it is one variable: set LZH_CLAUSE_PARSER=1, and
       # only for a base that does not segment. `--keep-marks` (worth +2.34 LAS on a
       # punctuation-trained arm, -3.80 on one that has never seen a mark) rides with it.
       if [ -n "${LZH_CLAUSE_PARSER:-}" ]; then
         $PY scripts/add_clause_parser.py "$work.lut" "$work.seg" \
              ${LZH_KEEP_MARKS:---keep-marks} >/dev/null 2>&1
       else
         cp -R "$work.lut" "$work.seg"
       fi
       $PY scripts/add_sud_subject_rule.py "$work.seg" "$work.rule" --lang lzh >/dev/null 2>&1
       $PY scripts/add_sud_idiom.py "$work.rule" "$work.idiom" --drop sud_subject >/dev/null 2>&1
       # THE 異體字 MAP, AT THE TOKENISER. A character the treebank never showed is tagged at
       # 51.79 % UPOS accuracy with 39.13 % PROPN precision (93.13 / 93.79 overall), and most such
       # characters are ordinary words in another glyph — 无=無 occurs 146 685 times in kanripo and
       # 0 times in Kyoto. The map is applied at ORTH, before segmentation, because NORM is only
       # one of the encoder's four channels: NORM-only moves 无's PROPN count 98 -> 69, rewriting
       # the glyph takes it to ZERO. NO RETRAIN — the parser is segmenter-agnostic — and
       # `--verify` asserts every file outside tokenizer/ is byte-identical AND that treebank
       # orthography passes through untouched. `token._.lzh_src` gives the caller its own spelling
       # back. Rebuild the table with scripts/build_lzh_variant_norm.py — it needs assets_unihan/
       # and a SikuBERT download, so it is NOT rebuilt here: an absent table skips normalisation
       # rather than failing the build, which is the right asymmetry for an artefact a build
       # machine may not be able to produce.
       if [ "${LZH_VARIANT_NORM:-1}" != "0" ] && [ -f "${LZH_VARIANTS:-models/lzh_variant_norm.json}" ]; then
         $PY scripts/bundle_lzh_variants.py --src "$work.idiom" \
              --variants "${LZH_VARIANTS:-models/lzh_variant_norm.json}" \
              --out "$work.var" --verify >/dev/null 2>&1 \
              || { echo "  lzh: variant bundling FAILED — skip"; continue; }
       else
         cp -R "$work.idiom" "$work.var"
       fi
       # SENTENCE JOINING. The base arm IS the senter (no `senter` pipe, no clause_parser — see
       # above), so it inherits Kyoto's 句讀 segmentation: reported speech comes apart (5 944 of
       # 59 215 training blocks have UNBALANCED 「」) and clauses break at commas. `sent_join`
       # imposes the reading convention instead, re-heading the roots the parser opened inside a
       # balanced quoted span or after a pause mark. The relation is HARVESTED, not chosen: inside
       # a quote a further clause takes the SAME governor and relation as the span's first clause
       # (Kyoto gives 1 743 of 1 755 spans exactly one external attachment, `comp:obj` of 曰);
       # elsewhere it comes from build_lzh_sent_joins.py's (mark kind, previous-head UPOS) table.
       # ⚠ IT GOES LAST, AFTER sud_idiom: it rewrites arcs, and every sud_* pipe reads the tree, so
       # an earlier position would change their input and couple this to that layer.
       # ⚠ AND ITS SIGN FLIPS BETWEEN HARNESSES. Over 10-sentence documents the gold uses the
       # other convention on both counts (31.2 % of its blocks END at a pause mark) and it costs
       # LAS 74.93 -> 71.67 / SENTS_F 80.41 -> 55.08. Under `--gold-preproc` every document IS one
       # gold sentence, so the only available error is OVER-splitting and it BUYS SENTS_F
       # 90.79 -> 95.18 for LAS 76.46 -> 76.30. docs/chinese-family.md has the full table.
       # LZH_SENT_JOIN=0 leaves it out; LZH_SENT_JOIN_ARGS passes --no-pause-join etc.
       if [ "${LZH_SENT_JOIN:-1}" != "0" ]; then
         # The join table is derived from the treebank in seconds and lives under the gitignored
         # models/, so build it rather than fail on a fresh checkout — unlike the variant map, this
         # one needs nothing a build machine lacks.
         :   # nothing to build: the clause rule is in the code, not in a harvested table
         $PY scripts/add_sent_join.py "$work.var" "$work.sj" \
              ${LZH_SENT_JOIN_ARGS:-} >/dev/null 2>&1 \
              || { echo "  lzh: add_sent_join FAILED — skip"; continue; }
       else
         cp -R "$work.var" "$work.sj"
       fi
       # `lzh_upos_rules`: post-morphologiser UPOS repair from the parser's view of a token's
       # CHILDREN and HEAD -- the family the morphologiser's own DEP channel cannot see. On the arm
       # that ships: UPOS 91.66 -> 92.69, 之 58.42 -> 89.36, NOUN/VERB unchanged (86.78 -> 86.77).
       # It carries NO weights, so every model file stays byte-identical; only config/meta change.
       # ⚠ IT EMITS A UPOS THE 0.3.0 WHEEL NEVER DID -- genitive 之 becomes PART, not SCONJ. That is
       # the traditional analysis and the point of the layer, but it is a BEHAVIOURAL change for
       # anything keyed on the old output, hence a version bump rather than a clobber.
       # ⚠ NOT MUTED, for the reason recorded just below: a swallowed registration failure once
       # shipped a broken wheel.
       if [ "${LZH_UPOS_RULES:-1}" != "0" ]; then
         $PY scripts/add_lzh_upos_rules.py "$work.sj" "$work" \
              || { echo "  lzh: add_lzh_upos_rules FAILED — skip"; continue; }
       else
         cp -R "$work.sj" "$work"
       fi
       pkg lzh "$work" sud_kyoto \
            "$CODE_BASE,$CODE_SHARED,scripts/sud_tagger.py,scripts/lzh_tokenizer.py,scripts/clause_parser.py,scripts/sent_join.py,scripts/lzh_upos_rules.py,scripts/han_lemma_lut.py,scripts/sud_subject_rule.py,scripts/sud_subject_frames.py,scripts/char_seg_tokenizer.py,scripts/sa_presegment.py,scripts/sa_presegment_lex.py,scripts/sud_multifield_tagger.py,scripts/sud_static_channel.py,scripts/sud_feats_embed.py" ;;
       # sa: Subject is too sparse to ship (142 train / 14 test); the idiom layer still applies.
       # sa_compound must stay FIRST (the encoder reads MORPH); clause_parser before sud_idiom.
       # sa: the whole front end (CSLiser + de-CSLizer + de-sandhifier + Devanagari rendering)
       # is assembled by add_sa_frontend.py, which also inserts sa_compound / clause_parser /
       # sa_deva in their required positions. CSL is an INTERNAL representation only — the wheel
       # takes raw IAST or Devanagari.
  # ⚠ NOT MUTED. `>/dev/null 2>&1` here swallowed an E893 registration failure and shipped the
  # PREVIOUS pipeline with a clean-looking log — the failure mode CLAUDE.md lists as standing
  # hazard 2. Both steps now fail loudly and stop the build.
  sa)  $PY scripts/add_sa_frontend.py "$base" "$work.front" \
            --csliser models/sa_presegment_ortho \
            --unsandhi training_sa_mwt_unsandhi/model-best \
         || { echo "  sa: add_sa_frontend FAILED"; return; }
       $PY scripts/add_sud_reported_rule.py "$work.front" "$work.rep" --lang sa \
         || { echo "  sa: add_sud_reported_rule FAILED"; return; }
       add_idiom "$work.rep" "$work"
       # sa's `parser` may be the arc-factored decoder (sud.ArcFactoredParser.v1) rather than the
       # standard TransitionBasedParser -- SA_BASE names which; these three travel unconditionally,
       # the same "inert if unused, missing if needed" call already made for ko's analyser channel
       # above, since the alternative is an E893 the missing---code guard exists specifically to
       # catch before it reaches a user.
       pkg sa  "$work" sud_vedic_ufal_dcs \
            "$CODE_REP,scripts/sa_tokenizer.py,scripts/clause_parser.py,scripts/sa_presegment.py,scripts/sud_unsandhi.py,scripts/sud_affix_embed.py,scripts/sa_devanagari.py,scripts/sud_analyser_embed.py,scripts/sud_arcfactored_parser.py,scripts/sud_joint_biaffine.py,scripts/train_arcfactored.py,scripts/sud_cle.py" ;;
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
       # ko's list is no longer empty: the analyser channel's two modules travel whether or not
       # KO_BASE names an arm that uses them. Inert in a wheel that does not (`ko_analyser` loads
       # its backend lazily, so importing it costs nothing and needs no mecab), and the difference
       # between a working wheel and an E893 on the user's machine in one that does.
  ko)  pkg ko  "$base" sud_gsd "scripts/sud_ko_embed.py,scripts/ko_analyser.py" ;;
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
