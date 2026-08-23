# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository. It is the **map**; two companions hold the territory.

- **`NEGATIVE-RESULTS.md` — check it before retrying anything that looks obviously right.** It
  records ~20 measured dead ends (affix widening, decode-time lexicons, LLM multi-way relabelling,
  data upsampling, tree-aware encoders, beam training, agreement as an explicit parser input) and
  the meta-lessons behind them.
- **`docs/` — read the relevant file BEFORE touching the area it covers** (indexed below). Every
  one of them records at least one defect that the ordinary metrics reported as healthy.

## What this project is

Two coupled pieces of work over **Surface-Syntactic Universal Dependencies (SUD)** treebanks, now
thirteen languages: en, zh, yue, lzh, ja, ko, id, fa, ar, la, sa, ta, te — in **fourteen** wheels, since
English ships twice (`en_sud_ewt` EWT-only, `en_sud_ewt_gum` + GUM; both CC BY-SA).

1. **Small CPU spaCy pipelines** trained from SUD CoNLL-U and released as wheels. Component order
   is `[tokenizer, tok2vec, parser, morphologizer, lemmatizer, tagger, …language extras…, sud_*]`
   — the tagger sits **behind** the morphologiser because it reads UPOS+FEATS (`docs/xpos.md`), and
   `package_sud.sh` refuses to package an arm where it does not.
2. A **`udep` disambiguation pipeline**: SUD labels adpositional/case-marked dependents of verbs
   with the noncommittal `udep`; we relabel each as `comp:obl` (complement) or `mod` (modifier)
   using a local LLM via Ollama (no thinking, temperature 0), then retrain and compare. This is the
   core research contribution — see `docs/udep-relabel.md`, `README.md` and `metrics/<lang>/*.json`.

There is no package/test suite; "running it" means executing the spaCy CLI and the `scripts/*.py`
pipeline. Always use the project venv: `.venv/bin/python`.

## Environment (critical, non-obvious)

- **Python 3.12 only.** The machine default `python3` is 3.14, which has no spaCy wheels.
  `pip install spacy` does **not** pull in `click` (spaCy imports it directly) — pinned in
  `requirements.txt`.
- **Korean** needs mecab-ko for anything touching `config_ko.cfg` or the superseded morpheme arm:
  `export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib`. mecab-ko was installed via Homebrew
  (conflicts with and unlinked the Japanese `mecab`); `mecabrc` dicdir points at `mecab-ko-dic`.
- **Chinese** needs `jieba>=0.42.1` (a feature channel for the char segmenter; the wheel vendors a
  pruned copy). **Cantonese** needs `spacy-pkuseg`. **Japanese** needs `sudachipy`. **Arabic** needs
  `camel-tools` plus a hand-run `camel_data -i …`.
- **Ollama** must be running with the per-language model pulled (`qwen3:8b`, or `gemma4:latest` for
  ar/la — `OLLAMA_MODEL` selects it; `disambiguate_pp.MODEL` reads it). A single request already
  saturates the Metal GPU — parallel requests / `OLLAMA_NUM_PARALLEL>1` give **no** speedup (~3
  calls/s ceiling). Don't parallelise.
- **No GPU path for spaCy.** thinc's GPU backend is CuPy/CUDA only; `thinc-apple-ops` is installed
  and thinc uses `AppleOps` (Accelerate/AMX). Ollama does use Metal, so LLM passes and spaCy
  training contend only mildly (1.7 → 1.1 decisions/s). The BLAS backend is worth more than any
  architecture change measured here — `AppleOps` is **2.6× per request** over `NumpyOps` on
  identical weights.

## Common commands

```bash
PY=.venv/bin/python
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib   # only when Korean is involved

# Convert CoNLL-U -> .spacy
$PY -m spacy convert <file>.conllu <out_dir>/ --converter conllu -n 10

# Train (English uses configs/config.cfg; per-lang configs/config_<lang>.cfg)
$PY -m spacy train configs/config_zh.cfg --output training_zh/ \
  --paths.train corpus_zh/<prefix>-train.spacy --paths.dev corpus_zh/<prefix>-dev.spacy

# Evaluate — everything except en needs --gold-preproc (see below)
$PY -m spacy evaluate training_zh/model-best corpus_zh/<prefix>-test.spacy --gold-preproc
```

Layer drivers, run in this order per language: `retrain_seg.sh` → `train_morph.sh` →
`train_lemma.sh` → `train_sud.sh`, then `package_sud.sh` (which supersedes `package_seg.sh` /
`package_morph.sh` / `package_lemma.sh`). Relabel drivers: `relabel_retrain.sh` (verb scope),
`relabel_retrain_ext.sh` (extended scope), `retrain_udep_ruled.sh` (derived-rule residue commits).
Other drivers: `train_vocal.sh` (ar/fa vocalisation augmentation), `train_xpos.sh` / `eval_xpos.sh`
(the conditioned tagger), `train_la_ext_macron.sh` / `train_la_aug.sh` (Latin orthography),
`train_ko_analyser.sh` (the Korean analyser channel and its capacity control, three seeds each).
Superseded but kept: `train_baselines.sh`, `train_all_retok.sh` / `eval_retok.sh` /
`relabel_retrain_retok.sh` (the matched-tokenisation arms), the `*_new.sh` drivers that brought in
fa/ar/la/sa/lzh/ja, `both_scripts_release.sh`, and `rebuild_sa_csl_rev.sh` (+ `hyphen_to_pipe_sa.py`,
`strip_pipe_sa.py`) for the pausa-normalised Sanskrit representation.
`spacy train` writes scores to `train_*.log`; `spacy evaluate --output` writes
`metrics/<lang>/metrics_<lang>_*.json`. **Metrics live under `metrics/`, not the repo root** (moved
2026-08-23; filenames unchanged, so the language is still readable off any single copied name). The
drivers that write them `mkdir -p` the tree first, because several send stderr to `/dev/null` and a
missing directory would fail silently. LLM decision caches are in `caches/`, gold benchmark sets in
`gold/`.

**`metrics/release/*.json` is the RELEASED arm; every other `metrics/<lang>/*.json` is a development one.**
The distinction earns its keep: several development files outlived the generation they describe and
the README was quoting them. The release set was measured on the arm each wheel ships, identified by
hashing `parser/model` out of the DOWNLOADED wheel — **a training directory of the right name is not
evidence**. Re-measured against the downloaded v0.3.0 assets on 2026-08-23: **la was stale by a
whole arm** (the wheel ships the lemma-vector parser, LAS 71.72 → **73.23**), **sa by two
generations** (37.35 → **48.54**), and **ta and te had no release file at all**. en/en_gum/id/zh/ja/ko
reproduce exactly. **ar, fa, lzh and yue were stale too**, and for the standing reason: each was
re-released onto a different arm (ar/fa the vocalisation-augmented ones, `ea1886f`; lzh/yue after the
subtype restorations, `0d49e18`/`05539b9`) and nobody re-measured. All four wheels hash byte-identical
to the arm `package_sud.sh` names, so **nothing needed retraining** — and the corpus to score on is
the one the arm's OWN `config.cfg` names, not the one whose name looks right (`fa` on `corpus_fa_ext`
rather than `corpus_fa_sud` is a full LAS point). `docs/release-notes.md` has the table. ja cannot be scored with the stock reader at
all: use `scripts/eval_ja_infl.py --reader infltag`, or read LAS 72.06 for a model that does 90.04.

## Where the details live

| doc | covers | the trap it records |
|---|---|---|
| `docs/results-notes.md` | what is behind each number in the README's results table, per language | the release set is **not cross-language comparable** — sa is measured on a different test set |
| `docs/training-data.md` | corpus sizes, and what each treebank does and does not annotate | te trains on 5 097 tokens and la on 586 604; DCS's 1.7 M tokens carry **no dependency annotation** |
| `docs/release-notes.md` | what changed in each wheel and when; which release figures are stale | `ar`, `fa`, `lzh` and `yue` do **not** re-derive from any local corpus, and the gap is unexplained |
| `docs/layers-and-tokenisers.md` | the seg / morph / lemma layers; the per-language tokenisers; zh's jieba-decision channel | `seg` is a BASE recipe, not a stackable layer; a corpus lexicon only works **jackknifed** |
| `docs/udep-relabel.md` | the research contribution — comp/mod prompting, extended scope, derived rules, `reparandum` → `conj:dicto` | each relabel rewrites the **test** gold too, so `comp:obl` F has a moving denominator |
| `docs/xpos.md` | one tagset per arm (la's composite codes, en's `,`), the warm-started tagger conditioned on UPOS+FEATS | conditioning must enter **above** the encoder; a warm start must match label **order**, not just the set |
| `docs/vocalisation.md` | `ar_vocalise` / `fa_vocalise`, the augmentation recipe, normalise-vs-augment, ar's trained Idiom | augmentation costs are **not uniform across labels** — rare ones pay first |
| `docs/sud-misc-layer.md` | `Idiom` / `InIdiom` / `Subject` / `Reported` / `Shared`: which arm ships per language, and why | the layer is **coupled to the base underneath it**; `annotating_components` must list `tok2vec` |
| `docs/packaging-and-release.md` | `package_sud.sh`, wheel contents, the release audits, serverless sizing | **a directory is not a release** |
| `docs/latin.md` | three treebanks, macrons in and out, orthographic augmentation, `la_macronise`, where the parser's errors actually are | the macron table's lookup key must be orthography-tolerant, least-normalised first; 63 % of attachment errors sit in a non-projective sentence, and no headline metric says so |
| `docs/sanskrit.md` | the raw-sandhied-text front end, the CSLiser, the sandhi machinery, the joint multi-task arm | an **unset** MORPH and an **empty** one are different inputs (cost 6.8 LAS) |
| `docs/chinese-family.md` | zh traditional-only + `zh_script`, lzh's restored punctuation and `clause_parser`, yue | `_looks_simplified` cannot be "would `s2t` change it?"; `keep_marks` is coupled to the arm underneath |
| `docs/lzh-tokenisation.md` | lzh's multi-character tokens, the trained char segmenter, the Heart Sūtra gold set, and every lexicon/gazetteer route measured | the released lzh tokeniser splits 孔子, and **no standard metric can see it** — `gold_preproc` bypasses the tokeniser |
| `docs/korean.md` | the eojeol tokenisation, the mecab-ko analyser channel, the sentenciser, the constrained scrambler | 34.5 % of test tokens are unseen STRINGS and parse 29.6 LAS below the rest; the headline never said so |
| `docs/languages.md` | en's two arms and two licences; id's FEATS and lemma-casing fixes | an arm name is not a language — the two places that confused them both failed silently |
| `docs/dravidian.md` | ta's two treebanks and its akṣara-decomposition tokeniser; te's missing multiword tokens and missing morphology; the head-final order augmenter | Telugu's lemma column is `_` on EVERY token and spaCy keeps that as a literal string; and MTG ships **no** multiword tokens, which is an annotation policy, not a fact about Telugu |

## The fourteen wheels

**ja, ko, la and sa are at v0.3.0** and **ta and te are new at 0.1.0**, all six on the `v0.3.0`
release; the other eight are at v0.2.0 on `v0.2.0`. Published on the GitHub Release, not in git.

The 0.2.0 set is re-clobbered in place as layers land, so `pip install -U` will NOT pull those —
which is why the four above took a version bump instead. Most recently clobbered: **sa at 0.3.0 on 2026-08-23** — the
re-tuned `Reported` rule (test F 48.76 → 53.05 as the wheel runs), a CODE-ONLY change in which
all six weight files are byte-identical to the previous asset, verified out of the DOWNLOADED
wheel (`docs/sud-misc-layer.md`). Before that, **zh, id and lzh
at 0.2.0 and ta at 0.1.0, all on 2026-08-22 at 19:47 UTC** — the `SEG_BATCH` memory cap in
`char_seg_tokenizer.py` (and `ta_tokenizer.py`), a SOURCE-ONLY change in which every model byte is
unchanged: each wheel was rebuilt by unpacking the released asset, editing the bundled file and
repacking, so RECORD and those two files are the ONLY entries that differ, and all four reproduce
their previous token stream and full-pipeline parse digest exactly. Earlier the same day, zh alone
took the traditional jieba dictionary (tokeniser only); before that lzh and yue on 2026-08-19. **This paragraph goes stale
faster than anything else in this file** (sa shipped 0.3.0 within a day of it last being written),
so re-derive it rather than trusting it — the asset list, with the upload times that reveal a
clobber, is one command:

```bash
gh release view v0.3.0 --json assets -q '.assets[] | "\(.name)  \(.updatedAt)"'
```

| lang | wheel | licence | tokeniser | note |
|---|---|---|---|---|
| en | `en_sud_ewt` | CC BY-SA 4.0 | rule | EWT only — the narrower attribution surface |
| en | `en_sud_ewt_gum` | CC BY-SA 4.0 | rule | + the ten non-NC GUM genres, +66 % train tokens; owes GUM's CC BY attribution |
| ar | `ar_sud_padt` | CC BY-NC-SA 4.0 | rule + `ar_tokenizer` | vocalisation-augmented; ships the `Vform` table and a trained Idiom pipe |
| fa | `fa_sud_perdt` | CC BY-SA 4.0 | rule + `sud.FaNormTokenizer.v1` | normalises Arabic letterforms **in**; ships ezāfe rules without the GPL lexicon |
| la | `la_sud_ittb_proiel_perseus` | CC BY-NC-SA 4.0 | rule + `-que` split | orthography-augmented; parser reads predicted lemma vectors + per-feature morphology (LAS 71.72 → 73.23), table **sealed** into the bytes; `la_macronise` ships without its data |
| sa | `sa_sud_vedic_ufal_dcs` | CC BY-SA 4.0 | `sa.SanskritInputTokenizer.v3` | accepts **raw sandhied** IAST or Devanagari; joint multi-task arm |
| zh | `zh_sud_gsd` | CC BY-SA 4.0 | char tagger + jackknifed lexicon + jieba BMES off a TRADITIONAL jieba dictionary | traditional-only; vendors a pruned jieba, now without its simplified `dict.txt` (`build_jieba_trad_dict.py` supersedes the `t2s`-the-text channel; **re-clobbered at 0.2.0 on 2026-08-22**, every non-tokeniser weight byte-identical) |
| yue | `yue_sud_hk` | CC BY-SA 4.0 | pkuseg trained on yue | test-only treebank → deterministic 80/10/10 split |
| lzh | `lzh_sud_kyoto` | CC BY-SA 4.0 | `sud.CharSegTokenizer.v1` (trained) | custom `lzh` language; `clause_parser`; punctuation restored from kanripo; segmenter recovers 孔子/匈奴 (token F 0.9624 → 0.9825) |
| ja | `ja_sud_gsd` | CC BY-SA 4.0 | SudachiPy | |
| ko | `ko_sud_gsd` | CC BY-SA 4.0 | eojeol, spaCy's rule tokeniser | requires `python-mecab-ko`: the parser reads the morphemes an eojeol hides (`docs/korean.md`, raw LAS 55.81 → 73.16). Ships a `senter`; no SUD MISC layer |
| id | `id_sud_gsd` | CC BY-SA 4.0 | char tagger, enclitics SPLIT | `id_lemma_case_fix` after the lemmatiser |
| ta | `ta_sud_ttb_mwtt` | CC BY-NC-SA 3.0 | `sud.TamilSandhiTokenizer.v1` (trained) | TTB + test-only MWTT split 80/10/10; parser reads LEMMA + per-feature morphology (+1.34 LAS over its capacity control); akṣara decomposition makes sandhi splitting ordinary segmentation, token F 0.8389 → 0.9420 |
| te | `te_sud_mtg` | CC BY-NC-SA 3.0 | `sud.TeluguSplitTokenizer.v1` (lookup) | no lemmas and no FEATS in the treebank, so no lemma/morphology channel; MTG ships NO multiword tokens and 20 were added from its own evidence |

Per-language relabel model: fa/sa/lzh/en/id/zh/ko/yue → `qwen3:8b`, ar/la → `gemma4`, **ja → qwen3
with a native-Japanese prompt**. zh and ko ship **no** SUD MISC layer at all.

## Tamil and Telugu (`docs/dravidian.md`)

**RELEASED 2026-08-19 on the `v0.3.0` release, both at 0.1.0**: `ta_sud_ttb_mwtt` (TTB + the
test-only MWTT, split 80/10/10) and `te_sud_mtg`. ta was **re-clobbered at 0.1.0 on 2026-08-22**
for the `SEG_BATCH` cap — source only, every model byte unchanged; te was not touched, since its
lookup splitter runs no segmenter. Both **CC BY-NC-SA 3.0**. Verified after upload by
hashing `parser/model` and `tok2vec/model` out of the DOWNLOADED assets, then installing from the
public release URL into a clean target with `scripts/` off `sys.path`.

Four things about them are load-bearing for anyone touching the area:

- **te MTG carries no lemmas and almost no FEATS** (115 values in 6 465 tokens); its XPOS is a
  verbatim copy of UPOS. So te has no lemma/morphology channel. Its empty lemma column is a TRAP
  rather than an absence: `spacy convert` keeps `_` as a literal string, and all 5 082 training
  tokens came out with `token.lemma_ == "_"`. `scripts/prep_te.py` falls back to IDENTITY.
- **te had no multiword tokens at all** ("Word count: 6465, Token count: 6465" in its own README),
  against Tamil's 9.67 % of orthographic words. `scripts/split_te_mwt.py` re-annotates 20 of them
  from the treebank's own evidence; `training_te_nomwt_*` keeps the unsplit control.
- **`min_action_freq = 30` deletes most of the label inventory at this corpus size** — 7 of ta
  TTB's 19 deprels, 19 of the combined arm's 33, 14 of te's 29 — silently, with their recall pinned
  to zero. `scripts/make_dravidian_config.py` sets it to 1.
- **Neither ships a SUD MISC layer**, and that is measured rather than skipped: ta `Subject`
  reaches P 75.0 % over EIGHT predictions, a 95 % interval of [41 %, 93 %] that spans the floor,
  and `Shared` is capped at zero by a candidate mask reaching no gold at all.

## Conventions and invariants

**Naming.** English artifacts are unsuffixed (`corpus/`, `training/`, `metrics/en/metrics.json`); other
languages take `_<lang>`. Relabel variants: `_rl` (verb scope), `_ext` (extended scope), `_rl2`
(contrastive-prompt rerun). Layers: `_seg`, `_morph`, `_lemma`, `_sud`. These compose
(`training_ko_retok_rl/`, `metrics/zh/metrics_zh_simp_rl_gp.json`); `metrics_*_{gp,raw}.json` are the
gold-preproc and raw end-to-end evaluations.

**gold_preproc (essential for every language but en).** `spacy evaluate` re-tokenises raw text with
the model's tokeniser; a mismatch with gold tokens collapses alignment (Korean LAS once dropped to
~30). Configs set `gold_preproc = true` and **evaluation must pass `--gold-preproc`**. en (spacing
matches) doesn't need it. All research metrics in `CLAUDE.md` and `docs/` are gold-preproc unless
marked raw.

**Editing configs programmatically:** load with `Config().from_disk(p, interpolate=False)` — the
default interpolation resolves `${paths.train}` to null and silently breaks CLI path overrides
(this caused `E913`).

**The freeze recipe** (how every layer above the base arm is added): source the arm's existing
components, **freeze** them, and train ONLY the new component, giving it its **OWN small
`HashEmbedCNN`** (width 64 / depth 3 / embed 2000) rather than a listener. A dedicated encoder is
immune to treebanks whose XPOS is orthogonal to UPOS (id: 33/46 XPOS values map to >1 UPOS), and
co-training is dominated (see NEGATIVE-RESULTS). Frozen components come out **byte-identical**
(verify per-arm with `cmp` on `*/model`), so lower-layer metrics need no re-verification. sa is the
one exception — it now ships a **joint multi-task** arm (`docs/sanskrit.md`).

**Custom readers (`scripts/gold_tok_corpus.py`, `sampling_corpus.py`).**
`sud.GoldTokCorpus.v1` yields whole multi-sentence docs (corpora are `convert -n 10`) with **gold
tokenisation** for the predicted doc — segmentation is learned with zero tokeniser skew, and the
parser is therefore **segmenter-agnostic**, which is why a tokeniser can be swapped into a released
arm with no retrain. `sud.CompoundCorpus.v1` additionally copies **only** the `Compound` feat from
the reference (sa's input feature). `sud.SamplingCorpus.v1` rebalances by sampling, not duplication.
Two traps for any custom reader: it MUST be finite per call (spaCy's `initialize` iterates the whole
training corpus to collect labels, so `while True` hangs at 100 % CPU with no output), and the
predicted doc must be built from GOLD WORDS, never `nlp.make_doc` (the sa tokeniser rewrites its
input, so re-tokenising raises E949).

**Anything that rebuilds a `Doc` owns carrying EVERY annotation.** `clause_parser` has been bitten
by this twice — first dropping lemma/morph, then dropping Token-level extensions
(`_.unsandhied` came out blank on every token of the assembled sa model). Doc-level extensions are
not enough.

**An UNSET MORPH and an EMPTY one are different inputs.** `set_morph("")` produces key 456, an unset
token key 0, and both render as `''` — so no string-level check catches the difference, and the
encoder silently sees a value it never met in training. This cost sa's raw path **6.8 LAS**. Any
component that stamps MORPH must set it only where it has a value.

**`spacy convert --converter conllu` reads MISC for only `SpaceAfter=No` and the NER pattern**
(`spacy/training/converters/conllu_to_docs.py`) and
discards the rest, so `Subject=`/`Idiom=`/`Unsandhied=` never reach `.spacy` corpora. Transport them
through a column that survives (`hoist_sud_gold.py` puts them in FEATS under a `Sud` prefix;
`make_unsandhi_corpus.py` puts `Unsandhied` in LEMMA). **spaCy keeps CoNLL-U `_` as a LITERAL
value**, not as missing — writing `_` for tokens with no gold taught the sandhi transducer
`FORM → "_"` on 5 043 tokens. Fall back to identity instead.

## Standing hazards — each has been paid for more than once

1. **A directory is not a release.** Neither `training_*/` nor `build_sud/` says anything about what
   users have; `build_sud/` has held two wheels of the same name at different generations. Verify
   against `gh release view`, the asset size, and weight hashes taken **out of the downloaded
   wheel**. Upload by NAME — `find build_sud -name '*.whl'` is still the wrong command.
2. **A default that names the right arm — or the right CORPUS — is the fix; a comment telling the
   next person is not.** Extended to corpora on 2026-08-23: `train_sud.sh`'s `src_conllu()` still
   named `corpus_sa_csl_rev`, the superseded pausa-normalised generation, long after
   `rebuild_sa_csl_rev.sh` was listed below as superseded. Anything copying that map got Sanskrit
   **unrelabelled** (`udep` on 7.89 % of tokens against 0.00 % in the current generation) and on a
   tokenisation the released arm does not share — and a superseded corpus loads, converts and
   trains exactly like a current one, so nothing raised. Fixed, and every other `csl_rev` reference
   in `scripts/` and `configs/` is now either repointed or carries a SUPERSEDED banner.
   This has been paid for four times over — lzh nearly shipped a generation backwards three times
   through `package_sud.sh` defaults, then ar did. The durable fix is a default plus a refusal:
   `pkg()` will not package an arm whose pipeline has `tagger` before `morphologizer`.
3. **Check the branch is not behind main before building anything** — `git log --oneline
   <branch>..main`. A six-commit-behind branch once rebuilt and uploaded all eleven wheels, shipping
   lzh a generation backwards and eleven empty `License:` fields.
4. **`gold_preproc` hides tokeniser and sentence-boundary defects.** Gold tokens bypass the
   tokeniser (so `TOK` goes unscored) and every dev example is already one sentence (so `SENTS_F`
   reads 100.00 for a parser that never learned to *start* one). zh shipped both defects. Confirm on
   a raw end-to-end eval, or by typing two sentences at the model.
5. **Re-measure the MISC layer after any base retrain.** Every pipe there reads the base's own
   predictions, and the Idiom rule is a CONJUNCTION of two of them, so upstream errors multiply
   rather than add. Ship decisions have reversed on a base change in both directions.
6. **Rare labels pay first.** Headline `morph_acc`/`TAG` are dominated by common labels, so a change
   can raise them while a rare label (`ExtPos`) falls 10 points and takes a downstream layer with
   it. Any decision resting on a rare label must be re-measured, never inferred from the headline.
7. **Copying weights needs the label ORDER, not just the label set.** spaCy orders parser actions by
   `(frequency, label_string)` descending and a tagger's `W` is indexed by label id, so a rename or
   a warm start can silently scramble every class. `rename_deprel_label.py` and `--warm-start` both
   refuse unless the sequence matches position for position.
8. **A component that silently loses an input must refuse to load.** Copying state by GUESSING
   attribute names, plus a `from_disk` that falls back when a directory is absent, together shipped
   a zh wheel that returned one token per input string. Verify the **reloaded** model, never the
   in-memory one — and note that assigning `nlp.tokenizer` does **not** update the config, so
   `nlp.config["nlp"]["tokenizer"]` must be set too.
9. **`max_epochs = 0` silently defeats a corpus-level augmenter**: spaCy lists the corpus ONCE, so
   one style is sampled per document for the whole run and the run looks normal. Use `-1`, add
   `shuffle = true` to the reader (the loop stops shuffling), and collect labels with
   `init_aug_labels.py` — a missing edit-tree label does not raise, it teaches label 0.
10. **A per-sentence metric cannot see a cost that is linear in the length of the CALL.** Every
    number in this repo is computed sentence by sentence, so a tokeniser that batched its whole
    input into one `predict` looked healthy for four languages while costing 10-14 kB per character
    of the calling string — invisible until someone handed it a book. Peak memory, like `TOK` under
    `gold_preproc`, is a dimension the scoreboard does not have. Ask what the largest single call a
    user can make looks like, not what the test set looks like.
11. **Ask the model rather than assuming its input regime.** A CSLiser trained on spaced text was
    fed space-split chunks for a whole generation (−4.83 F), and a jieba channel asked a different
    question at inference than at training would do the same. Record the regime in the artefact
    (`reads_spaces`, `jieba_t2s` / `jieba_dict` in `vocab.json`) and read it back — and where the
    regime needs a FILE, ship the file beside the weights and refuse to load without it, because
    zh's traditional jieba dictionary silently replaced by jieba's own simplified one is a model
    that loads, segments and is wrong only on the vocabulary the two disagree about.

## Operational notes

- **`spacy train … | tail -N` hides everything until the command exits.** Two runs looked stalled for
  hours and one genuinely was; `model-last`'s mtime is the reliable progress signal (rewritten at
  EVERY eval), and `python -u` is needed for live output when redirecting.
- `config_zh` init reporting a "missing pkuseg model" is really the gitignored `userdict.txt`
  artifact (legacy pkuseg path).
- Backups of superseded representations: `backup_sa_prestrip/` (pipe-join CoNLL-U),
  `backup_sa_prepipe/` (hyphen), `backup_la_preperseus/`, `backup_{en,la}_prexposnorm/`,
  `archive_residue_pass{1,2}/`.
