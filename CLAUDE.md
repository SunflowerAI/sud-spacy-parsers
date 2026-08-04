# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Failed experiments live in `NEGATIVE-RESULTS.md`.** Check it before retrying anything that looks
obviously right — it records ~20 measured dead ends (affix widening, decode-time lexicons, LLM
multi-way relabelling, data upsampling, tree-aware encoders) and the meta-lessons behind them.

## What this project is

Two coupled pieces of work over **Surface-Syntactic Universal Dependencies (SUD)** treebanks, now
eleven languages: en, zh, yue, lzh, ja, ko, id, fa, ar, la, sa.

1. **Small CPU spaCy pipelines** trained from SUD CoNLL-U and released as wheels
   (`[tokenizer, tok2vec, tagger, parser, morphologizer, lemmatizer, …language extras…, sud_*]`).
2. A **`udep` disambiguation pipeline**: SUD labels adpositional/case-marked dependents of verbs
   with the noncommittal `udep`; we relabel each as `comp:obl` (complement) or `mod` (modifier)
   using a local LLM via Ollama (no thinking, temperature 0), then retrain and compare. This is the
   core research contribution — see `README.md` and `metrics_*.json`.

There is no package/test suite; "running it" means executing the spaCy CLI and the `scripts/*.py`
pipeline. Always use the project venv: `.venv/bin/python`.

## Environment (critical, non-obvious)

- **Python 3.12 only.** The machine default `python3` is 3.14, which has no spaCy wheels.
  `pip install spacy` does **not** pull in `click` (spaCy imports it directly) — pinned in
  `requirements.txt`.
- **Korean** needs mecab-ko for anything touching `config_ko.cfg` or the superseded morpheme arm:
  `export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib`. mecab-ko was installed via Homebrew
  (conflicts with and unlinked the Japanese `mecab`); `mecabrc` dicdir points at `mecab-ko-dic`.
- **Chinese** needs `jieba>=0.42.1` (a feature channel for the char segmenter, declared in the
  wheel's `meta.json`). **Cantonese** needs `spacy-pkuseg`. **Japanese** needs `sudachipy`.
- **Ollama** must be running with the per-language model pulled (`qwen3:8b`, or `gemma4:latest` for
  ar/la — `OLLAMA_MODEL` selects it; `disambiguate_pp.MODEL` reads it). A single request already
  saturates the Metal GPU — parallel requests / `OLLAMA_NUM_PARALLEL>1` give **no** speedup (~3
  calls/s ceiling). Don't parallelise.
- **No GPU path for spaCy.** thinc's GPU backend is CuPy/CUDA only; `thinc-apple-ops` is installed
  and thinc uses `AppleOps` (Accelerate/AMX). Ollama does use Metal, so LLM passes and spaCy
  training contend only mildly (1.7 → 1.1 decisions/s).

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
Superseded but kept: `train_baselines.sh`, `train_all_retok.sh` / `eval_retok.sh` /
`relabel_retrain_retok.sh` (the matched-tokenisation arms), the `*_new.sh` drivers that brought in
fa/ar/la/sa/lzh/ja, and `rebuild_sa_csl_rev.sh` (+ `hyphen_to_pipe_sa.py`, `strip_pipe_sa.py`) for
the pausa-normalised Sanskrit representation.
`spacy train` writes scores to `train_*.log`; `spacy evaluate --output` writes `metrics_*.json`.

## Conventions and invariants

**Naming.** English artifacts are unsuffixed (`corpus/`, `training/`, `metrics.json`); other
languages take `_<lang>`. Relabel variants: `_rl` (verb scope), `_ext` (extended scope), `_rl2`
(contrastive-prompt rerun). Layers: `_seg`, `_morph`, `_lemma`, `_sud`. These compose
(`training_ko_retok_rl/`, `metrics_zh_simp_rl_gp.json`); `metrics_*_{gp,raw}.json` are the
gold-preproc and raw end-to-end evaluations.

**gold_preproc (essential for every language but en).** `spacy evaluate` re-tokenises raw text with
the model's tokeniser; a mismatch with gold tokens collapses alignment (Korean LAS once dropped to
~30). Configs set `gold_preproc = true` and **evaluation must pass `--gold-preproc`**. en (spacing
matches) doesn't need it. All research metrics in this file are gold-preproc unless marked raw.

**Editing configs programmatically:** load with `Config().from_disk(p, interpolate=False)` — the
default interpolation resolves `${paths.train}` to null and silently breaks CLI path overrides
(this caused `E913`).

**The freeze recipe** (how every layer above the base arm is added): source the arm's existing
components, **freeze** them, and train ONLY the new component, giving it its **OWN small
`HashEmbedCNN`** (width 64 / depth 3 / embed 2000) rather than a listener. A dedicated encoder is
immune to treebanks whose XPOS is orthogonal to UPOS (id: 33/46 XPOS values map to >1 UPOS), and
co-training is dominated (see NEGATIVE-RESULTS). Frozen components come out **byte-identical**
(verify per-arm with `cmp` on `*/model`), so lower-layer metrics need no re-verification. sa is the
one exception — it now ships a **joint multi-task** arm (see the Sanskrit section).

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

## The layer stack

Each layer was added to all eleven arms and re-released over v0.1.0 (clobber).

- **Sentence segmentation** (`make_seg_config.py` → `config_<lang>_seg.cfg`, `seg_code.py`,
  `retrain_seg.sh`; `regen_idko_corpora.sh` rebuilds the cleaned-up id/ko corpora). `gold_preproc`
  had a hidden cost: the parser saw one pre-segmented sentence at a time and never learned to
  *start* one, so raw multi-sentence input collapsed into a single tree (raw `SENT F` 0). Fixed by
  training through `sud.GoldTokCorpus.v1` with `sents_f` weight 0.05. **Raw end-to-end LAS / SENT F,
  old→new:** ar 69.4→72.4 / 0→66, fa 79.2→85.3 / 0→99, ja 81.9→85.8 / 0→96, id 68.3→73.4 / 0→87,
  ko 68.6→74.3 / 0→90, la 63.9→70.9 / 0→74, zh 54.3→57.4 / 0→99, yue 52.0→60.0 / 2→81. `en` was
  already fine; lzh/sa segment via `clause_parser` (their treebanks carry no in-text boundaries).
  Note `spacy train --code` takes ONE file (unlike `package`), which is what `seg_code.py` is for.
- **UPOS morphologisers** (`make_morph_config.py` → `config_<lang>_morph.cfg`, `train_morph.sh`).
  The pipelines were `[tok2vec, tagger, parser]` and the tagger predicts XPOS, so `token.pos_` was
  empty. The config
  derivation nulls `init_tok2vec` (else yue's Mandarin-init `zh_both_tok2vec.bin` clobbers the
  sourced encoder) and keeps only factory args common to the standard **and** `ja.morphologizer`
  factories (the latter rejects `label_smoothing`/`overwrite`/`extend`). **`pos_acc`:** en 0.934,
  ar 0.946, fa 0.960, ja 0.967, id 0.928, ko 0.939, la 0.955, zh 0.896, yue 0.911, lzh 0.912,
  sa 0.877. ~+2 MB per wheel.
- **Lemmatisers** (`make_lemma_config.py` → `config_<lang>_lemma.cfg`, `train_lemma.sh`). A
  `trainable_lemmatizer` (edit-tree, `backoff="orth"`) learns FORM→LEMMA string edits, so it is
  language- and script-agnostic.
  **`lemma_acc`:** en 0.949, ar 0.907, fa 0.981, ja 0.979, id 0.957, ko 0.977, la 0.936, sa 0.848,
  zh/yue/lzh 0.999 (lemma ≈ form). ~+1 MB.
- **SUD MISC layer** (`train_sud.sh`, `package_sud.sh`) — see its own section below.

## Tokenisers

Direction per language is chosen by whether the treebank tokenisation is a deterministic function of
the text. **Only en and ko-eojeol are deterministically matchable**; zh and id stay statistical, so
keep `--gold-preproc` for fair parser comparison. Relabel decisions are tokenisation-agnostic, so
`relabel_retrain_retok.sh` relabels at the original tokenisation (cached) and **transfers** labels
through transforms rather than re-querying.

**Treebank-trained character segmenters** (`sud.CharSegTokenizer.v1`, `char_seg_tokenizer.py`,
`make_seg_pairs.py`) now serve zh and id: `sa_presegment`'s character tagger reused verbatim, one
rewrite label per character (`=` keep, `= ` word break, `=-` compound break), greedy argmax. The
wrapper serialises the segmenter and its lexicon beside the weights, so a wheel is self-contained.

| lang | tokeniser | note |
|---|---|---|
| **zh** | char tagger + jackknifed corpus lexicon + **jieba's BMES decision** | strict token F 0.8385 (pkuseg) → **0.9210** |
| **id** | char tagger, enclitics SPLIT | replaced `coarsen_id.py`'s merge; `-nya` now gets its own `mod@poss` |
| **ko** | eojeol, spaCy's RULE tokeniser | 0.3070 → **99.77** against the ORIGINAL SUD_Korean-GSD |
| **yue** | pkuseg trained from scratch on yue | word-F1 0.95 vs char 0.63 |
| **lzh** | `lzh_tokenizer.py`, one Han char = one token | no spaCy `lzh` module exists |
| **sa** | `sa_tokenizer.py` + CSLiser front end | see the Sanskrit section |
| **la** | rule tokeniser + `la_tokenizer.py`, enclitic `-que` SPLIT | Perseus strict token F 0.9738 → **0.9944** |
| **en, fa, ar** | rule tokeniser | en already at the rule ceiling (F1 0.991) |
| **ja** | SudachiPy | |

**Latin `-que` is rule-separable, and the failure was invisible to every metric.** spaCy's stock `la`
rules split nothing ending in `-que`, but ITTB and Perseus write `Animosque` fused and analyse it as
`Animos` + `que` (CCONJ, `cc`) — so real classical orthography reached the model with a token
boundary missing. Three things hid it: PROIEL respaces its own `# text` (`ne que mittatis`), so it
never fuses; **not one of the 198 MWT range lines carries `SpaceAfter=No` on the host** (a CoNLL-U
convention — nothing has to say there is no space inside a multiword token), so `spacy convert`, which
drops range lines, rebuilt the corpus text SPACED; and `gold_preproc` bypasses the tokeniser at
evaluation. `fuse_mwt_spaceafter.py` makes the convention explicit (MISC only, idempotent, 2 979
sub-tokens across the derived la family).

Unlike Indonesian this needs **no trained segmenter**: the productive side needs no lexicon, since
any host may take the enclitic, and the exception side is CLOSED. `build_la_enclitic_lut.py` harvests
it (80 forms in `la_enclitics.py`) — PROIEL is the best evidence, because a single token it leaves
ending in `-que` is lexicalised by its own analysis, and it is the only source for accidental endings
(`relinque`, `oblique`, `aeque`) a naive suffix rule would maul. Held out on test, "split unless in
the list" errs on **7 of 189** `-que` words (3.70 %) with **159 of 189 unseen in train**, so it
generalises; three of the seven are Perseus leaving a productive `-que` unsplit (`Aethiopasque`), i.e.
our rule right and the gold wrong. Lookup is on the lowercased macron-free form, so macronised input
matches without enumerating vowel lengths — which also keeps Morpheus-derived data out of a
CC BY-NC-SA wheel. **`-ne`/`-ve` are deliberately left alone**: 3 splits in 1013 and 0 in 4, against
thousands of ablatives in `-ne` (`ratione`, `ordine`).

**No retrain** — `gold_preproc` + `sud.GoldTokCorpus.v1` means the parser is segmenter-agnostic, so
`add_la_enclitic_tokenizer.py` swaps it into the released arm and `--verify` confirms all seven
component weight files come out byte-identical. Perseus test, raw end-to-end: TOK 98.25 → **99.70**,
POS 81.01 → 82.13, UAS 62.97 → **65.19**, LAS 51.31 → **53.35**, SENT F 88.21 → 89.79; ITTB+PROIEL
unchanged (TOK 99.99 → 100.00, LAS flat); gold-preproc metrics identical to the decimal.
⚠ **Assigning `nlp.tokenizer` does NOT update the config.** `to_disk` writes the config as it stands,
so the reloaded model rebuilds a stock `spacy.Tokenizer.v1` and `from_disk` quietly refills it with
the base rules — it loads, runs, splits nothing, and says nothing. `nlp.config["nlp"]["tokenizer"]`
must be set too (as `bundle_zh_charseg.py` does), and the swap script now re-verifies the RELOADED
model rather than the in-memory one. Like `ar_tokenizer`, the la tokeniser publishes
`doc.user_data["mwt_ranges"]` so a CoNLL-U consumer can write `12-13 Animosque` back out.

**The lexicon feature only works JACKKNIFED**, and it is the whole ballgame. A word list harvested
from train covers 100 % of train and 87.6 % of test, so the model learns a reliability the feature
will not have and never develops a fallback. Naive it is worse than useless; per-fold
(`--jackknife K`, train-time coverage → 87.0 %) it is the biggest single win:

    zeroed channel (capacity control)   94.37 dev / 0.8802 test
    naive corpus lexicon                93.00 dev            <- below the control
    jieba dictionary, external          94.79 dev / 0.8859
    corpus lexicon, JACKKNIFED          95.20 dev / 0.8902

**jieba's DECISION, not its dictionary** (`scripts/zh_jieba_feature.py`) is what took zh past that.
The earlier experiment asked jieba for substring MEMBERSHIP; this reads its segmentation off per
character as a **BMES** code (4 values, so the channel is exactly a lexicon channel's width and the
capacity control stays parameter-identical). Headroom was measured first: jieba boundaries are
P 0.9730 / R 0.8793 (very precise, under-splitting), and **jieba is right at 72.5 % of the positions
the shipped model gets wrong** — reachable only by a feature, since 97 % of the segmenter's errors
are confident ones. Strict token F on test, 5 runs each (`scripts/eval_zh_seg.py`; model init is
unseeded, so the spread is real):

| arm | mean | range |
|---|---|---|
| shipped `zh_seg_jk` (corpus lexicon jackknifed, `n_sources` 1) | 0.8898 | one fixed model |
| capacity control: same + a ZEROED second channel (`n_sources` 2) | 0.8761 | 0.8679–0.8855 |
| **corpus lexicon + jieba decision** (`n_sources` 2) | **0.9203** | 0.9188–0.9210 |

Going 1 → 2 sources COSTS 1.37 on its own (char embed 56 → 48), so the feature is worth **+4.42
against its own architecture**. Raw end-to-end (`scripts/eval_zh_raw.py`, `training_zh_lemma`, same
100 docs): control 0.4673 LAS (0.4361–0.4945) vs **0.5269** (0.4979–0.5608). **Never quote zh raw
LAS from a single segmenter run** — a 0.22-point token-F spread produces a 6.3-point LAS spread,
because a few misplaced boundaries misalign whole spans in the scorer. An earlier draft claimed
"+2.51 LAS" from exactly that mistake.

**The `pad` regression (fixed, but it only reaches retrained models).** `build_lex_model` dropped the
`pad` argument `build_model` has always had, so thinc's `_list_forward` inserted no zero rows between
sequences and layer 1's window read across sequence boundaries. The reach is **one character**
(|Δ| 0.813 at position 0, 0.0013 at position 1, 0 after 2), so every affected zh example was a
sentence-INITIAL split (`台大`→`台/大`). Measured across the family, only zh moves at all (−0.27);
id/yue/sa are identical batched vs per-text, so no published sa figure needs revising. Two things to
know: **`Model.from_bytes` restores `attrs` from the checkpoint**, so an existing model keeps
`pad: 0` and its old behaviour — an architecture fix does not show up until you retrain; and the
single wrapper covers the embed too, so pad rows arrive as character id 0 = `PAD_CHAR` (a reserved
index, deliberately) rather than as zero vectors. `eval_zh_seg.py` predicts per text by DEFAULT;
`eval_seg_batching.py` reports all three groupings.

**Superseded, but the reasoning still applies.** zh once bent the *tokeniser* to the treebank
(pkuseg trained on GSDSimp, `train_pkuseg_zh.py`; word-F1 ~0.88 vs jieba 0.80 — NB the ~0.94 pkuseg
reports is the lenient *cut-point* F). id once coarsened the *treebank* (`coarsen_id.py` merged each
MWT host+enclitic, since `-lah` is a clitic 73× but inside whole words 1723× and not rule-separable).
ko once retokenised to mecab morphemes (`retokenize.py --lang ko`) with eojeol-internal
**functional-head** structure per mSUD — case particle (ADP) / verbal ending (AUX) heads, lexical
stem is `comp:obj`/`comp:aux`, evidenced by `mSUD_Nenets-Tundra`. `retokenize.py` also has a general
char-span align + reproject path (merge/split/crossing + cycle/root repair), and holds a
**reversibility invariant**: new tokens are surface substrings carrying `SpaceAfter=No`, so
concatenation reproduces the text. Matching the tokeniser to the treebank is usually better than
re-tokenising the treebank, and only needed at all when you want a *different granularity* than the
treebank has.

⚠ **"Match the treebank" was ambiguous for ko and the two readings differ hugely.** The mecab arm
scored TOK 1.0000 against ITS OWN retokenised treebank and 0.3070 against the original. Switching to
eojeol discards the Korean case-particle relabel result (`comp:obl` F 0.169 → 0.386), which needed
the particle as a separate token — accepted by user decision.

## The `udep` relabel (the research contribution)

**Scripts.** `disambiguate_pp.py` is the foundation module imported everywhere (`parse_conllu`,
`descendants`, `render`, and `query` — the canonical Ollama call). `build_gold.py` (en) /
`lang_gold.py` (others) build the *confident* comp/mod benchmark from unambiguous `udep` cases:
COMPLEMENT = the verb lexically selects the adposition (curated `(verb, adp)` frames); MODIFIER =
temporal/causal adposition or temporal object (note the temporal-object override — a frame with a
year object is a modifier, "believe in 1999"). SUD's own committed labels are too sparse and noisy to
serve as gold, hence the rule build. `eval_prompts.py` / `lang_bench.py` benchmark prompt variants
(`eval_prompts.PREFIXES["fewshot12_def"]` is the canonical English prompt; `en_errors.py` is the
error analysis that drove the contrastive shots);
`zh_bench.py`/`id_bench.py`/`en_bench.py` hold curated same-adposition contrastive few-shot.
`relabel.py` / `lang_relabel.py` apply the chosen prompt (rule first, model only for the genuinely
ambiguous remainder), resumable via on-disk `relabel_cache*.jsonl`. `relabel_ext.py` covers the
**extended scope** into separate `*.relabeled_ext.conllu`. `udep_audit.py` / `udep_probe.py` /
`hard_examples.py` are the analysis behind the scope decisions.

**Prompts are static prefix (definitions + few-shot) + short variable suffix (the sentence)** so
Ollama reuses the cached prefix KV (~4× speedup). Keep them that way.
Block-based rewriters preserve the file byte-for-byte except target DEPREL cells — verify round-trip
before long runs.

**Extended scope** adds: ADP dependents of NOUN/PROPN/ADJ heads; clausal verb PPs; participial complex
prepositions (`according/based/following` → mod); a Korean case-suffix rule; zh 的/之, lzh 之, ja の
associative PART → mod; ko ADV-of-VERB → mod. **Partitives (NUM/DET/PRON heads) stay `udep`** —
SUD's documented default, by user decision.

**Results, `comp:obl` F, base → verb-rl → ext** (LAS within ~1 throughout):

| | base | verb-rl | ext | |
|---|---|---|---|---|
| id | 0.463 | 0.565 | **0.703** | prepositional, genuinely ambiguous |
| fa | 0.705 | 0.815 | 0.794 | ext dilutes an already-strong verb set |
| ja | 0.000 | 0.720 | 0.688 | GSD commits **no** `comp:obl` — the class is synthesised from scratch |
| ar | 0.617 | 0.659 | 0.634 | |
| la | 0.678 | 0.691 | 0.684 | |
| lzh | 0.716 | 0.659 | 0.701 | ext = the coverb rule below |
| en | — | 0.740 | 0.730 | large well-disambiguated verb set already |
| zh | 0.190 | 0.307 | 0.356 | |
| ko | 0.169 | 0.247 | 0.386 | at eojeol tokenisation this result is not yet reproduced |
| sa | 0.404 | — | 0.352 | case-based; stays un-relabelled |
| yue | 0.308 | 0.261 | 0.348 | |

**Findings.**
- Relabelling lowers headline **LAS by ~1–2** everywhere (the binary is harder than the noncommittal
  label) while **UAS is unchanged** — only labels change. The metric that reflects disambiguation
  quality is per-label **`comp:obl` F**. Caveat throughout: each relabel rewrites the *test* gold
  too, so `comp:obl` F has a moving denominator.
- Value scales with how genuinely ambiguous the adpositional system is: high for prepositional
  systems (en/id/fa/ja/ar/la), near-vacuous where the `udep` adpositions are ~all circumstantial
  (lzh's plain-`udep` residue) or where the system is case-based (sa).
- **Korean is not near-vacuous** once the case suffix on noun dependents is used — the verb-ADP view
  simply missed where its signal lives. Same lesson as lzh's locative complements.
- **Two `udep` families**: prepositional (fa/ar/la/lzh/ja/zh/yue/en/id — the adposition is the ADP
  head of the NP) use the verb-frame gold; case-based (sa, ko) use the dependent's morphological
  Case or case particle.
- Per-language relabel model: fa/sa/lzh/en/id/zh/ko/yue → qwen3:8b, ar/la → gemma4, **ja → qwen3 with
  a native-Japanese prompt**.

**Language-specific rules worth knowing.**
- **lzh coverbs.** The bulk of the signal is not on plain `udep` but on the annotators' **subtyped**
  `udep@lmod` (locative, ~3029) and `udep@tmod` (temporal, ~105), which the plain scope never
  reached. Decided from the annotators' own category + the head verb's class (XPOS field 3):
  **@tmod → mod**; **@lmod → comp:obl** only under a locus-selecting verb class (移動 motion / 姿勢
  posture / 設置 placement / 存在 existence / 生物 birth-death), else mod. This commits ~815 test
  coverbs and nearly doubles the comp:obl class (182→355) at precision 0.72 with LAS flat. Object
  FEATS `Case=Tem`/`Case=Loc` is the same signal for plain-`udep` coverbs.
- **lzh 於 routing.** After Loc/Tem, the residue splits ~evenly person 958 / non-person 912. The
  treebank commits **0 comp:obl and 0 mod on 於+person** (recipient-dative vs comparison vs
  passive-agent — maximally ambiguous, and inherently unvalidatable since there is no gold), so only
  the LLM can adjudicate it; 於+non-person IS committed (84:54) so a loose frame rule fits
  (`COMP_FRAMES["lzh"]` derived at minc=2/thresh=0.70 → ~15 frames: 至於/達於/在於/異於/甚於/長於/怒於…;
  the default minc=8/thresh=0.85 yields none). The rule intercepts cases *before* the cache, so no
  re-querying is needed.
- **Sanskrit case rule.** Recipients are **dative** (confirmed in-treebank: dā/prayam+Dat), not
  locative — the locative-of-locus is the Vedic ritual `hu` "offer into fire-LOC", which SUD leaves
  `udep`/mod. So Loc/Abl/Voc/Nom → mod, recipient datives → comp via (verb, Case) frames; blanket
  Dat → comp is avoided (the dative-of-purpose is adjunctival).

### `udep` beyond comp/mod: derived rules commit 10 730

`relabel_ext.py` asks one question, so anything that is not an adpositional or case-marked oblique
stays `udep` — 32 415 tokens over nine treebanks, dominated by material where no oblique/modifier
choice is being deferred: Persian's relativiser که (5060), English `'s` and infinitival `to` (950),
Japanese adnominal/copular た/だ (355).

**`udep_residue_audit.py`** answers "what SHOULD this be?" from the treebank's own committed
decisions — for each residual token, the DEPRELs annotators used for the same (head UPOS, dep UPOS,
dep lemma) signature. **`apply_udep_rules.py`** commits what is dominated past 90 % on ≥ 20 committed
examples, writing `*.udep_ruled.conllu` (DEPREL column only). Rules are DERIVED, never hardcoded.

    fa 7156  (NOUN<-SCONJ که -> mod, 98 % of 375)      lzh 1834 (VERB<-NOUN 今/後/初 -> comp:obj)
    ja  802  (NOUN<-AUX た/だ -> mod, 99 % of 1392)     en 526   ar 311   zh 54   id 33   ko 12   yue 2

Japanese is the clearest set: NOUN<-AUX た is a relative clause (the tense auxiliary heads it), だ is
the adnominal copula な, VERB<-AUX だ the adverbial に — the same copula in both non-finite guises,
recovered independently by the evidence. fa/lzh/ja retrained and re-released
(`retrain_udep_ruled.sh`; fa LAS 87.18, lzh 79.01, ja 88.21). The point is OUTPUT CORRECTNESS, not
accuracy: on 40 test sentences the old fa model emitted `udep` on 34 and `mod` on 4, the new one
`mod` on 35. en/ar (0.22 %/0.10 %) and id/ko/zh/yue (≤ 0.05 %) were skipped. Pre-rule treebanks kept
as `*.pre_ruled`. **fa also needs its `_sud` arm rebuilt** — it ships from a Subject layer stacked
above the lemma arm, which the base chain alone would miss.

An LLM pass over the remaining residue was built and **abandoned** — see NEGATIVE-RESULTS.

### Korean eojeol relabel (committed 2026-08-04)

424 DEPREL cells across `assets_ko/SUD_Korean-GSD/ko_gsd-sud-{train,dev,test}.relabeled_ext.conllu`
(313/53/58), all `udep` → `mod` (392) or `comp:obl` (32); DEPREL is the only column touched. This is
the extended relabel rebuilt at the eojeol granularity the released arm now uses, backed by +138
entries in `relabel_cache_ext_ko.jsonl` (112 modifier / 26 complement). **None of it reached the
v0.1.0 wheels** — the shipped `ko_sud_gsd` predates it, so its `comp:obl` F 38.6 does not include it.

## SUD relation conformance (`normalise_reparandum.py`)

Audited against the guidelines: `conj` is correctly **chained** (each conjunct → the previous, `cc` →
the conjunct it precedes) in every treebank and no transform disturbs it; `appos` is never emitted
bare (apposition is the sanctioned `conj:appos`, 46 260). The UD relation **`reparandum`** survived
un-converted in a few upstream SUD releases and is rewritten to SUD's **`conj:dicto`** — 696 across
all derived files, distinct instances la 32 / yue 165 / zh 2, **DEPREL column only** (`reparandum` is
also a Latin gerundive word form, so FORM/LEMMA must be untouched). A pure label rename; la/yue/zh
bases were retrained so released models emit it. Other non-official UD carry-overs (`mod@poss`,
`@unmarked/@desc/@predet/@preconj`, `compound@prt`) were left as-is by user decision; `@lmod/@tmod`
and the other language-specific semantic subtypes are legitimate SUD conventions the pipeline relies
on.

## Language-specific notes

### Latin (`la_sud_ittb_proiel_perseus`)

Trains on a plain `cat` of **ITTB + PROIEL + Perseus** (each keeps its own sent_ids);
`add_perseus_la.sh` is the reproducible driver (`merge|macron|relabel|train`). Perseus ships only
train + test, so it is added train→train / test→test and dev stays ITTB+PROIEL.

- **XPOS blanking (non-obvious).** The three treebanks use mutually-incompatible XPOS tagsets.
  ITTB+PROIEL already mixed two and coped (TAG ~92), but Perseus's sparse 9-position tagset on ~1.3k
  sents tags at ~34 % and tanks combined TAG/LAS. `blank_perseus_xpos.py` blanks field 5 on the
  Perseus tail of each split; UPOS/FORM/dependencies are kept so Perseus still trains the parser.
  Orthogonal to the macron (FORM) and relabel (DEPREL) transforms.
- **Results (ext+macron union = release).** Apples-to-apples on the ITTB+PROIEL test: LAS
  77.7→**78.3**, UAS 83.1→83.8, `comp:obl` F ~69 — Perseus *improves* the original domain.
  Perseus-only test LAS ~54.6 (classical poetry, genuinely hard). The combined-test headline (LAS
  73.9) is lower only because the test now includes Perseus.
- **Macrons in.** One union parser handles plain + macronised input
  (`train_la_ext_macron.sh` trains on plain-ext ∪ macron-ext; `macronise_la.py` uses the Alatius
  Docker macroniser, `transfer_macrons.py` composes the FORM transform onto the ext deprels).
- **Licence: CC BY-NC-SA** — all three sources are NonCommercial, the only such released model.

**Macronised OUTPUT: the `la_macronise` component.** The inverse of the above. Alatius = RFTagger +
a Morpheus-derived lexicon; our pipeline already does the tagging half, so only the lookup remains.
`build_la_macron_lut.py` harvests it from the plain/macron treebank pair, keyed `L1 (form,upos,feats)`
/ `L2 (form,upos)` / `L3 (form)`, backoff-pruned 152 443 → 42 817 entries. **Two tables now,
cascaded**: the harvest falls through to **Morpheus itself**, fetched at runtime by
`fetch_morpheus()` (4 MB on the wire → ~2.2 MB in `~/.cache/sud-spacy/`).

| | harvest has the word (92.1 %) | it does not (7.9 %) | whole-token |
|---|---|---|---|
| harvested alone | 98.23 % | 52.46 % | 94.42 % |
| Morpheus alone | 93.98 % | 90.42 % | 93.71 % |
| **cascaded** | 98.23 % | 90.42 % | **97.63 %** |

On Perseus (OOV share 7.9 → 23.8 %): 87.02 / 95.75 / **97.33**. Agreement with Alatius on held-out
test is 94.09 % whole-token / 97.13 % per-vowel with predicted morphology — *agreement*, not gold
vowel length (Alatius is itself ~98–99 %). Morphology is matched through a nine-slot key
(`ud_key`/`ldt_key`) with a backoff LADDER rather than one exact key, because the tagger is imperfect
(`cano` → ADJ) and an exact key turns every mis-tag into a total miss. The S4/S3 suffix levels are
**gone from the builder** (still read for older tables): at 52.46 % they were barely better than a
coin toss, which is what the Morpheus fall-through replaced. **The bound was VOCABULARY, not
morphology** — OOV levels were 71 % of all errors from 8 % of tokens, and perfect morphology would
have bought +0.23.

`_PARADIGM` is a small exceptionless override keyed on `(InflClass, Case, Number, final letter)`
(a-stem Nom/Voc sg `-a` short, Abl sg `-ā` long; o-stem Dat/Abl sg `-ō`; e-stem Abl sg `-ē`) plus
`_LEMMA_CLASS` (PROPN carries no `InflClass` in ITTB/PROIEL, so declension comes from the lemma's
ending). It exists because the table memorises pairs and cannot express a paradigm rule — nominative
`Gallia` came out `Galliā` because the treebank only attests the ablative. **The harvested data is
WRONG on these cells** (Alatius's RFTagger contradicting gold morphology), so the rule *lowers*
measured agreement (−0.03) while raising real accuracy. It faithfully transmits Case errors, so
`config={"paradigm": False}` disables it.

Purely additive: spaCy tokens are immutable, so it sets `token._.macron`/`doc._.macron` and never
touches `token.text`. `la_parse_macronised.py` joins parse + macrons into CoNLL-U or a table;
`--mode reparse` re-parses the macronised string (measured free by `eval_la_reparse.py` over 4300
sents — LAS 70.30 plain / 70.33 ours /
70.31 Alatius) but `attach` is the default for being one pass and parse-identical.

⚠ **The component ships WITHOUT its data** (licensing): Morpheus is CC BY-SA 3.0, which forbids
adding restrictions, and the la wheel is CC BY-**NC**-SA. `package_lemma.sh la` runs
`add_la_macronise.py … --no-lut`. Users get data by calling `fetch_morpheus()` once (no config — the
component finds the cache itself), or `bash scripts/build_la_macron.sh` for a harvested table to
cascade on top, or both; route (1) needs none of Docker, the Morpheus compile or RFTagger.
**Fetching is not redistributing** — GPL restricts distribution,
not use — which is why the easy route exists, and the cache sits outside the model directory so the
wheel stays clean. With no data it **degrades, it does not raise** (`require_data=False`): every
token passes through unchanged and ONE `RuntimeWarning` per component instance names both routes —
necessary now the pipe is in the DEFAULT pipeline, since a raising component would break every
ordinary `nlp(text)`. The warning's fetch line is built from `__name__` so it reads correctly from
inside the wheel. RFTagger (non-commercial) is used only to label the treebank offline.

### Sanskrit (`sa_sud_vedic_ufal_dcs`)

The most involved arm. It accepts **raw sandhied text** (IAST or Devanagari); CSL is an internal
representation no caller produces. Assembled by `scripts/add_sa_frontend.py`:

    tokenizer  sa.SanskritInputTokenizer.v3, carrying TWO trained models
                 stage 0  CSLise     raw text -> CSL   (`sa_presegment`, char tagger, 4.8 MB)
                 stage 1  de-CSLise  CSL -> tokens + Compound  (mechanical, exact)
                 stage 2  de-sandhi  MWT members -> unsandhied (`sud_unsandhi` transducer)
    sa_compound   FIRST      Compound for callers who pass TOKENS rather than text
    ... trained components ...
    clause_parser            per-sentence re-parse, punctuation morphology
    sa_deva       LAST       Devanagari FORM/LEMMA + Translit/LTranslit, iff input was Devanagari

Per-token output: `token._.unsandhied` (padapāṭha), `token._.translit`/`_.ltranslit`,
`token._.src_span` (character span in the RAW input). Wheel ~20 MB.

**Accuracy on classical/epic (the target domain):** CSLiser sentence-PM 80.17 IAST / 78.63
Devanagari; de-CSLizer exact; de-sandhifier 0.9885; morph 0.9259; pos 0.9597; lemma 0.9592. On Vedic
the same arms score morph 0.7787 / lemma 0.8719 — the deliberate cost of DCS being 90 % of the
morphology training data.

**Representation: DCS multiword tokens** (`restructure_sa_csl.py`, `rebuild_sa_csl_mwt.sh`). An MWT
is an **orthographic word**, per DCS's own readme, not a compound. Measured directly on DCS: fusion
covers bound junctions (compound / preverb / privative) and **vowel coalescence** only — NOT
avagraha (`ko 'nasūyakaḥ` keeps its space), NOT consonant-final + vowel-initial. Every token INSIDE
an MWT is unsandhied (0/219 internal and 0/182 final carry sandhi) because the range line holds the
sandhied surface; a token that IS its own orthographic word keeps its sandhied surface (36.2 %
differ). `# text` stays the **CSL** line (the tokeniser reads it — CSL carries the syntactic word
boundaries that let it split without solving segmentation first); the orthographic rendering is
emitted as `# text_ortho`. NB DCS is romanised IAST; the Devanagari treebanks differ (UFAL writes
`वह्निरिद्रः` solid because the script cannot render a bare consonant before a vowel).

**Cost, accepted by user decision:** against the previous pausa-normalised representation, tag_acc
−1.14 / UAS −1.13 / **LAS −1.68** (0.5601 → 0.5433), because standalone tokens no longer collapse
their sandhi variants — vocabulary 32 318 → 38 123 types, hapax 58.3 → 61.3 %. Bought: DCS
consistency, gold `Unsandhied` kept as annotation, a 100 %-exact tokeniser contract on 77 % of
tokens. As always here, the test FORMs themselves differ between arms, so comparisons across
representations are not strictly like-for-like — the trees are identical, only the surface changes.

**Sandhi machinery.**
- `external_sandhi.py` — forward classical external-sandhi engine (`join_pair`): vowel coalescence
  (savarṇa/guṇa incl. `a+ṛ` → word2 `r`, vṛddhi, yaṇ, ayādi **glide-preserving** — `e/o+V → ay/av`,
  NOT bare hiatus, so the junction stays reversible), visarga, `m→ṃ`, `-t→-d/-c/-j`, `-n→-ṃs/ñ/nn`,
  `t+ś→cch`, stop voicing; `internal=True` suppresses external-only rules. **No gold sandhied text
  exists**, so this is rule-based generation validated by round-trip + textbook unit tests.
  ⚠ The glide-preserving ayādi is load-bearing: with bare-hiatus ayādi, the `-a/-ā` hiatus would be
  ~33 % ayādi and the visarga-reversal rule wrong.
- `apply_vedic_sandhi.py` — applies it within each Vedic sentence. `generate()` chains junctions
  **sequentially left-to-right**; computing them independently mishandles single-character words,
  notably the emphatic particle `u` (`atha u āhuḥ` → `ath' u āhuḥ` wrong vs `ath' ô āhuḥ` right).
- `sa_csl_prep.py` (UFAL) — alignment-based CSL: Devanagari→IAST, re-segment MWTs, hard cases
  hand-corrected via `sa_ufal_csl_overrides.tsv`; typographic double quotes → guillemets `«»`.
- `desandhi_csl` (in `sa_tokenizer.py`) + `revert_csl_sandhi.py` — the reversal, in six stages:
  (0) dropped visarga, ayādi glide, yaṇ, and bare-glide `v`/`y` → the underlying vowel; (0.5) guṇa of
  a following vocalic ṛ/ḷ (word2's `r`/`l` before a consonant → `ṛ`/`ḷ` — unambiguous because no
  native word begins `r`/`l` + consonant); (1) the notation-marked coalescence and avagraha;
  (1.5) context-sensitive sibilant/palatal junctions, gated by a **gold-derived lexicon** of genuine
  consonant-final / ch-initial stems (`_SIB_FINAL` 17 / `_C_FINAL` 19 / `_J_FINAL` 22 / `_L_FINAL` 1
  / `_CH_INITIAL` 90); (2) deterministic final-consonant reductions (`-s`/`-r` → `-ḥ`, voiced stop →
  voiceless, `-ṃ` → `-m` before a non-sibilant consonant, `-nn` → `-n`); (3) the **law of finals**
  (`_LAW_OF_FINALS`, 57 entries, Whitney §141-2 + gold) normalising genuine consonant-final stems to
  their pausa form (`vāc`→`vāk`, `diś`→`dik` but `viś`→`viṭ`), applied to compound members too so a
  stem has one form regardless of position; (4) daṇḍa normalisation (any double daṇḍa → `‖`).
  Stage 0 must run before stage 1 (a coalescence-derived hiatus must not be read as a dropped
  visarga) and before stage 0.5 (which needs word2 still consonant-initial).
  **Punctuation:** non-coalescent sandhi applies straight ACROSS a sentence-medial mark (a comma, or
  a single daṇḍa when the sentence closes with a double one — a metrical, not phonological, break);
  a pause (`. ? !`, a double daṇḍa, a lone single daṇḍa) blocks every rule, since the flanking words
  already stand in pausa. The medial test is **document-dependent**, exactly as in `clause_parser`'s
  `sent_scheme="danda"`. Coalescent reversions stay adjacency-only (no mark can sit inside a fused
  syllable). Verified by inserting a mark at all 147 368 non-coalescent junctions: 0/206 440 reverted
  forms change.
  **Not reverted** (genuinely ambiguous even with the lexicon): `-j`/`-h` finals and word-final
  `-c`/`-ś`/`-ṣ` at pause, and `-ā`/`-a` before a voiced consonant. Known accepted collision: the
  ~0.8 % `-u`-stem vocative in `-o` (95 train tokens — `go`→`gaḥ`, `viṣṇo`→`viṣṇaḥ`); an `_O_FINAL`
  guard would fix it but would drift the corpus, so it is a candidate for the next rebuild.

**Tokeniser contract.** `sa_tokenizer.py` is the only tokeniser in the project that *rewrites* what
it reads, so `doc.text` is NOT the input and a token form is generally not a substring of it. Every
token therefore carries `token._.src_span`, the half-open span of the RAW input it came from
(extensions registered at `--code` import, `has_extension`-guarded since loading two models in one
process imports it twice). Mechanism: normalisation is not length-preserving, but **every boundary
this tokeniser can produce falls on a character that `normalise` maps 1:1** (whitespace, `_PUNCT`,
`-`), so the input is segmented at those anchors, normalised per segment and aligned; the piecewise
result is **checked** against `normalise(text)` and all spans drop to None if they disagree — an
exotic input can cost the offsets but can never change the tokens. In practice the spans **tile the
input exactly** (209 455/209 455 corpus tokens spanned); the only unspanned characters are the
inter-token spaces, and a compound member owns its join hyphen.

**`Compound=Yes` as an INPUT feature (+1.30 LAS).** sa is the first arm to READ morphology, and
every component's embed lists `MORPH`. Stripping the compound-join marker to leave clean wordforms
cost ~0.5 LAS because the marker had been *visible* inside the token form; this puts the cue back as
a feature. Deriving it needs `_NON_COMPOUND_JOIN` (23 types): the CSL hyphen joins samāsa members
(`Compound=Yes`), verb preverbs and the privative `a-/an-`, and excluding the closed
upasarga+privative list lifts the bare marker's P 0.775 / R 0.713 to **P 0.9998 / R 0.9997**.
Three pieces are needed and omitting any one breaks it silently: `sud.CompoundCorpus.v1` (under
`gold_preproc` the tokeniser never runs, so the feature would be absent in training and present at
inference — it copies **only** `Compound`, which is not leakage precisely because the tokeniser
supplies the identical value at runtime); `clause_parser` re-imposing the tokeniser's verdict (the
morphologizer overwrites `token.morph`); and explicit `MultiHashEmbed` + `MaxoutWindowEncoder` in
place of `HashEmbedCNN`, which hard-codes NORM/PREFIX/SUFFIX/SHAPE. Result: LAS +1.30, UAS +1.64,
`Compound` F 0.889→0.999, propagating to Mood/Tense/Person/Case.
**`sa_compound`** (first in the pipeline) re-derives the feat from token adjacency for callers who
pass pre-tokenised input — exact on real text (19 584/19 584, precision 1.0000). Token-input LAS:
0.5169 (no feat) / 0.5478 (fallback) / 0.5601 (full feat). Evaluate with
`scripts/eval_sa_compound.py`, not `spacy evaluate`, and do **not** add `sa_compound` to an eval
pipeline that already gets the feat from the reference (it overwrites 1 019 gold values with its own
737).

**Per-component affix windows (`sud.MultiHashEmbedAffix.v1`, `sud_affix_embed.py`).** `MultiHashEmbed`
plus one hash-embedded table per configured affix length, computed from the token string in the
forward pass — **exactly equivalent to the stock layer when no affix is configured** (verified
byte-for-byte by `check_affix_embed.py`), so switching an arm stays single-variable. Safe on the
*dedicated* encoders because the freeze recipe means only the new component sees it (all frozen
components verified byte-identical). Lexeme-level `SUFFIX` stays at spaCy's 3.
`make_sa_morph_arms.py` generates the arms (asserting each differs from the baseline **only** inside
`morphologizer.model.tok2vec`) and `train_sa_morph_arms.sh` runs and scores them.
**Adopt suffix 5 / 8 000 rows (+2.0 MB):** morphologiser +1.19 mean over three seeds (baseline
0.8050/0.8075/0.8087 vs 0.8208/0.8169/0.8191 — **no overlap**), lemmatiser +1.60. The features that
move are the ones an information-theoretic probe predicted in advance — Voice +17.4, VerbForm +6.6
(passive `-yate`, participial `-mānaḥ`, future `-ṣyati`, all outside a 3-character window) — and a
`w96` capacity control that widens the encoder *without* the feature buys only +0.44 with Voice
+2.9, so **the gain is the feature, not the parameters**. On the DCS representation it replicates:
morph_acc +1.33, pos_acc +1.34, lemma_acc +1.43. Do NOT put it on the base `tok2vec`
(see NEGATIVE-RESULTS).

**`sud_unsandhi`: a LEARNED sandhi reversal (0.9788 vs the rule's 0.9446).** Under DCS a standalone
token keeps its sandhied surface, so the padapāṭha form is no longer derivable from FORM — but the
treebank records it on 100 % of Vedic tokens, so it can be learned. It must be: the treebank wants
`saṃ`→`sam`, `udag`→`udak`, `nir`→`niḥ` but `prāc`→`prāc`, `catur`→`catur`, `tad`→`tad`. Identical
surface shapes, opposite answers — the choice is lexical. Implementation: spaCy's edit-tree
lemmatiser under the freeze recipe, trained through a corpus where LEMMA has been replaced by the
`Unsandhied` value (`make_unsandhi_corpus.py`), then re-homed into `SudUnsandhi`
(`add_sud_unsandhi.py`) which writes `Token._.unsandhied` instead of `token.lemma_`, so both
edit-tree components coexist. 28.9 % of training tokens need a genuine edit, so this is not a
majority-class win. **Known limit:** it predicts from the FORM the tokeniser produced, so it cannot
repair a token already de-sandhied wrongly (the ~4 % MWT-internal residue); standalone tokens, 77 %
of the total, are exact by construction, so the compounding is bounded.

**The CSLiser (`sa_presegment.py`, `train_samhita.py`).** A character tagger turning continuous or
spaced sandhied text into CSL. Gold data is synthesised, not annotated: CSL and the true sandhied
surface differ at exactly ONE junction class — vowel coalescence, which CSL splits to stay
reversible. `external_sandhi.COALESCE_SURFACE` is that one difference, **derived by iterating
`_coalesce`** so it cannot drift from the engine. `make_samhita_pairs.py` writes the triples with
zero round-trip failures. Labels are character-INDEPENDENT (59 of them), so the model learns "insert
a break here", not one class per character; the word-vs-compound distinction comes free from
`apply_vedic_sandhi`'s `bound[]` (Hellwig & Nehrdich 2018 conflate the two; we cannot, because the
compound divider is what makes `Compound=Yes`). The task is **local** — a pure count model over
character windows (`baseline_samhita.py`) plateaus at ±3 (36.07 F unigram → 67.67 at ±1 → 83.84 at
±2 → **86.00 at ±3** → 86.10 at ±5), so a small CNN suffices and 86 F is the floor.
Model: `Embed → depth × residual(expand_window + Maxout) → Softmax`, pure Thinc. Early-stop on dev
**split-location F**, not character accuracy (84 % of characters are "keep").
`models/sa_presegment_dcs`, trained on 20 164 Vedic + 172 966 DCS sentences: **97.91 split-loc F /
74.87 PM on DCS (the target domain)**, 93.98 / 53.75 on Vedic, 92.55 / 40.47 on unseen
Suśrutasaṃhitā — the honest number for arbitrary input. For scale: rcNN-SS 96.84 F / 87.08 PM,
TransLIST 98.86 / 93.97 on their own benchmarks, with far more data; **do not read these as
like-for-like**. Weakest frequent class is the compound break `=-` (F 60.2, recall 52.9) — precisely
what feeds `Compound=Yes`.

⚠ **`_cslise` fed the ortho CSLiser an input regime it was not trained on** (fixed 2026-08-04, worth
4.83 split-location F). `to_csl`/`_cslise` split the normalised text on spaces and predicted chunk by
chunk — CORRECT for the original CSLiser, trained on continuous saṃhitā with no space in its
vocabulary, and wrong for the released `sa_presegment_ortho`, 381 775 of whose 386 260 training rows
contain a space. Whole spaced string (as trained) 0.8731 split-loc / 0.7882 PM vs space-split chunks
(as deployed) 0.8248 / 0.7269, so the released model had been doing worse than its own published
numbers. **The fix asks the model instead of assuming**: `Presegmenter.reads_spaces` is True iff `' '`
is in the character inventory `build_vocabs` filled from the training rows. Same shape as the
unset-vs-empty MORPH bug — an invariant true of an earlier model, silently false for its replacement,
with nothing raising. Chunking stays for a continuous-saṃhitā CSLiser, where it is required.

**End-to-end budget.** `scripts/eval_sa_raw.py` scores span-matched — spaCy's `Example` aligner
CANNOT be used, because a wrong coalescence label emits different CHARACTERS, not just a different
split, so the two texts diverge.

| input | token F | LAS |
|---|---|---|
| gold tokens (gold-preproc floor) | — | 0.5457 |
| gold CSL (oracle) | 1.0000 | 0.5420 |
| raw saṃhitā, CSLiser output | 0.8699 | **0.4382** |

So the tokeniser + transducer together cost **0.4 LAS** and the CSLiser costs **10.4** — any further
effort belongs there. NB the oracle only reaches parity thanks to the unset-vs-empty MORPH fix;
before it, perfect tokenisation still lost 6.81 LAS.

**The released arm is JOINT MULTI-TASK**, breaking the freeze recipe every other arm uses:

    metric      3 encoders   1 encoder      metric      3 encoders   1 encoder
    tag_acc       0.8850      0.8908        dep_uas       0.6805      0.6514
    pos_acc       0.8866      0.8933        dep_las       0.5439      0.5140
    morph_acc     0.7787      0.7836        UFAL LAS      0.3873      0.4163
    lemma_acc     0.8719      0.8745        wheel size    25.85 MB    19.16 MB

Everything improves except parsing, and on held-out UFAL (classical prose, the actual use case) LAS
rises while Vedic falls. Accepted by user decision because the target is classical. NB the UFAL
figure rests on **416 tokens** and Vedic on 18 161, so the cost is far better measured than the gain,
and `spacy convert -n 10` makes a 60-sentence holdout only 6 resampling units.

**The README reports Sanskrit on UFAL, not Vedic**, since the arm was chosen for classical prose:
1843-token `corpus_sa_ufal_eval` test, gold-preproc, **UAS 52.2 / LAS 37.3 / `comp:obl` F 27.6 /
TAG 72.1 / POS 75.2**. This does NOT reproduce the 0.4163 above (a 494-token arm-selection holdout,
where the shipped arm scores 31.97). Three UFAL test sets give 32.0 / 37.3 / 41.6 — a ~10-point
spread driven by which few hundred tokens you pick, so treat any single classical figure as
approximate and prefer the largest test when publishing.

**Also noted, not done:** DCS (CC BY 4.0) carries REAL editorial sandhied text aligned with per-token
`Unsandhied=`, and the join is already present (`# sent_id = 70280_1` is a DCS sentence id + clause
index; MISC carries DCS's `LemmaId`/`OccId`). That is the only way to check whether
`external_sandhi.py`'s rule-based generation matches real sandhi, which the whole representation —
released model included — currently rests on unverified.

### Classical Chinese, Chinese, Cantonese

- **lzh** (`lzh_sud_kyoto`) has no spaCy module: `scripts/lzh_tokenizer.py` registers a custom `lzh`
  language + char tokeniser, loaded via `--code` and bundled in the wheel.
- **`clause_parser.py`** (lzh + sa, last pipe before the `sud_*` pipes) recovers per-sentence parses
  on punctuated editions, because both treebanks carry no in-text sentence boundaries and raw LAS
  otherwise collapses to ~48/~41. A **sentence** is the span between two sentence-final marks; within
  it the content tokens are concatenated **with sentence-medial marks removed** and parsed as one
  doc, then every mark is reattached as `punct` (to the head of its left unit if medial, to the
  sentence root if final). Which marks are final is set by `sent_scheme`
  (`add_clause_parser.py --sent-scheme`): lzh uses `""` + empty `sent_punct`, so every mark is final
  and each 句讀 unit is parsed in isolation; **sa uses `sent_scheme="danda"`**, a document-dependent
  rule (`?`/`!` plus an optional trailing closing quote always end a sentence; then only the period
  if the text has non-decimal periods, else only the double daṇḍa if any are present, else the single
  daṇḍa). Medial units are parsed together — the parser relates them, no fabricated `parataxis`.
  It also **normalises punctuation morphology**: with almost no punctuation in Kyoto/Vedic the tagger
  hallucinates content tags on it (？→名詞,糧食 "noun, food"; brackets become ROOTs), so every
  Unicode P* token is forced to `pos=PUNCT` + a deterministic XPOS (the Kyoto
  `s,記号,{句点,読点,括弧開,括弧閉}` map for lzh; sa sets `punct_tag = "PUNCT"` so the daṇḍa is not
  stamped with Japanese-tagset notation). gold-preproc eval bypasses `clause_parser`, so this is
  purely a raw-inference fix.
  ⚠ **`sent_punct` no longer defaults to ""** (which made EVERY mark sentence-final, so a sentence
  was one 句讀 unit). It now defaults to the genuinely sentence-final marks (`_SENT_FINAL`:
  。．.！？!?…।॥), with any run of closing quotes/brackets trailing one pulled back onto the sentence
  it closes (`」』）”»`, chained; an OPENING mark after a final mark starts the NEXT sentence). `""`
  is kept as an escape hatch for the old behaviour. `keep_marks` (default False) hands the
  sentence-medial marks to the parser instead of stripping and reattaching them by rule — correct
  ONLY for a parser trained on a punctuated treebank, and worth **+2.34 LAS** there, while costing
  **3.80 LAS** on one that has never seen a mark. Same input, opposite verdict: the setting is
  coupled to the arm underneath it.

#### Restoring punctuation to Kyoto, and relating the units (lzh)

Kyoto's README states it included no spaces or punctuation because Classical Chinese had none — so
the parser had seen **5 punctuation tokens in 374 560** and `clause_parser` had to strip every mark
before parsing. But the treebank was built FROM the Kanseki Repository (the sent_ids ARE kanripo
ids), the kanripo editions ARE punctuated, and both are CC BY-SA 4.0. `align_kanripo_punct.py` puts
the marks back: **100 193 inserted**, round-trip exact (drop the PUNCT tokens, renumber, and fields
2–8 come back byte-identical).

- **Align by CONTENT, not by identifier.** The sent_id's second field is a kanripo file number for
  論語 and 禮記 and nothing of the sort elsewhere (戰國策 numbers 046–501 against kanripo's 000–010).
  Global anchor resync, not a local window: kanripo files carry front matter the treebank never took,
  and a bounded search abandons the whole work.
- **The WITNESS is a git BRANCH.** 戰國策 aligns 0.3 % on `master`, 75.5 % on `WYG`, **93.6 % on
  `tls`** — whose 502 files match Kyoto's 046–501 exactly. A work that aligns badly is worth
  retrying per branch before being written off.
- Coverage: 禮記 99.7 %, 孟子/論語 100 %, 戰國策 93.6 %, 楚辭 93.8 %. **十八史略 only 49.8 %** (Kyoto
  sources it from its own `18shilue` path, not kanripo) and **KR4h0169 has no repository at all**;
  those pass through unpunctuated, which is fine — mixed input matches deployment.
- The real rate is **1.81 units per sentence**, with 1 841 of 3 062 test sentences a single unit. An
  early estimate of ~5 from 論語's boundary distribution was wrong — 論語 is comma-heavy, the corpus
  at large is 。-dense.
- ⚠ **A mark landing exactly on a unit boundary belongs to the unit it CLOSES.** `_bisect` alone
  assigns it to the FOLLOWING unit at offset 0, which put **2 780 of the sentence-final marks as the
  FIRST token of the next unit and none as a trailing one** — so `sent_final`/`sent_group` ran one
  unit late and every merged sentence was mis-segmented (spans beginning with the previous clause's
  comma and running past the ？ that should have closed them). **The round-trip check cannot catch
  this**: the text and the character offsets are correct either way, only the OWNERSHIP is wrong.
  There is now an explicit invariant — no unit may open with a sentence-final mark — checked per
  work at write time. Opening marks (「) are the deliberate exception, belonging to what follows.

**Relating the units** (`cross_unit_rules.py`) is the genuinely new annotation, since Kyoto relates
none. Rules are harvested from the annotators' own IN-UNIT clause-to-clause links at ≥ 90 %
dominance on ≥ 20 examples, `udep_residue_audit.py`-style. Three things that are not obvious:

- **Direction is a parameter and both occur.** 而 → `conj:coord` (99.7 %, n=4 517) attaches the
  following unit to the preceding; but 則 REVERSES it — the preceding clause is a `mod` of the
  則-clause (91.5 %, n=213). Harvesting post-head links alone misses 則 entirely, because its evidence
  lives in the pre-head configuration.
- **Lexical signatures only for closed classes.** 秦 opens a complement clause 63/63 times, but that
  is a state name; keyed on it the rule memorises the corpus. Open-class words go through their
  category, where an opening PROPN reaches 91.6 % and the individual names are noise. 曰 is the one
  declared exception (quotative frame, `parataxis` 91.2 % of 137), carved out by name so the
  principle stays intact.
- **The head helps, but asymmetrically, and only measured held-out.** Deriving on train and verifying
  on dev: opener alone 18.0 % coverage / 99.0 % accurate; **opener + GOVERNOR CLASS 21.4 % / 98.1 %**
  (以 alone is 52.4 % and unusable, but conjoined with 行為 it clears the bar); the dependent's own
  head lemma 100 % accurate but 2.9 % coverage; governor class or dependent class ALONE survive at
  zero rules. The opener dominates because Classical Chinese marks clause linkage at the LEFT EDGE.

Coverage reaches **37.1 %** of 41 498 boundaries. `若/雖/苟/縱` are declared from the grammar rather
than derived — 句讀 segmentation guarantees they never appear unit-internally (n=8/5/2/2) — and 如 is
deliberately EXCLUDED from that set, being 'be like'/'go to' as well as 'if'.

**The residue (62.9 %) is left as sentence breaks, not filled.** `--rules-only` merges only the
rule-derived boundaries; `--write` alone fills the rest with `parataxis` and is kept for comparison.
Measured on identical input (each punctuated sentence parsed as one doc, no `clause_parser`),
within-unit content edges — the only ones with real gold — against the RELEASED chain:

    released arm (no punctuation in training)   69.18 UAS / 61.91 LAS
    rules-only arm                              80.77 / 73.84      +11.93 LAS

The released arm collapses on punctuated editions because it has **never seen a mark** (5 tokens in
374 560) and attaches content words TO the punctuation — in 小大由之。 it makes the full stop the head
of three tokens, and in 信近於義 it hangs 信 and 於 off the opening 「. That is the failure this whole
exercise fixes, and it is a failure on the real editions users feed it.

The cost is on bare unpunctuated 白文, where the new arm has learned to rely on marks to join
clauses: **77.19 → 75.03 LAS (−2.16)**, single seed each. Both figures moved when the boundary-
ownership bug above was fixed (the cost grew from −1.34), so treat them as one measurement, not a
settled result — the seed replicates were **not** redone on corrected data. A comparison against the
full-merge `parataxis` arm was made only on the mis-segmented data and is withdrawn rather than
restated.

**The MISC layer survives the base change, but does not gain.** Re-measured end-to-end on the
corrected arm (the idiom rule reads the base's own `ExtPos` and `unk` as a CONJUNCTION, so it is the
most exposed thing downstream of a retrain): Idiom F **66.18** against the released 66.0, InIdiom
**66.18** against 68.8, both at precision 100 % with recall 49.45 — precision up, recall down, which
is this layer's standing pattern. Gold-trees mode stays 100 %. An earlier reading of 67.83 / 71.33
was taken on the mis-segmented data and is withdrawn.

**`merge_group` resolves subordinating edges FIRST.** In 子曰：「學而不思則罔」 the quote rule attaches
學而不思 to 曰 and then 則罔 claims 學而不思 — which looks like two edges fighting over one head. It is
not: 曰's complement was never 學而不思 but 罔. Laying down the backward edges, then attaching each
forward edge to `span_head(i+1)`, removes all 1 831 apparent conflicts rather than dropping them.
The counter is kept and prints a LOUD failure if the argument ever breaks.
- **Both Han scripts** (`zh_sud_gsd_simp_trad`, `lzh_sud_kyoto`; `both_scripts_release.sh`). zh trains
  on the two *real* treebanks for the same sentences — `SUD_Chinese-GSD` (traditional) +
  `SUD_Chinese-GSDSimp` (simplified auto-conversion) — **not** an OpenCC re-traditionalisation
  (simplification is lossy/many-to-one). The ext relabel lives on GSDSimp;
  `transfer_relabel_gsd.py` overlays it onto aligned GSD tokens (udep-only + alignment guard;
  comp:obl/mod is script-independent). lzh has no simplified counterpart treebank, so its simplified
  half IS OpenCC `t2s` of Kyoto (`opencc_conllu.py`, char-level, length-preserving). zh combined LAS
  69.3 / comp:obl F 32.6; lzh 79.0 / 70.9 — both within ~0.2 LAS across scripts.
- **Cantonese** (`yue_sud_hk`; `split_yue.py`, `yue_tokenizer.py`, `train_yue.py`,
  `train_pkuseg_yue.py`, `bundle_yue_pkuseg.py`). Coverb/prepositional like zh/lzh; ext adds
  associative 嘅 (PART → mod, like zh 的 / lzh 之 / ja の) and the annotators' `udep@tmod` (而家/今日 →
  mod). SUD_Cantonese-HK is **test-only** (1004 sents) → deterministic 80/10/10 round-robin split,
  which also copies empty XPOS←UPOS so the tagger predicts UPOS in `tag_`. No spaCy `yue` module.
  **tok2vec is Mandarin-init by default**: `config_yue.cfg` bakes `init_tok2vec =
  zh_both_tok2vec.bin` (extracted from `training_zh_both/model-best` via `model.to_bytes()`; it needs
  the `[pretraining]` component/layer block, because spaCy's cross-lang `source=` is blocked by E150
  vocab-lang). vs from-scratch: TAG +0.4–1.4, UAS +0.7/+1.2, baseline LAS +1.15; comp:obl F within
  100-sentence noise. Fine-tuning pkuseg from `zh_gsdboth` ties from-scratch (0.9474 vs 0.9472), so
  the self-contained from-scratch model ships; a userdict *hurt* (0.93).

### Indonesian

**FEATS bug (fixed).** `coarsen_id.py` hardcoded the merged token's FEATS to `_` (the same bug the
lemma column had), so `corpus_id_coarse_rl` had **0 %** non-empty FEATS despite ~42 % of source
tokens carrying real morphology — and `spacy train`'s own dev `MORPH_ACC` misleadingly read 100.00
(trivially correct against an all-empty gold). Fixed by using `rt.feats` (the `Tok` class already
carried it, `retokenize.py:29`, just never read); retrained to a real **`morph_acc` 0.909**.

**Lemma sentence-initial-casing fix.** The `trainable_lemmatizer` mis-lemmatised sentence-initial
capitalised **hyphenated** forms (`Anggota-anggota` → itself instead of `anggota`) even when the
capitalised instance was in training. Edit trees are literal-content substitutions, so a capitalised
token and its lowercase counterpart get two **different** trees, and sentence-initial capitalisation
makes the capitalised tree a near-singleton the classifier can't learn to select — the correct tree
already existed in the model (`trees.apply` gives the right answer); the classifier just doesn't
predict it. Plain capitalised words are unaffected (they share trees with hundreds of other simple
downcasings). Fixed by **`id_lemma_case_fix`** (appended after the lemmatizer like `clause_parser`):
overrides from a `FORM.lower()`→`LEMMA` table (`build_id_lemma_lut.py`, hyphenated forms only, 398
entries embedded as a dict literal), but **only** when the prediction equals the raw surface form and
the token is simple initial-cap — so it never touches an already-correct prediction.
Noticed but **not fixed** (pre-existing): the raw tokeniser inconsistently splits some capitalised
hyphenated reduplications (`Argumen-argumen` → 3 tokens) that lowercase forms tokenise as one.

### Korean

The eojeol arm reads the original `assets_ko/SUD_Korean-GSD`, whose FEATS is 4.7 % populated — so its
`morph_acc` 95.36 is ~the base rate for predicting empty and says nothing. POS 83.05 / lemma 78.30
are real.

## SUD's own MISC layer (`sud_misc.py`, `sud_idiom.py`, `sud_tagger.py`)

Output slot: **`Token._.sud_misc`** (a dict; `sud_misc.py` owns it with `set_misc`/`get_misc`/
`misc_string`, `has_extension`-guarded). `token.morph` is deliberately **not** touched, so a MISC
feature never masquerades as a morphological one. All keys go to MISC, following the treebanks — note
SUD's guidelines list `Subject` among the FEATS features, so data and prose disagree; we follow the
data. Gold transport is via `hoist_sud_gold.py` (see the MISC/convert gotcha above). Side effect: the
frozen morphologiser is then scored against gold FEATS carrying keys it never learned, so
**`morph_acc` in these arms' logs reads artificially low** — cosmetic, score weight 0.

### `Idiom`/`InIdiom` — exact, no training

SUD marks idioms with features, not a `fixed` relation: the head carries `Idiom=Yes` + an `ExtPos`,
other members `InIdiom=Yes`, unanalysable members attach by `unk`. Measured over train in all seven
treebanks that annotate idioms, that is an exact recipe:

    Idiom=Yes    <=> has ExtPos AND has an `unk` dependent                 P = R = 100 %
    InIdiom=Yes  <=> attaches by `unk`, and walking up through consecutive
                     `unk` links reaches a head with ExtPos                P = R = 100 % (la 99.9)

Both conjuncts are needed: `unk` alone gives `InIdiom` precision fa 6.5 / ar 53 / en 75 %; `ExtPos`
alone over-predicts `Idiom` in English (702 ExtPos vs 477 Idiom). Both inputs are already predicted,
so this needs **no training and no retrain** — appended at packaging time by `add_sud_idiom.py`.
End-to-end it is much lower, because it inherits the morphologiser's `ExtPos` and the parser's `unk`
errors; `eval_sud_idiom.py` reports both, and the gap is the honest measure. Test, gold trees →
end-to-end F: ja 100→96.8/95.7, en 100→84.6/82.1, sa 99.6→77.7/81.3, fa 100→72.7 (n=6), ar
100→67.3/68.4, lzh 100→66.0/68.8, **la 100→35.3/50.0** (la has only 489 train `ExtPos` in 586k
tokens, so `ExtPos` is almost never predicted). Precision holds (78–98 %); **recall is the limiter**.

### `Subject` — trained, but the rule wins in two languages

The **value** is determined by (deprel, head UPOS) at 100 % (zh 91 %) over 3–10 contexts per
language; the **presence** is genuinely lexical, and that is the hard part.

`sud_tagger` is a custom `TrainablePipe` because spaCy ships no generic token classifier (`Tagger`
hardcodes `doc.c[j].tag` and `get_aligned("TAG")`; a second `morphologizer` would wipe the first's
morph; `Token._.` is unreachable by `get_aligned`, E983). It subclasses `Tagger`, keeps
`spacy.Tagger.v2` unchanged, and overrides the output slot, the gold source and the scorer. **`O` is
an explicit negative class** — `Tagger` maps a `""` label to *missing* (no gradient), wrong for a
majority class that must be learned. `sud_subject_rule.py` is the lexical alternative: a
(head lemma, deprel, head UPOS) frame table from `build_sud_subject_frames.py`. Compared end-to-end
on test by `eval_sud_subject.py`:

    lang   trained F   rule F   ships     n(test)
    en       80.0       63.9    trained     266
    fa       89.5       71.6    trained      38
    la       66.3       53.0    trained     674
    yue      66.7       36.4    trained       6   (not meaningful either way)
    lzh      59.0       80.7    RULE        174
    zh       27.7       31.6    NEITHER     302
    sa       10.5       12.5    NEITHER      14   (142 train instances)

The split is principled: Classical Chinese raising rides on a handful of verbs (可/能/欲), which a
7-entry table captures and a small neural encoder cannot beat; en/fa/la raising has a long lexical
tail where the table's recall is fine but its precision collapses (en rule P 51 % vs trained 82 %).
**zh ships no `Subject` layer** — an annotation wrong two times in three is worse than none — and
since zh annotates no idioms either, **the zh wheel carries no SUD MISC layer at all**.

### `Reported` — bootstrapped from scratch

(`sud_reported_gold.py` builds the gold.) `Reported=Yes` occurs **zero** times in every treebank here
(the deprel form is 8 Latin tokens), so the class is synthesised, as ja's `comp:obl` was. It supersedes an older `parataxis:obj` analysis,
and that history fixes the target: the paratactic analysis existed for **direct** speech, so
`Reported=Yes` marks a complement of a speech/writing verb quoted verbatim.

Two independent direct-speech signals, and which one fires is a property of the language's
**punctuation habits**, not of the phenomenon: **quotation marks** in the complement's subtree (the
only signal in ar, 712/2297 candidates, and fa, 99/1606); and **a `discourse` dependent** inside the
complement — not a quotative marker but the direct-vs-indirect discriminator, because only verbatim
speech can host the speaker's own interjections. That is what makes la and sa tractable at all (both
have **0** quoted candidates), and the markers found are exactly right: en `no`/`well`/`yes`, la
`autem`/`quidem`/`uero` under `dico`/`inquit`, sa `vai`/`eva`/`hi` plus the quotative **`iti`** (908).
Indirect evidence commits the negative: an overt complementiser (en `that`, fa که, ar أنّ) or Latin's
accusative-and-infinitive. NB the test is on the **complement token itself**, not its subtree — SUD
makes the subordinator the head of the clause it introduces, so a complementiser anywhere else
belongs to an embedded clause, which inside a verbatim quote proves nothing.

**Latin needs almost no LLM** (`la_finite_direct`): Latin reports statements indirectly with the
accusative-and-infinitive, and every finite indirect clause carries an overt subordinator, which
under the functional-head analysis IS the complement token. So a finite complement of a speech verb
that is not itself a subordinator **has no way to be indirect** (219 cases in train — `dicit ,
meditatus sum…`, `dixit , fiat lux`). The one exception, the indirect question, is finite and
subordinator-less but requires the SUBJUNCTIVE, so mood separates it: an indicative clause containing
`qui` has a relative pronoun (75 cases), a subjunctive with no interrogative is a jussive inside a
quote (6). Only subjunctive + interrogative (39) is withheld — withheld **to the model**, not
committed as indirect.

**Reported speech is a CLAUSE.** A speech verb also takes ordinary nominal and prepositional objects
(`dicit hoc`, `loquor de X`), which must never reach the model — in Latin they were 4427 of a
4724-case residue. Candidates carry a `clausal` flag (complement is VERB/AUX/SCONJ or has a
`VerbForm`); a non-clausal residue case is dropped unannotated. Rule commits / model residue after
all three refinements: sa 1321/39, ar 997/1350, la 285/346, en 204/394, fa 110/487 — 2616 queries,
down from ~11 240. Residue goes to `disambiguate_pp.query` (resumable
`relabel_cache_reported_<lang>.jsonl`).

**The `--structural` encoder: input features matter more than architecture.** The first arms used the
standard added-layer encoder (±3 receptive field over NORM/PREFIX/SUFFIX/SHAPE) and scored F
0.12–0.40 — right for `Subject`, wrong here, where every cue is non-local (the governing verb can be
far from the clause head, quotation marks sit at the clause EDGES, and Latin's diagnostic is the
complement's own VerbForm/Mood plus the ABSENCE of a subordinator). `make_sud_config.py --structural`
swaps in explicit `MultiHashEmbed` + `MaxoutWindowEncoder` so the embed can read **`DEP`**, **`LEMMA`**
(collapses inflection, so a speech verb is one symbol across its paradigm — decisive for la/ar/sa),
**`POS`/`MORPH`** (the whole Latin finite-vs-infinitive diagnostic) and **`IS_QUOTE`**, at window 3 /
depth 4 (±12, reaching the clause edges). This **requires `annotating_components`**: the corpus
readers build the predicted doc from gold words only, so DEP/POS/MORPH/LEMMA would be absent in
training and appear from nowhere at inference (same reasoning as sa's `Compound` feature). Result:
ar test F 37.4 → 46.7 (recall 26.7 → 45.4); dev ar 0.36→0.51, fa 0.20→0.40, sa 0.40→0.56, en
0.36→0.32, la broke (0.0004).

**Which component wins is predicted by where the gold came from** — end-to-end on test
(`eval_sud_reported.py`), against the share of gold the RULES committed rather than the LLM:

    lang   plain   structural   rule    ships       gold rule-derived
    ar     37.4      46.7       73.5    rule            95 %   (997 / 1047)
    la      0.0       0.0       17.7    neither         91 %   (285 / 314)
    sa     39.6      58.0       68.8    rule            73 %   (1321 / 1814)
    en     27.6      35.0       66.7    rule            45 %   (204 / 456)
    fa     20.0      40.0       23.5    STRUCTURAL      13 %   (110 / 836)

A rule reproduces the rule-committed portion almost by definition and cannot touch the LLM-decided
remainder; a trained model can learn either. So the rule wins where the gold is mostly rule-derived
and loses in Persian, whose gold is 87 % LLM-decided (rule P 1.00, R 0.13 — it only fires on the 110
cases it committed itself). This is a property of how the class was built, not a fact about the
languages. **ar/sa/en ship the rule; fa and la ship no `Reported` layer** — Persian's structural arm
beats its own rule but at P 0.50, and Latin needs a four-deep chain of predicted
lemma/deprel/VerbForm/Mood that compounds too badly. `add_sud_reported_rule.py` removes the trained
pipe when it adds the rule, so no dead weights ship. Lexicons live in `sud_reported_data.py`,
imported by BOTH the gold builder and the runtime component so they cannot drift.
**Read these numbers with care:** there is no independent gold for `Reported` — the target is itself
these rules plus an LLM pass — so they measure *reproducibility at inference*, not correctness.

### The MISC layer is COUPLED to the arm underneath it

Every component here reads the released pipeline's own predictions (`ExtPos`, `unk`, deprel,
VerbForm/Mood, lemma), so retraining a base arm silently moves the MISC layer with it. The idiom rule
is the most exposed, being a CONJUNCTION of two predictions — upstream errors multiply rather than
add. Measured when sa switched from the freeze recipe to the joint multi-task arm (end-to-end test):
Idiom F 77.7→55.1, InIdiom 81.3→58.6, Reported 68.8→57.4 — precision holds, **recall collapses**.
**Re-run `eval_sud_idiom.py` and `eval_sud_reported.py` after any base retrain**; the gold-trees mode
of the idiom eval does not use the model and stays 100 %. What did NOT go stale:
`sud_subject_frames.py` re-harvests byte-identically after the udep-residue commit (raising
complements are `comp:obj`/`comp:obl`/`comp:pred`, never `udep`).

## Packaging and release

`scripts/package_sud.sh` is the current entry point: it picks the winning arm per language, adds
`sud_idiom` to the seven idiom-annotating arms (en/lzh/ja/fa/ar/la/sa), and keeps the per-arm
surgery — `add_clause_parser.py` for lzh/sa, `add_id_lemma_case_fix.py` for id,
`bundle_yue_pkuseg.py` for yue, `bundle_zh_charseg.py` for zh, `add_la_macronise.py --no-lut` for
la, `add_sa_frontend.py` for sa.
**The `sud_*` pipes go LAST, after `clause_parser`** on lzh/sa: `clause_parser` reassigns every head
and deprel, so a rule reading `unk` must see the tree it leaves behind, and running last also means
the Doc rebuild cannot drop the annotation.

Wheels live on the GitHub Release (v0.1.0, re-clobbered as layers landed), not in git (`dist/`
gitignored). Rebuild a custom-code wheel with
`spacy package <model> <out> --code scripts/a.py,scripts/b.py --build wheel`.

**Gotchas.**
- `spacy package` loads each `--code` file **standalone** via `spec_from_file_location`, so neither
  `from . import sud_misc` (no package) nor `import sud_misc` (scripts/ not on `sys.path`) works —
  only a file-path fallback does. Each module carries a `_sibling()` helper covering all three load
  contexts (wheel / `seg_code.py` / `spacy package`).
- **Declare runtime requirements in `meta.json`** before packaging. The ja wheel once required only
  `spacy>=3.8.14` and hit an ImportError on every load; zh now declares `jieba>=0.42.1`, yue
  `spacy-pkuseg`.
- **Training-only imports must not be module-scope in a bundled file.** `sa_presegment` importing
  `sa_tokenizer`, and `sa_presegment_lex` importing `eval_samhita`, both broke the zh wheel.
- **A component that silently loses an input must refuse to load.** `bundle_zh_charseg.py` REFUSES
  to write a model whose saved `vocab.json` lacks the `jieba_source` marker — without it the wheel
  would load, run with one input deleted, and say nothing (the same silent degradation as sa's
  `Compound` on token input).
- **Verify in a clean `--target`/venv install**, not just the loose training directory.
- For a **code-only** re-release, diff the wheel against the previous asset file by file: the sa
  code-only wheels differed in exactly 2 and 3 of 29 files, proving the weights were untouched.

### Release audit, 2026-08-04 — and the lesson

Prompted by discovering the live zh wheel was still the **pkuseg** one: the char-tagger swap had been
built (`build_zh_charseg`, itself a generation behind the best local arm) but never uploaded. So the
"zh 0.8385 → 0.8902" note described a bundle users never had, and the jump they actually got from
the re-release is token F **0.8385 → 0.9210**.

Every published v0.1.0 asset was then downloaded and read. Three checks, each catching what the
previous cannot: (1) **structure** — pipeline, bundled modules, `Requires-Dist`, tokenizer artefacts,
read from the wheel itself; (2) **weights** — `parser/model` and `tagger/model` hashed out of the
wheel and compared with the arm `package_sud.sh` selects (22/22 matched); (3) **chain integrity** —
the freeze recipe makes `parser/model` byte-identical up base → morph → lemma → sud, so a break means
an upper arm was stacked on an older lower one. Check 3 is the one that earns its keep, because 1
and 2 both pass on a wheel that faithfully ships the wrong generation.

**The finding was id**: its published wheel declared `spacy.Tokenizer.v1` — the older COARSENED arm —
while the treebank-trained segmenter with enclitics split had been finished 14 hours before the
release. `package_sud.sh` had simply fallen through to the generic
`base=training_${lang}_lemma/model-best`. Fixed. **Two false alarms**, recorded so the next audit
doesn't chase them: fa and ja show a `parser/model` mismatch against their `training_<lang>_seg`
directory — those are the PRE-udep-ruled bases left on disk after `retrain_udep_ruled.sh`; the live
chains are internally consistent. A stale sibling directory is not a stale release.

**The general lesson, now twice-learned: a directory is not a release.** Neither `build_*/` nor
`training_*/` says anything about what users have. `gh release view v0.1.0 --json assets`, the asset
size, and the wheel's own `config.cfg` do.

## Operational notes

- **`spacy train … | tail -N` hides everything until the command exits.** Two runs looked stalled for
  hours and one genuinely was; `model-last`'s mtime is the reliable progress signal (rewritten at
  EVERY eval), and `python -u` is needed for live output when redirecting.
- `config_zh` init reporting a "missing pkuseg model" is really the gitignored `userdict.txt`
  artifact (legacy pkuseg path).
- Backups of superseded representations: `backup_sa_prestrip/` (pipe-join CoNLL-U),
  `backup_sa_prepipe/` (hyphen), `backup_la_preperseus/`, `archive_residue_pass{1,2}/`.
