# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Failed experiments live in `NEGATIVE-RESULTS.md`.** Check it before retrying anything that looks
obviously right — it records ~20 measured dead ends (affix widening, decode-time lexicons, LLM
multi-way relabelling, data upsampling, tree-aware encoders) and the meta-lessons behind them.

## What this project is

Two coupled pieces of work over **Surface-Syntactic Universal Dependencies (SUD)** treebanks, now
eleven languages: en, zh, yue, lzh, ja, ko, id, fa, ar, la, sa — in **twelve** wheels, since
English ships twice (see the English section: `en_sud_ewt` CC BY-SA, `en_sud_ewt_gum` CC BY-NC-SA).

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

**`metrics_release_*.json` is the RELEASED arm; every other `metrics_*.json` is a development one.**
The distinction earns its keep, because several development files outlived the generation they
describe and the README was quoting them: the en row was a RAW run in a table declaring
gold-preproc (79.63/84.40 raw vs **81.33/86.26** gold-preproc), and ar/la/yue were still the `_ext`
arms from before the segmentation retrain the wheels actually contain (la 73.95 → **72.26**,
ar 78.45 → **77.34**, yue 65.64 → **64.51**, with `comp:obl` F moving as far as yue 26.7 → 46.2).
The release set was measured on the arm each wheel ships, identified by hashing `parser/model` out
of the DOWNLOADED wheel — a training directory of the right name is not evidence.

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
| **zh** | char tagger + jackknifed corpus lexicon + **jieba's BMES decision**, on TRADITIONAL | strict token F 0.8385 (pkuseg) → 0.9210 (simplified) → **0.9242** (`zh_seg_jbdec_trad`) |
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
bases were retrained so released models emit it.

⚠ **`en` was missed by that pass and shipped the UD relation until 2026-08-08.** Found while
building the EWT+GUM arm. The released `en_sud_ewt` had `reparandum` in its parser's label
inventory and no `conj:dicto`, and emitted it on real input — 18 predictions over the 44 sentences
whose gold carries the relation. 36/9/4 tokens in train/dev/test, 147 cells over the nine tracked
derived files. Confirmed against the **downloaded v0.2.0 asset**, not a local directory.

**Fixed WITHOUT retraining the parser, and that is the point.** `reparandum` → `conj:dicto` is a
pure label rename, so retraining on renamed data yields the same model up to RNG; renaming the
action inside the trained parser is the exact analogue and keeps every weight — and therefore every
published metric — byte-identical, so a clobbered wheel differs in the one thing that was wrong.
`scripts/rename_deprel_label.py` does it. **The hazard it guards is real**: spaCy orders actions by
`(frequency, label_string)` DESCENDING (`TransitionSystem.initialize_actions`), so the label string
is a tiebreak and a rename can renumber the actions, silently misaligning weights that are indexed
by action. en is one string away — `reparandum` and `comp:aux@pass` both have frequency 31 in
LEFT-ARC, and the order survives only because `conj:dicto` also sorts above `comp:aux@pass`. The
script refuses unless the full (action, label) sequence is unchanged position for position, and
`--verify-parses` re-parses a corpus with both models (0 heads, 0 deprels differing over all 2077
test sentences).

**Only `sud_shared` needed retraining**, because `sud_shared_data._is_conj` counts `conj:dicto` as a
conjunct and `reparandum` not, so the coordination mask moves (75 candidates across the splits, and
the repair token itself is now excluded as a conjunct — which drops `mod ADJ after` from the rule
table under `--min-count 20`). `sud_subject` reads only NORM/PREFIX/SUFFIX/SHAPE with gold from
MISC, so the released pipe was copied back in rather than re-initialised; it re-evaluates to exactly
its published 82.01. The rebuilt wheel is **29 of 38 files byte-identical** to the shipped one — the
movers are `parser/moves`, `sud_shared/model`, `vocab/strings.json` and metadata. Test: Shared
62.6 → 63.10, Subject 82.01 and Idiom 84.62/82.14 unchanged, every ship decision intact.

Other non-official UD carry-overs (`mod@poss`,
`@unmarked/@desc/@predet/@preconj`, `compound@prt`) were left as-is by user decision; `@lmod/@tmod`
and the other language-specific semantic subtypes are legitimate SUD conventions the pipeline relies
on.

## Language-specific notes

### English — TWO arms, two licences (`en_sud_ewt`, `en_sud_ewt_gum`)

`en_sud_ewt` (CC BY-SA 4.0, EWT only) is unchanged and stays the commercially usable wheel.
`en_sud_ewt_gum` (**CC BY-NC-SA 4.0**) adds the ten non-NonCommercial GUM genres — 340,324 train
tokens, +66 % on EWT. Built by `scripts/build_en_ewt_gum.sh` (steps `merge relabel fix verify filter
corpus base`), then the ordinary `train_morph → train_lemma → train_sud → package_sud` chain, which
all take `en_gum` as an arm name.

**Why two wheels rather than a filter.** GUM's LICENSE says the treebank is CC BY-NC-SA *and* that
the NC comes from the individual sources; the second reading supports filtering, the first offers
the ANNOTATIONS under NC whatever the document — and annotations are what a model absorbs. So the
merged wheel ships NC regardless of the filter, and users choose. **GUM's NC genres are FIVE**
(essay, fiction, letter, podcast, whow), not two.

**The relabel is free if the ORDER is right.** The original development corpus was EWT+GUM
concatenated, so `relabel_cache*.jsonl` already holds every GUM decision — but the keys are
POSITIONAL (`path|sentence_index|token_id`). Relabel the unfiltered EWT-first concatenation, filter
the NC genres LAST: 34,461 targets at **zero** model calls. Filtering first shifts every later index
and throws away half the cache. `build_en_ewt_gum.sh` step 2 refuses to run if the dry run bills
anything. The Perseus XPOS trap does NOT apply — GUM's 46 tags are a strict subset of EWT's 49.

**`Reported` gold keys differently and that is what makes IT cheap** — `sent_id|comp_id`, not
positional — so `base_lang()` pointing en_gum at `relabel_cache_reported_en.jsonl` makes the EWT
half free: of 565 residue decisions 394 hit, and all 171 misses were GUM. See the `Reported`
section; an arm name is not a language, and the two places that confused them both failed silently.

**Apples-to-apples on the EWT-only test** (identical gold — the EWT half of the en_gum test is
byte-identical to it, 2077/2077 blocks; RAW end-to-end, not gold-preproc, so these are ~1.7 LAS
below the released figures in `metrics_release_en*.json` and are a comparison, not a headline):
LAS **79.63 → 80.26**, UAS 84.40 → 84.82, TAG 93.09 →
93.20, `comp:obl` F **+1.52**, `udep` +4.56; against LEMMA −0.12, MORPH −0.19, SENT F −0.41. Same
shape as Perseus for Latin — the extra treebank IMPROVES the original domain. ⚠ Single seed each and
init is unseeded, so read +0.63 as suggestive. Do NOT quote the arm's own dev LAS (0.8125) against
EWT's (0.7969): different dev sets.

Released metrics (en_gum, its own test): pos_acc 0.9464, lemma_acc 0.9615; MISC layer Subject
**77.95** (trained), Shared **58.15** (trained, mask ceiling 68.82), Reported **57.58** (RULE, v
trained 35.64), Idiom/InIdiom 79.81/79.11 — every ship decision the same as en's, but re-measured on
this arm rather than inherited, as `package_sud.sh` warns. This arm never had the `reparandum` gap:
no such label in its parser's inventory.

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

#### Orthographic augmentation: sampling replaces the two copies

The union above buys exactly the two spellings it contains, at twice the data, and says nothing
about the other axes on which printed Latin varies. `train_la_aug.sh` trains the same architecture
on the **macronised copy alone**, with `la_augment.py` rewriting each document into a freshly
sampled EDITION STYLE every epoch (`la_orth.py` holds the transforms, `la_orth.Style` a style).
Macron-stripping is exact — it reproduces the plain FORM on 586 604/586 604 tokens and leaves LEMMA
untouched — so the macronised treebank is a strict superset and the plain spelling is derived, never
stored. Four axes, each resting on something measured rather than assumed:

- **Macrons**, per word. **Breves**, per word on top: a breve marks a SHORT vowel, so the candidates
  are the vowels Alatius left unmarked, minus diphthong members (no independent length) and minus
  the glides (not vowels).
- **`u`/`v`.** Which `u` is the consonant is lexical (`silua` → *silva*, `minuere` stays vocalic), and
  the treebanks disagree in a useful way: **ITTB writes `u` throughout, PROIEL and Perseus write
  `v`**, so the v-writing half is labelled data for the u-writing half. `build_la_glide_lut.py`
  harvests it (29 734 types; answers 78.3 % of ITTB's u-tokens outright) and derives a
  (prev-prev, prev, next) context rule for the residue, **97.94 % accurate held out by type**. NB
  the usual 0.90-dominance bar is WRONG here and costs 3.1 points: word-initial `u`+`e` sits at 0.89
  (glide in every real word, non-glide only in the v-writing treebanks' own spelling slips). The bar
  is for rules that COMMIT an annotation; this one only picks which of two attested spellings to
  show the model.
- **`i`/`j`.** The one axis with **no in-corpus evidence** — not one `j` in 586 604 tokens — so it is
  a rule: `i` is the consonant when it opens the word or follows a syllabic vowel and a vowel
  follows. Three guards make it hold: no `ji` exists in Latin (saves `iis`, `ii`, `iit`); the `u` of
  `qu`/`ngu` and a glide `u` are not syllabic vowels (saves `quia`, and `uiam` = *viam*); and the
  `eo` "go" family is excluded **by LEMMA**, because `iens`/`ierunt` are shaped exactly like
  `iecit`/`iacere` and only the lexeme separates them (16 tokens, all correct). Output spot-checked:
  `ejus`, `hujusmodi`, `major`, `jussit`, `Isajae`, `Pompejo`, `trajecta`, archaic `quojusdam`.
- **`ae`/`oe` → `æ`/`œ`**, only where the pair is a real diphthong — **which the macrons already
  record**: Alatius writes hiatus `āēr`, `āeris`, `poētae` but diphthongal `aere`, `caelum`, so an
  unmarked literal `ae` is the diphthong and a marked one is not.
- **Sentence-initial capitals**, per document (an edition is consistent). Worth having: ITTB
  capitalises **0 %** of sentence openings, PROIEL 16 %, Perseus 28 %.

**Results.** `make_la_variant_conllu.py` renders the test set in each style — same trees, same gold,
only FORM moves — and `eval_la_variants.py` scores both arms across all of them (gold-preproc):

| | LAS union → aug | LEMMA union → aug | |
|---|---|---|---|
| plain | 72.26 → 71.72 | 90.96 → 90.90 | the two the union arm was built for |
| macron | 72.16 → 71.32 | 88.74 → 88.45 | |
| **breve** | **18.74 → 64.91** | **20.67 → 73.22** | one unseen character inside 78 % of words |
| vj | 71.18 → 71.77 | 86.72 → **90.78** | |
| lig | 70.55 → 71.67 | 88.08 → **90.64** | |
| caps | 71.46 → 71.86 | 88.89 → 90.56 | |
| **all axes at once** | **17.93 → 64.90** | 18.92 → 71.81 | |

So ~0.5–0.8 LAS on the two spellings the union covered, in exchange for the LAS spread across
orthographies collapsing from **54.4 to 7.0**. The lemmatiser gains most and loses nothing (−0.05 on
plain, +2.6 to +4.1 on the glide/ligature axes) — edit trees are literal string edits, so it was the
component least able to generalise across spellings on its own. **TAG is the real cost**: −2.7 on
plain, since ITTB's 1 952-label composite XPOS is the most form-sensitive target here. Dev, un­augmented
union: LAS 74.19 vs 75.29, pos 95.09 vs 95.49, `lemma_acc` 93.36 vs 93.60. Morph and lemma stack by
the usual freeze recipe (parser byte-identical at each layer).

⚠ **`max_epochs` must be `-1`, and that has two silent consequences.** At `0` spaCy's
`create_train_batches` does `examples = list(corpus(nlp))` ONCE and reshuffles that same list every
epoch, so a corpus-level augmenter samples a single style per document for the whole run — the run
looks normal and trains on one fixed perturbation. `-1` streams instead, but then (1) the training
loop stops shuffling, so the reader needs `shuffle = true` (harmless: under `sud.GoldTokCorpus.v1` a
document IS an example, so it is the same shuffle by another name), and (2) `init_nlp` initialises
from `islice(train_corpus(nlp), 100)` — which truncated the tagger to **639 of 1 952** labels and
killed training on the first batch carrying a missing one. Hence `init_aug_labels.py`, which
collects labels over several augmented passes. `spacy init labels` cannot substitute: it runs
through the same `init_nlp` and reproduces the truncation.

**The lemmatiser's labels are the ones that actually need the passes.** Tagger/parser/morphologiser
labels are properties of the TREES, which augmentation never touches. Edit trees are properties of
the FORMS, so `uītae`, `vitae` and `vītæ` → `uita` are three labels, and a missing one does NOT
raise — `get_loss` maps it to `tree2label.get(tree_id, 0)` and the token is quietly taught label 0.
Growth is sub-linear (union 18 512 trees → +1 pass 20 132 → +5 22 498 → +10 26 029), so 10 passes is
affordable and lands at 29 123 kept labels; measured, **0.50 %** of tokens in a fresh augmented pass
fall outside the kept set, against the union arm's own **1.19 %** — better than the baseline, not
worse.

**ADOPTED, 2026-08-09 (user decision).** The released `la_sud_ittb_proiel_perseus-0.2.0` is now the
augmented chain: `training_la_aug` → `_aug_morph` → `_aug_lemma` → `_aug_sud`, with `package_sud.sh`
naming it and `LA_BASE` to get back the union. Measured on the released arm (gold-preproc, plain
test, `metrics_release_la*.json`): combined LAS 72.26 → **71.72**, UAS 79.17 → 78.72, TAG 80.35 →
**77.61**, `comp:obl` F 64.80 → 64.75; ITTB+PROIEL 76.58 → **75.90**, Perseus 53.47 → **53.53**. That
is the bill; the benefit is the orthography table above, where the LAS spread falls from 54.4 to 7.0.
Wheel 17.7 → 27.3 MB, almost all of it the lemmatiser's edit-tree inventory (18,512 → 29,123 labels).

**The SUD layer is trained through the SAME augmenter** (`configs/config_la_aug_sud.cfg`), not on the
union corpus: `sud_subject` reads NORM/PREFIX/SUFFIX/SHAPE, so a pipe trained on two fixed spellings
sitting on an orthography-robust parser would be the arm's own weak point. Grafting the augmenter
onto a `make_sud_config.py` config needs its two companions too — `max_epochs = -1` and
`shuffle = true` — for the reason recorded above: at `0` spaCy lists the corpus ONCE and a
corpus-level augmenter then samples one style per document for the whole run. The SUD pipes need no
`init_aug_labels`, because Yes/No/O are properties of the TREES, which augmentation never touches;
only the lemmatiser's edit-tree labels are properties of the FORMS.

⚠ **Promoting the arm REVERSED a ship decision, and re-measuring is why it was caught.** `Shared`
on the augmented base is trained **38.11** v rule 36.78 v morphologiser 10.23 — where on the union
base the table won, 35.85 v 35.10. The reversal is trustworthy in the direction it points: this is
the three-feature arm whose `model-best` is picked on the mean of Subject/Reported/Shared, the same
handicap that cost Shared ~5 points on the union base, so it runs AGAINST the winner here. The
candidate mask also reaches 48.84 % of gold against the union base's 45.0 %, i.e. the augmented
parser recovers the coordination the layer is defined over slightly better. `Subject` stays trained
(67.02 v the rule's 52.41) and `Reported` still ships nowhere (rule 17.65, trained 8.00, n=24).

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

**`malus` vs `mālus`: the `MP` level, and why the cascade is interrupted for it.** L3 answers 90 %
of tokens and is keyed on the STRING ALONE, so on a word whose vowel length depends on its part of
speech it returned the corpus majority and Morpheus — which knows the difference — was never
reached. `malus` ADJ and `mālus` NOUN, `liber` "book" and `līber` "free", were literally one
question. Two changes, together: `_RUNGS` gains two POS-bearing rungs **at the end** (so a token
with full FEATS, already answered by a more specific rung, is untouched), and the builder marks the
**4 094 POS-SPLIT forms** — those where part of speech alone settles the length and settles it
differently per part of speech. For those, and only those, the UPOS-aware answer jumps ahead of L3.
Morpheus table format 3; a format 1/2 cache lacks the list, so `MP` never fires and the cascade is
exactly what it was — `fetch_morpheus()` again to get it.

Three guards, each found by a token that went wrong:
- **Only a DECISIVE rung answers** (`rung_mask`), never `mask`'s form-wide majority fallback. That
  fallback was giving vocative `canis` the `cānīs` of `cānus` — displacing a correct answer with a
  *worse* kind of majority than the one being replaced.
- **Never inside an idiom.** SUD gives an idiom's head the IDIOM's part of speech and records it in
  `ExtPos`, so `satis` in `satis facit` is tagged VERB while the word is still the adverb "enough";
  reading that UPOS as the word's own produced `satīs`. The treebank flags it, so the guard is exact.
- Ambiguity that POS cannot settle (`os` "mouth"/"bone", `populus` "people"/"poplar") is not in the
  list and still gets the old answer, because a rung is stored only where it is decisive.

**Read the measurement carefully — agreement with Alatius is the wrong referee here.** It barely
moves (gold morphology 97.60 → 97.59 whole-token, predicted 97.39 → 97.34), and it *should* fall:
Alatius is RFTagger-predicted on exactly these hard words. Of the 24 held-out tokens where the new
answer differs from it, ~18 are ours right and its tagger wrong — `mēnse` not `mēnsē` (the
third-declension ablative is short), `ūtī` from `ūtor` not the conjunction `utī`, `capī`, `ācer`,
`audīte`, `īnsignis`; 2 are treebank POS errors we faithfully propagate (`regis` tagged VERB in
*Pharaonis regis Aegypti*); 3 are the iambically-shortened imperative `cave`. Taking Morpheus's
GOLD-POS answer as referee instead, on the 1 516 POS-split test tokens that have one:

    old (L3 corpus majority)      86.02 %          new (predicted UPOS + FEATS)   92.74 %

**+6.7 points on the words this is about**, and that is with the tagger at its weakest: UPOS
accuracy on POS-split tokens is **87.92 %** against 92.35 % overall, since these are precisely the
words that are hard to tag. Same shape as `_PARADIGM` below — measured agreement with Alatius falls
while real accuracy rises.

`_PARADIGM` is a small exceptionless override keyed on `(InflClass, Case, Number, final letter)`
(a-stem Nom/Voc sg `-a` short, Abl sg `-ā` long; o-stem Dat/Abl sg `-ō`; e-stem Abl sg `-ē`) plus
`_LEMMA_CLASS` (PROPN carries no `InflClass` in ITTB/PROIEL, so declension comes from the lemma's
ending). It exists because the table memorises pairs and cannot express a paradigm rule — nominative
`Gallia` came out `Galliā` because the treebank only attests the ablative. **The harvested data is
WRONG on these cells** (Alatius's RFTagger contradicting gold morphology), so the rule *lowers*
measured agreement (−0.03) while raising real accuracy. It faithfully transmits Case errors, so
`config={"paradigm": False}` disables it.

**The lookup key is now ORTHOGRAPHY-TOLERANT, which promoting the augmented arm made compulsory.**
The table is keyed on the treebank's own spelling, so `jussit`, `silva`, `cælum` and `mēnsĕ` all
missed and the component returned the form unchanged — on exactly the editions the parser had just
learnt to read. Two changes, both in `resolve`:

- **Both length marks come off the PRIMARY key**, not just macrons. A breve can only ever miss, and
  missing is not harmless: Morpheus's suffix levels answer almost anything, so `ŏstēnsum` was
  answered off its last four characters (mask 0) instead of falling through to the entry that knows
  it is `ostēnsum`. A wrong answer that pre-empts the right one is worse than no answer.
- **A ladder of fallback keys, least normalised first** (`key_ladder`): length-stripped, then
  ligatures expanded, then `j`→`i` / `v`→`u`. Order is load-bearing — folding the glides at the
  first step reroutes every `v` form to the `u` spelling, and the two are SEPARATE entries with
  different answers (`vitae` → `vītae`, `uitae` → nothing), because the treebanks disagree among
  themselves. Folding early cost 5.5 points on a breve-marked edition.

The mask is a bitmask over character positions, so each key carries a map back to the form: a
ligature is one character in the form and two in the key. The OUTPUT keeps the caller's orthography
and replaces only the macrons — `jussit` stays `jussit`, `cælum` stays `cælum`.

**A BREVE VETOES the inference, over every level and over the paradigm rule.** It is not noise to be
normalised away: it is the caller stating that this vowel is short, which is exactly the claim a
macron would contradict. So the breve positions are cleared from the mask last, after the lexicon,
Morpheus and `_PARADIGM` have all had their say, and they are carried through into the output —
`intĕllectam` comes back `intĕllēctam`, our answer where the caller said nothing and theirs where
they did, and `mĕnsĕ` comes back unchanged. The `-B` suffix on the level records that a breve
overruled something.

Whole-token agreement with Alatius, same gold throughout, FORM alone re-rendered:

    style   raw key   + fallback   + breve veto
    plain    93.78      93.78         93.78
    vj       95.91      95.92         95.92
    lig      92.50      92.93         92.93
    breve    70.79      93.78         94.47
    all      70.47      95.08         95.80

**Plain is unchanged to the decimal** — the raw key is still tried first, so nothing that answered
before answers differently. And breve now scores ABOVE plain, which is the point of honouring it:
a marked edition is telling the macroniser something it would otherwise have to guess, so it should
come out ahead of an unmarked one rather than merely level with it.

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

**DCS trains the morphologiser and lemmatiser, NOT the parser** — worth stating plainly because
the joint arm makes it look otherwise. `make_sa_multitask_corpus.py` builds the DCS docs with no
heads or deps at all (the only representation spaCy reads as genuinely missing), so the parser takes
no gradient from them: 244 481 sentences / 1 732 852 tokens feed tag/morph/lemma, and the parser sees
only the 21 647 sentences / 163 308 tokens of Vedic + UFAL that carry syntax. The tagger is in the
first group but is predicting the morphologiser's labels — sa's XPOS is a copy of UPOS on 100 % of
tokens in both halves. So quote the DCS size against `pos_acc`/`morph_acc`/`lemma_acc` and the
syntax size against UAS/LAS; a dataset figure for this arm is meaningless without saying which.

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
- **Both Han scripts, one script INSIDE** (`zh_sud_gsd`, `lzh_sud_kyoto`; `zh_script.py`,
  `retrain_zh_trad.sh` → `finish_zh_trad.sh` → `add_zh_script.py`). Both arms are **traditional-only**
  as of 0.2.0 and normalise at the boundary instead of training on two scripts: `ZhTradTokenizer`
  converts simplified in, the `zh_script`/`lzh_script` component converts FORM/LEMMA back out.
  Released figures: zh LAS 68.86 / UAS 73.29 / comp:obl F 28.68, lzh 77.20 / 82.92 / 66.47
  (gold-preproc, `metrics_release_*.json`). The superseded both-scripts arms trained on the two
  *real* treebanks for the same sentences — `SUD_Chinese-GSD` + `SUD_Chinese-GSDSimp`, **not** an
  OpenCC re-traditionalisation (simplification is lossy/many-to-one) — with the ext relabel living on
  GSDSimp and `transfer_relabel_gsd.py` overlaying it onto aligned GSD tokens; lzh's simplified half
  was OpenCC `t2s` of Kyoto. Dropping the augmentation costs the lzh PARSER 2.4 LAS (79.0 → 76.57
  dev), accepted so 遠 pools with itself rather than competing with 远 — 22.7 % of zh's type
  inventory is a cross-script twin (15,848 types collapse to 12,248 under `t2s`).
  `both_scripts_release.sh` regenerates the superseded arms.

  ⚠ **`_looks_simplified` cannot be "would `s2t` change it?", and that shipped.** Simplification is
  many-to-one and several merged forms are themselves good traditional characters — `s2t` maps
  台→臺, 里→裡, 面→麵, 后→後, 只→隻 — so traditional input tested positive and came back `t2s`-converted
  to simplified: 45 of 500 traditional GSD test sentences. Ask instead for evidence of the script
  with an EXCLUSIVE inventory: simplified iff `t2s(text) == text` (no traditional-only character)
  **and** `s2t(text) != text` (something to convert), which also leaves a text with no
  script-distinguishing character alone. GSD and GSDSimp are the same 4,997 sentences in the two
  scripts, so each labels the other: **3 / 4,997** traditional read as simplified, **9 / 4,997**
  simplified read as traditional, 0.120 % overall. The twelve residuals are genuine — GSD's own
  traditional text writes 酒吧里 and 何家干, and the simplified side carries the era name 乾德.

  ⚠ **jieba's channel must be asked about the `t2s` rendering on a traditional arm.** jieba's
  dictionary is simplified: its boundary decisions score F 0.8920 on traditional text and **0.9223**
  on the `t2s` conversion — the latter matching what the simplified arm was built on (P 0.9730 /
  R 0.8793), so the entire gap is vocabulary. Codes are per character and `t2s` preserves length
  (500/500 test sentences), so the answer transfers by position, with a length check falling back to
  the raw text. `--jieba-t2s` trains it; **`jieba_t2s` is written into the segmenter's `vocab.json`**
  and read back by `char_seg_tokenizer.load_segmenter` and `eval_zh_seg.py`, because a channel asked
  a different question at inference than at training is the `reads_spaces` trap again.
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
`misc_string`/`feats_string`, `has_extension`-guarded). `token.morph` is deliberately **not** used as
the slot, so a predicted SUD feature never has to compete for room with a morphological one.

**Which CoNLL-U column a key belongs to is a property of the KEY.** `Idiom`/`InIdiom`/`Reported`/
`Subject` are MISC (column 10) features in every treebank here; **`Shared` is a FEATS (column 6)
one** — 10 178 tokens in SUD_English-EWT train, all in field 6, none in field 10. We follow the data
rather than the prose throughout (SUD's guidelines list `Subject` among the FEATS features and the
data does not), so the two groups are declared separately in `sud_misc.py` (`SUD_MISC_KEYS` /
`SUD_FEATS_KEYS`) and serialised by `misc_string` / `feats_string`. At runtime both live in the one
dict.

Gold transport is via `hoist_sud_gold.py` (see the MISC/convert gotcha above), which now reads
**both** source columns: a key is looked for in MISC, then in FEATS, and one found in FEATS is
*consumed* — leaving `Shared` beside `SudShared` would make the reference carry the same gold twice.
Carrying an already-hoisted key forward is what keeps the script idempotent for the FEATS-sourced
keys; without it a second run finds no bare `Shared` to re-derive from and silently deletes the gold.
Side effect: the frozen morphologiser is then scored against gold FEATS carrying keys it never
learned, so **`morph_acc` in these arms' logs reads artificially low** — cosmetic, score weight 0.

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
    la       67.0       52.4    trained     674   (augmented base; 66.3 / 53.0 on the union one)
    yue      66.7       36.4    trained       6   (not meaningful either way)
    lzh      66.2       80.0    RULE        174
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
languages. **ar/sa/en ship the rule; fa ships the STRUCTURAL trained pipe; la ships no `Reported`
layer** — Latin needs a four-deep chain of predicted lemma/deprel/VerbForm/Mood that compounds too
badly. (fa's figures below predate the `annotating_components` fix; retrained it is F 46.15 at
P 54.55 against its rule's 23.53, which is why it now ships.) `add_sud_reported_rule.py` removes the trained
pipe when it adds the rule, so no dead weights ship. Lexicons live in `sud_reported_data.py`,
imported by BOTH the gold builder and the runtime component so they cannot drift.
**Read these numbers with care:** there is no independent gold for `Reported` — the target is itself
these rules plus an LLM pass — so they measure *reproducibility at inference*, not correctness.

### `Shared` — the one key the morphologiser was already predicting

`Shared=Yes|No` says whether a dependent of a conjunct is shared with the other conjuncts — in
`identifying and breaking up terror cells`, `up` is `Shared=No` (it belongs to the second conjunct)
and `cells` is `Shared=Yes` (the object of both). It is the **broadest** of the five keys: every
treebank here annotates it, so the language list is no longer the `Subject` list. Only ja is left out
(27 `Yes` in 168 333 tokens — the call made for sa's `Subject`).

**It differs from the other four in being a FEATS feature**, which means the released morphologisers
have been predicting it all along inside their FEATS bundles, and badly: en test P 0.68 / R 0.15,
with `Shared=Yes` correct **4 times out of 247**, and 253 of en's 572 morph labels contain the key
(la 2110 of 6170), so it roughly doubles the label inventory it is carried in. A pipe therefore has
to *beat* that rather than merely exist, and where one ships it takes the feature over —
`clear_morph` deletes `Shared` from `token.morph` so the wheel has one answer rather than two.

**The candidate mask is the whole design** (`sud_shared_data.py`, shared by the harvester, the rule
and the eval so they cannot drift). A token is a candidate iff its head is a conjunct, its own
relation is neither `cc` nor `conj`, and it lies **outside** the span between the first and last
conjunct — a dependent sitting between two conjuncts is inside its own conjunct's territory and SUD
does not mark it. On en train that reaches 92.9 % of gold `Shared` while cutting the field from
204 578 tokens to 15 499, of which 63 % carry the feature. It is a recall device, not a rule (39 % of
what it admits is unmarked), and `sud_tagger` takes it as a `mask`: outside it the gold is *missing*,
not `O`, so the model spends no capacity reproducing a constraint it is being given.

Test, end to end over gold tokens (`eval_sud_shared.py`; "mask" = the share of gold the mask reaches
on a **predicted** parse, a ceiling on rule and trained alike):

| lang | mask | morph | rule | trained | ships |
|---|---|---|---|---|---|
| fa | 80.2 | 27.1 | 58.3 | **67.7** | trained |
| en | 70.6 | 24.7 | 55.1 | **62.6** | trained |
| lzh | 65.5 | 41.3 | 52.7 | **58.8** | trained |
| ar | 60.2 | 37.8 | 52.6 | **54.6** | trained |
| id | 57.1 | 36.1 | 49.1 | **53.6** | trained |
| la | 48.8 | 10.2 | 36.8 | **38.1** | trained — on the AUGMENTED base, which is what ships; the superseded union base preferred the rule, 35.9 v 35.1 |
| ko | 37.6 | 11.3 | 28.6 | 32.5 | neither (P 40.1) |
| zh | 32.7 | **37.5** | 29.1 | 31.5 | neither — the MORPHOLOGISER wins, uniquely |
| yue | 28.4 | 6.7 | 16.0 | 21.5 | neither (P 27.7, n=74) |
| sa | 17.3 | 8.6 | 9.4 | 3.8 | neither |

**Two different tests, and conflating them is a mistake worth not repeating.** Whether to ship
*anything* is a precision question — an annotation wrong more often than right is worse than none,
which is what kept `Subject` out of the zh wheel. *Which arm*, once both clear that, is decided on
**F**, as every other choice in this layer is (lzh's `Subject` rule at 75.8 over 68.8; ar/sa/en's
`Reported` rules). An earlier draft used the precision floor as a tiebreaker and shipped la's
trained pipe over its higher-F rule; that was wrong. Where nothing ships the
morphologiser's FEATS value is left alone: for zh that is the best arm available, for ko/yue/sa it is
merely the status quo. **id and ko had no SUD layer at all before this**; id now has one.

**The mask column predicts the whole table, and it is a fact about the PARSER.** The mask is defined
over the coordination, so its quality is parse quality on exactly that structure — not on the
sentence at large. Sanskrit is the worked example: on GOLD trees the harvested table reaches dev
F 52, but on sa's own predicted trees (LAS ~0.51) the mask covers 17 % of its gold, the trained pipe
saw almost no positive example, and it learnt nothing (F 3.8). Read the mask line before either arm.

**Architecture, measured on en (dev F, `sud_tagger`'s own scorer).** The encoder is a property of the
FEATURE, not of the language, so `make_sud_config.py` takes `--encoder` and `--mask` **per feature**:
en trains `Subject` (local), `Reported` (structural) and `Shared` (tree) in one arm.

    default encoder, no mask   0.323        structural + mask   0.586
    structural, no mask        0.547        tree + mask         0.616   <- ships
    tree, no mask              0.609

**⚠ The SUD layer must be trained on the arm that SHIPS, and for lzh that is not the obvious one.**
lzh's released chain is `training_lzh_rm_morph` — punctuation-restored, rule-merged, and with NO
trained lemmatizer (`han_lemma_lut` replaces it at packaging). Its `Shared` pipe was first trained on
`training_lzh_lemma` instead, whose parse is a different model's, so its coordination mask was a
different mask; and the resulting wheel was published, silently reverting the punctuation arm,
`--keep-marks` and the lemma table. `train_sud.sh` now names the arm `training_lzh_rm_sud` after the
chain it belongs to, and `src_conllu` gives lzh the `.punct.rulemerged` files — the plain
`.relabeled_ext` ones carry no PUNCT tokens, so a corpus built from them would not even align under
`gold_preproc`. The conclusion survived the correction (trained 58.8 v rule 52.7 v morphologiser
41.3); the numbers moved. **The rule TABLE is generation-coupled too** — `build_sud_shared_frames.py`
harvests lzh from the same `.punct.rulemerged` files, since a table keyed on a tree with no
punctuation in it answers a different question.

**⚠ `model-best` in a multi-feature arm is picked on the WEIGHTED MEAN of its features' scores**, so
a pipe can be checkpointed at an epoch that suited its neighbours. Latin is the case that matters:
its `Shared` peaked at dev F 37.34 while the saved epoch holds 31.90, chosen for `Subject`'s sake —
and la is precisely where the trained arm lost to the rule. Retrained ALONE
(`SUD_FEATS=Shared SUD_SUFFIX=_shared`, which `eval_sud_shared.py` then prefers) it reaches dev 35.91
/ test 35.10, and **still** does not beat the table's 35.85, so la ships the rule on a fair
comparison rather than a handicapped one. **No other decision turns on this** — the same gap is
≤ 2.9 everywhere else (zh 2.27, yue 2.85, sa 2.79, lzh 1.88, en 1.01, ar 0.53, fa 0.06), and the
single-feature id/ko arms have none by construction. `graft_pipe.py` puts a solo-trained pipe back
into a multi-feature arm, checking first that the two share a base (it refuses when the frozen
components differ, so a pipe cannot be fed a different model's predictions).

**But a dev-F gap is not a test-F gain, and the one time it was checked it went the other way.**
After the lzh arm was rebuilt on the right chain its own gap fell to 0.09, leaving en the largest at
1.01. Trained solo, en's `Shared` reached dev 61.55 against the combined arm's 59.99 — and **test
62.23 against 62.62**. So the combined-arm checkpoint was the better model on held-out data, the
graft was not made, and no arm now carries a gap worth acting on (en 1.01, ar 0.53, lzh 0.09,
fa 0.06, id/ko 0 by construction). Retrain solo when a gap is large enough to change a DECISION, as
la's was; not to chase a point of dev F.

`sud.HeadDepsTagger.v1` wins because the evidence is not linear at any width: what matters is which
token is my head and what else hangs off it, which `[own | head | mean of dependents]` reads
directly. The rule (`sud_shared_rule.py` + `build_sud_shared_frames.py`, a backoff table over
(deprel, head UPOS, position)) is the comparison arm; its threshold defaults to a plain **majority**,
not the 0.90 dominance test `apply_udep_rules.py` uses — that script commits annotation to a
treebank, this one has to answer wherever the mask asks (en dev F 63.7 at 0.90 vs 75.7 at 0.50, and
zh/yue collapse to nothing at 0.90).

#### The pooling is a SEGMENTED REDUCTION, not a loop over tokens

`HeadDeps` originally built the third slice with a Python loop — `D[i] = X[idx].mean(axis=0)` once
per token, over a list of per-token index arrays walked off `Token.children`. That loop was never
inherent to the computation, only to writing it against the token API, and it is what made the
whole arm look like a bad fit for a GPU.

**`pool="deps"` — what the shipped pipe uses — needs no tree walk at all.** "All immediate
dependents" is exactly the INVERSE of the heads array: the edge list is `src = arange(n)`,
`seg = heads`, minus the root's self-loop. A ragged mean over that is a segmented reduction — one
gather, one `scatter_add`, one divide by the counts — so the layer costs O(1) array ops per document
instead of O(n). The backward is the same shape, because the gradient of a mean splits evenly:
dividing the PARENT row once and gathering it to each child is the identical arithmetic the loop did
as `dY[i, 2w:] / len(idx)`. Heads themselves come from `doc.to_array(HEAD)` (relative offsets stored
unsigned — view as signed and add the position) rather than a comprehension over `t.head.i`.

The other three modes vectorise too, and one detail is easy to get wrong: under `closed2` the
original filtered the GRANDCHILD's UPOS but left the intermediate link unfiltered
(`for c in t.children for g in c.children if g.pos_ in CLOSED_CLASS`). Filtering the middle link as
well would be a different feature. Multiplicity is likewise preserved rather than deduplicated — the
two-level modes can reach a token twice and the loop counted it twice, so the mean must too.

**Measured, on real parsed docs: 4.8–5.0x** for the layer (synthetic 5.5x). The counts are
accumulated with `scatter_add` on a vector of ones rather than `xp.bincount`, so no second backend
op has to exist and agree, and the denominator is clamped at 1 so a leaf divides to zeros not NaN.

**`scripts/check_head_deps.py` is the equivalence proof, and its reference is
`git show <ref>:scripts/sud_tagger.py`** — taken from git, not transcribed, so the check cannot
drift from what was actually there. Both wrappers get the SAME stub encoder, so only the pooling is
under test. Forward is BIT-IDENTICAL on all five modes; backward is bit-identical except `deps2` and
`closed2`, which differ by 4.768e-07. That was chased rather than waved through: against an exact
float64 accumulation the two implementations are **equidistant** (4.172e-07 each) and differ from
each other by exactly one float32 ULP at that magnitude, i.e. summation order, since a token there
receives several pooled contributions and the loop summed them per-parent while the edge list sums
them in edge order. The checker therefore demands exactness for the single-level modes and allows a
data-scaled 4-ULP budget for the two-level ones — a bound in ULPs of the largest gradient, not a
magic constant.

**A pure speed change, and verified as one**: no parameter shape moves, so existing weights are
untouched and every published `Shared` figure reproduces to the decimal (en_gum 58.15, en 63.10,
fa 67.71, lzh 58.78, with the mask and rule rows unchanged). A 400-step run confirms the TRAINING
path, which the eval never exercises (`SUD_SHARED_F` 13.41 -> 55.24 on a real loss). ⚠ Released
wheels BUNDLE `sud_tagger.py`, so a wheel keeps the old layer until it is re-packaged — a code-only
re-release is what hands users the faster inference.

**It does not overturn the GPU verdict.** It removes the specific blocker (O(n) kernel launches, and
one host->device copy per token to ship each index array across), but these remain small CNNs at
width 64–96 where transfer dominates, so the dependable payoff is faster CPU training — which is
where the pipe actually ships. Treat "makes GPU viable" as a hypothesis needing a probe.

### ⚠ `annotating_components` was missing `tok2vec` — every structural arm was trained on noise

Found while building `Shared`, and it **fails silently**. The tagger/parser/morphologizer/lemmatizer
in these arms are listeners on the shared encoder, so running them without `tok2vec` feeds them a
stale buffer. Nothing raises. On a 298-token dev doc the predicted parse came out with three distinct
deprels (`ROOT`, `comp:obj`, `goeswith`) and **no `conj` at all**, against twelve and four once
`tok2vec` runs — so a pipe reading DEP/POS/MORPH was reading noise, and the `Shared` mask was EMPTY on
every training doc, its loss a flat 0.00.

Fixed in `make_sud_config.py`; all arms retrained. What it moved, end-to-end on test:

    Reported  fa 40.0 -> 46.15 (structural, the one shipped)   ar 46.7 -> 45.98   sa 58.0 -> 52.17
    Subject   lzh 59.0 -> 68.83   en 80.0 -> 82.01   fa 89.5 -> 90.67   la 66.3 -> 62.60

**Every ship decision survives** (lzh still prefers its `Subject` rule at 80.0 against 66.2; ar/sa/en
still prefer their `Reported` rules).
⚠ **lzh's `Subject` rule cannot be scored on the bare `training_lzh_rm_morph`** — it keys on the head
LEMMA and that arm has no lemma layer, since lzh's is attached at packaging (`han_lemma_lut`). Doing
so returns a flat 0.00, which reads as a finding and is an artefact. `eval_sud_subject.lzh_rule_arm`
builds the same lemma layer the wheel ships, on demand, and evaluates against that. The `Subject` moves are seed noise, not the fix — that pipe uses the default
encoder and reads nothing structural, and model init here is unseeded.

**fa now SHIPS its `Reported` layer**, reversing an earlier decision (user decision, 2026-08-05).
That decision rested on P 0.50 — "half of what it emits is wrong" — measured before this fix.
Retrained it is **F 46.15 at P 54.55**, against its own rule's 23.53. fa remains the one language
where the trained pipe beats the rule for `Reported`, because its gold is 87 % LLM-decided and a
rule can only reach the 13 % it committed itself.

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
`bundle_yue_pkuseg.py` for yue, `bundle_zh_charseg.py` / `bundle_id_charseg.py` for zh/id,
`add_la_macronise.py --no-lut` for la, `add_sa_frontend.py` for sa.

**An arm trains more pipes than its wheel ships**, so trimming is part of packaging: en/fa/la/yue/ar/
lzh/id now all take `training_<lang>_sud` as their base (ar/lzh/id joined when `Shared` did), and the
pipes that lost their comparison are removed so no dead weights travel. `add_sud_reported_rule.py`
and `add_sud_idiom.py --drop` both remove pipes but both also ADD one, which is wrong wherever the
language does not want the thing being added — `drop_pipes.py` is the plain version, and yue is the
case in point (ships trained `Subject`, annotates no idioms, must not ship `Shared`).
**The `sud_*` pipes go LAST, after `clause_parser`** on lzh/sa: `clause_parser` reassigns every head
and deprel, so a rule reading `unk` — or `sud_shared`'s coordination mask — must see the tree it
leaves behind, and running last also means the Doc rebuild cannot drop the annotation.
⚠ This held **by accident of ordering** until lzh started taking a trained arm as its base:
`add_clause_parser.py` simply appended, which put `clause_parser` *after* `sud_shared`. It builds,
loads, and says nothing. It now positions itself `before=` the first `sud_*` pipe, so the invariant
is enforced rather than assumed — check `pipeline` in the built wheel's `config.cfg`, not the script.

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
  `spacy-pkuseg`, and **ar `camel-tools>=1.5.2`** — its tokeniser raises at LOAD time, so before this
  a plain `pip install ar_sud_padt` produced a model that could not be opened. Per-language
  requirements now live in `stamp_model_meta.py`, which already runs for every arm at packaging.
  The `camel_data -i …` download still has to be run by hand: a data fetch is not expressible as a
  pip dependency, so this reduces the missing pieces from two to one rather than to none.
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

**And the sharpest version of it, 2026-08-05: check that your BRANCH is not behind main before you
build anything.** A `Shared` branch six commits behind main rebuilt and uploaded all eleven wheels.
Main had meanwhile (a) replaced lzh's trained lemmatizer with `han_lemma_lut` and repointed its
packaging base to the punctuation-restored `training_lzh_rm_morph` with `--keep-marks`, and (b) added
`stamp_model_meta.py` so every wheel carries its licence. The upload therefore shipped lzh a
generation backwards and eleven wheels with an EMPTY `License:` field — and the local diagnosis went
the wrong way round, reading the correct 9.1 MB lzh asset as "stale" because a stale directory of the
same name sat beside the current one. `git log --oneline <branch>..main` would have said so in one
line. Corrected by merging main in, retraining lzh's pipe on the right arm, and rebuilding.

**The code-only re-release, 2026-08-09 — and the check that is worth more than the release.** The
segmented `HeadDeps` is a pure speed change, so the six wheels that BUILD that layer (ar, en,
en_gum, fa, id, lzh — `sud_tagger.py` also travels in la and yue, but their `Shared`/`Subject` pipes
use the plain tagger encoder and never instantiate it) were re-packaged at their own live versions
and re-uploaded. Each was then diffed **file by file against the DOWNLOADED asset**, and that is
what earned its keep: five came out differing in `.py` files and metadata alone, and **lzh moved
tok2vec, tagger, parser, morphologizer and `sud_shared`**. `package_sud.sh`'s lzh default still
named `training_lzh_rm_sud`, the both-scripts arm, after lzh went traditional-only end to end — so
the routine command rebuilt the superseded generation. The wheel built, loaded and parsed correctly;
only the hash comparison said otherwise. Repointed to `training_lzh_trad_sud`. **A default that
names the right arm is the fix; a comment telling the next person to remember is not** — this was
the third time lzh nearly shipped backwards.

### The zh wheel that could not segment, 2026-08-09

`zh_sud_gsd-0.2.0` went to the release with **no `tokenizer/segmenter/` directory**, so every input
string came back as ONE TOKEN. It built, loaded, parsed and round-tripped; `spacy evaluate
--gold-preproc` was unaffected, because gold tokens never run the tokeniser. What exposed it was
listing the wheel's own files against the 0.1.0 one, which has
`tokenizer/segmenter/{model.bin,vocab.json,lexicon.txt}` — the same class of check that caught lzh
above, and again the only thing that would have.

**Two silent fallbacks in series.** `add_zh_script.py` carried the segmenter over from the input
model's tokenizer by trying attribute names — `("segmenter", "lexicon", "_seg", "_lex")` — and
`CharSegTokenizer` holds it in **`seg`**, which is not among them. `to_disk` then writes a
`segmenter/` directory only when it has one, and `from_disk` falls back to no segmenter when the
directory is absent. Neither step raises. Copying state between objects by GUESSING attribute names
is what failed: the script now takes `--seg`/`--lexicon`, calls `load_segmenter`, and refuses to
write a model whose reload cannot split a test sentence into more than one token. And the wheel was
hand-built rather than run through `package_sud.sh`, whose zh branch still named
`sud_gsd_simp_trad` and still fell through to `training_zh_lemma` — both now fixed, the same
"a default that names the right arm is the fix" lesson.

**Rebuilt and re-uploaded at the SAME version** (0.2.0, clobbered, by user decision). All five
component weight files are byte-identical to the previous asset and to `training_zh_trad_lemma`, so
no published score moves; the diff is the three segmenter files, four `.py` modules and metadata.
Raw end-to-end on the traditional test: token_acc **0.9694**, strict token F **0.9242**. ⚠ Because
the version is unchanged, `pip install -U` will NOT replace a broken copy — `--force-reinstall`
will. Verified by downloading the published asset and loading it, not the build directory.

Two diffs that look alarming and are not, both on lzh: `__init__.py` differs only in IMPORT ORDER,
and `sud_subject_frames.py` is purely ADDITIVE (an `en_gum` key; lzh's own 7 entries, the ones its
Subject rule reads, are byte-identical). Check the table, don't trust the filename.

**Corollary, found 2026-08-05: `build_sud/` can hold two wheels with the SAME name.** A stale
`build_sud/lzh_rel_pkg/` sat beside `build_sud/lzh/`, each with its own `lzh_sud_kyoto-0.1.0-py3-none-any.whl`
(9.1 MB vs 14.5 MB, one a `han_lemma_lut` generation behind). The documented upload line is
`gh release upload v0.1.0 $(find build_sud -name '*.whl') --clobber` — which would have uploaded both,
and `--clobber` makes the winner whichever `find` yields last. Removed. **Count the wheels before
uploading**: one per language, or the release is a coin toss.

## Operational notes

- **`spacy train … | tail -N` hides everything until the command exits.** Two runs looked stalled for
  hours and one genuinely was; `model-last`'s mtime is the reliable progress signal (rewritten at
  EVERY eval), and `python -u` is needed for live output when redirecting.
- `config_zh` init reporting a "missing pkuseg model" is really the gitignored `userdict.txt`
  artifact (legacy pkuseg path).
- Backups of superseded representations: `backup_sa_prestrip/` (pipe-join CoNLL-U),
  `backup_sa_prepipe/` (hyphen), `backup_la_preperseus/`, `archive_residue_pass{1,2}/`.
