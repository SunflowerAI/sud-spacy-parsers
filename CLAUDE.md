# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Two coupled pieces of work over **Surface-Syntactic Universal Dependencies (SUD)** treebanks
(English, Chinese, Korean, Indonesian):

1. **Small CPU spaCy dependency parsers** (tagger + parser sharing one efficiency `tok2vec`)
   trained from SUD CoNLL-U.
2. A **`udep` disambiguation pipeline**: SUD labels prepositional/adpositional dependents of
   verbs with the noncommittal `udep`; we relabel each as `comp:obl` (complement) or `mod`
   (modifier) using **qwen3:8b via Ollama** (no thinking, temperature 0), then retrain and
   compare. This is the core research contribution — see `README.md` and the per-language
   results in `metrics_*.json`.

There is no package/test suite; "running it" means executing the spaCy CLI and the
`scripts/*.py` pipeline. Always use the project venv: `.venv/bin/python`.

## Environment (critical, non-obvious)

- **Python 3.12 only.** The machine default `python3` is 3.14, which has no spaCy wheels.
  `pip install spacy` does **not** pull in `click` (spaCy imports it directly) — it is pinned
  in `requirements.txt`. See `.venv/`.
- **Korean tokenizer** needs mecab-ko: the Korean spaCy pipeline (and any `config_ko.cfg`
  `init`/`train`) requires `export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib`. mecab-ko was
  installed via Homebrew (conflicts with and unlinked the Japanese `mecab`); `mecabrc` dicdir
  points at `mecab-ko-dic`.
- **Chinese tokenizer** now uses a pkuseg model trained on GSDSimp (`segmenter = "pkuseg"`
  in `config_zh.cfg`, `[initialize.tokenizer] pkuseg_model = "models/zh_gsdsimp_pkuseg"` + GSD
  user dict), which reproduces GSD word boundaries far better than jieba (word-F1 ~0.88 vs 0.80).
  Install with `pip install spacy-pkuseg`. See "Tokeniser–treebank matching" below.
- **Ollama** must be running with `qwen3:8b` pulled. A single request already saturates the
  Metal GPU — parallel requests / `OLLAMA_NUM_PARALLEL>1` give **no** speedup (~3 calls/s is
  the ceiling). Don't bother parallelizing.

## Common commands

```bash
PY=.venv/bin/python
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib   # needed whenever Korean is involved

# Convert CoNLL-U -> .spacy
$PY -m spacy convert <file>.conllu <out_dir>/ --converter conllu -n 10

# Train (English uses configs/config.cfg; per-lang use configs/config_<lang>.cfg)
$PY -m spacy train configs/config_zh.cfg --output training_zh/ \
  --paths.train corpus_zh/<prefix>-train.spacy --paths.dev corpus_zh/<prefix>-dev.spacy

# Evaluate — for zh/ko/id you MUST pass --gold-preproc (see below)
$PY -m spacy evaluate training_zh/model-best corpus_zh/<prefix>-test.spacy --gold-preproc

# Whole-pipeline drivers
bash scripts/train_baselines.sh       # convert+train+eval baselines for zh/ko/id
bash scripts/relabel_retrain.sh       # relabel udep -> retrain -> eval, all three langs

# Tokeniser-matching drivers (see "Tokeniser–treebank matching")
bash scripts/train_all_retok.sh       # prep (retokenise ko / coarsen id) + train zh/id/ko on matched tokenisation
bash scripts/eval_retok.sh            # eval each matched model: gold-preproc vs raw end-to-end (raw tok = tokeniser match)
bash scripts/relabel_retrain_retok.sh # relabel + transfer through transforms + retrain + eval, matched tokenisation
```

`spacy train` writes scores to `train_*.log`; `spacy evaluate --output` writes `metrics_*.json`.

### gold_preproc (non-obvious, essential for zh/ko/id)

`spacy evaluate` re-tokenizes raw text with the model's tokenizer; jieba/mecab segmentation
does **not** match the treebank's gold tokens, which collapses alignment (Korean LAS dropped
to ~30 before this fix). The configs set `gold_preproc = true` for the train/dev corpora and
**evaluation must use `--gold-preproc`**, so everything runs on gold tokens. English (spacing
matches) doesn't need it.

When editing configs programmatically, load with `Config().from_disk(p, interpolate=False)` —
the default interpolation resolves `${paths.train}` to null and silently breaks CLI path
overrides (this caused `E913` errors).

### Sentence segmentation (learned; `gold_tok_corpus.py`, `config_*_seg.cfg`, `retrain_seg.sh`)

`gold_preproc = true` had a hidden cost: it feeds the parser one **pre-segmented** sentence at a
time, so the parser never learned to *start* a sentence — on raw multi-sentence input the released
non-English models collapsed everything into one tree (raw `SENT F` 0). `en` was fine (trained
`gold_preproc=false`); `lzh`/`sa` segment via `clause_parser` (their treebanks carry no in-text
boundaries, so they can't learn them). The released models now **learn** boundaries via a custom
reader: **`scripts/gold_tok_corpus.py`** registers `sud.GoldTokCorpus.v1`, which yields whole
multi-sentence docs (corpora are `convert -n 10`) but with **gold tokenisation** for the predicted
doc — so segmentation is learned with **zero tokeniser skew** (matters for zh/yue pkuseg). Toolchain:
`make_seg_config.py` derives `config_<lang>_seg.cfg` (swaps both corpus readers + sets `sents_f`
weight 0.05); `seg_code.py` is the single `--code` loader (spacy **train** `--code` takes one file,
unlike **package**); `retrain_seg.sh <langs>` retrains each released arm into `training_<lang>_seg/`;
`regen_idko_corpora.sh` rebuilds the cleaned-up id/ko corpora from the surviving `*.relabeled.conllu`.
Each released arm was retrained with its own recipe (la = ext+macron union `corpus_la_ext_union/`,
zh = `config_zh_both` + baked pkuseg, yue = `zh_both_tok2vec.bin` init + `bundle_yue_pkuseg.py` swap)
and repackaged with `package_seg.sh` (sa/lzh repackaged from `model-seg`, new `clause_parser`, no
retrain). **Result (raw end-to-end LAS / `SENT F`, old→new):** ar 69.4→72.4 / 0→66, fa 79.2→85.3 /
0→99, ja 81.9→85.8 / 0→96, id 68.3→73.4 / 0→87, ko 68.6→74.3 / 0→90, la 63.9→70.9 / 0→74,
zh 54.3→57.4 / 0→99, yue 52.0→60.0 / 2→81 — raw LAS up everywhere (correct boundaries help parsing),
gold-preproc LAS within ~1 (the research metrics above are gold-preproc and still describe the
relabel contribution). Re-released over v0.1.0 (clobber); `en` unchanged.

### UPOS morphologisers (`make_morph_config.py`, `train_morph.sh`, `package_morph.sh`)

The released pipelines were `[tok2vec, tagger, parser]` — the `tagger` predicts **XPOS** (`tag_`),
so `token.pos_` (UPOS) was **empty** in output (no `morphologizer`; the parser's embed reads only
`["NORM","PREFIX","SUFFIX","SHAPE"]`, so POS never fed parsing anyway). To emit UPOS+morph for
downstream tasks **without changing parsing**, every released arm gained a `morphologizer` trained
with a **freeze recipe**: source the released `tok2vec`/`tagger`/`parser`, **freeze** them, and train
ONLY a new `morphologizer` that carries its **OWN small `HashEmbedCNN`** (width 64 / depth 3 / embed
2000). A *dedicated* encoder (not a listener) is the key choice — it is immune to treebanks whose
**XPOS is orthogonal to UPOS** (id: 33/46 XPOS values map to >1 UPOS). Verified empirically on id:
standalone-frozen 92.8 vs listener-on-frozen-encoder 92.2 (the orthogonality penalty) vs **co-train**
92.95 *but* LAS −0.3 / TAG −0.5 — so **co-training is dominated** (no UPOS gain, hurts parsing) and
was discarded. The frozen components are **byte-identical** to the release (verified per-arm with
`cmp` on `*/model`), so parse/seg metrics need no re-verification. `make_morph_config.py` derives
`config_<lang>_morph.cfg` from the released arm's (seg) config: sources+freezes the three, nulls
`init_tok2vec` (else yue's Mandarin-init `zh_both_tok2vec.bin` clobbers the sourced encoder and breaks
the parser's input), and keeps only factory args common to the standard **and** `ja.morphologizer`
factories (the latter rejects `label_smoothing`/`overwrite`/`extend`). `train_morph.sh` trains all 11
arms (en uses the plain `config.cfg`; lzh/sa source from `model-seg`); `package_morph.sh` packages
each (lzh/sa re-append `clause_parser` **after** the morphologiser — `clause_parser` reads `pos_` from
the whole-doc pass and preserves it through its re-parse; yue re-runs `bundle_yue_pkuseg.py`).
**UPOS (`pos_acc`), small encoder:** en 0.934, ar 0.946, fa 0.960, ja 0.967, id 0.928, ko 0.939,
la 0.955, zh 0.896, yue 0.911, lzh 0.912, sa 0.877 (UPOS ≥ the model's own XPOS acc where XPOS is
coarse-mappable; en UPOS 0.934 > XPOS 0.929). Wheels add ~+2 MB; re-released over v0.1.0 (clobber).

### Lemmatisers (`make_lemma_config.py`, `train_lemma.sh`, `package_lemma.sh`)

Every released arm also carries a `lemmatizer` (`token.lemma_`), added with the **same freeze recipe**
one layer up from the morphologiser: source + **freeze** the arm's `tok2vec`/`tagger`/`parser`/
`morphologizer` from `training_<lang>_morph/model-best`, and train ONLY a new **`trainable_lemmatizer`**
(spaCy's edit-tree lemmatiser, `backoff="orth"`) that carries its **OWN small `HashEmbedCNN`** (64/3/2000).
The edit-tree lemmatiser learns FORM→LEMMA string edits from the treebank `LEMMA` column, so it is
language- and script-agnostic (works for lzh/sa). `make_lemma_config.py` derives `config_<lang>_lemma.cfg`
from the arm's `_morph` config (inheriting its reader + train/dev data), sources+freezes the four,
appends the lemmatiser, and sets `lemma_acc` as the only score weight. The frozen components are
**byte-identical** to the morph arm (verified per-arm with `cmp`), so parse/seg/morph/UPOS metrics are
unchanged — lemma is a purely added layer. `train_lemma.sh` trains all 11 arms; `package_lemma.sh`
packages each with the **new names** (`sud_gsd_simp_trad`/`sud_ittb_proiel_perseus`/`sud_vedic_ufal_csl`),
re-appending `clause_parser` for lzh/sa and swapping pkuseg for yue. **`clause_parser` now also carries
`lemma`/`morph`** through its per-clause re-parse (it rebuilt the doc with only tag/pos/head/dep before,
which would have dropped lemmas on raw lzh/sa input). **`lemma_acc`:** en 0.949, ar 0.907, fa 0.981,
ja 0.979, id 0.957, ko 0.977, la 0.936, sa 0.848, zh 0.999, yue 0.999 (Chinese/Cantonese lemma ≈ form —
near-identity), lzh 0.999. (id/ko lemmas were recovered from the raw treebanks — see the coarsen/retok
notes above; their corpora had been 100 % `_`, which would have made those two lemmatisers vacuous.) Wheels add
~+1 MB; re-released over v0.1.0 (clobber). NB the lemmatiser is a purely additive layer — the frozen
tok2vec/tagger/parser/morphologizer stay byte-identical, so parsing/UPOS metrics are unchanged.

### Indonesian FEATS bug + lemma sentence-initial-casing fix (`coarsen_id.py`, `id_lemma_case_fix.py`)

`coarsen_id.py` hardcoded the merged token's FEATS column to `_` (same bug the lemma column had
before it was fixed to use `rt.lemma`), so `corpus_id_coarse_rl` — what the id `morphologizer` and
`lemmatizer` train on — had **0 %** non-empty FEATS despite the source treebank carrying real
morphology (`Number`, `Voice`, `PronType`, `NumType`, …) on ~42 % of tokens; the trained
morphologizer accordingly predicted empty morph on every live token, while `spacy train`'s own
dev-set `MORPH_ACC` misleadingly read 100.00 (trivially correct against an all-empty gold). Fixed
by using `rt.feats` (the `Tok` class already carried it, `retokenize.py:29`, just never read);
rebuilt `assets_id_coarse_rl`/`corpus_id_coarse_rl` and retrained `training_id_morph`/
`training_id_lemma` — now a real **`morph_acc` 0.909** (`Number=Plur` on reduplicated nouns,
`Voice=Act`/`Voice=Pass` on active/passive verbs, `PronType=Dem`/`NumType=Card` etc. all predicted
correctly on raw text).

Separately (unrelated root cause): the `trainable_lemmatizer` mis-lemmatised sentence-initial
capitalised **hyphenated** forms (`Anggota-anggota`→`Anggota-anggota` instead of `anggota`) even
though the exact same word lemmatises correctly lowercase, and even when the capitalised instance
*was* in the training data. Diagnosed via `EditTrees.add`: edit trees are literal-content
substitutions (the leaf stores the exact characters to delete/insert), so a capitalised token and
its lowercase counterpart get two **different** trees; sentence-initial capitalisation makes the
capitalised tree a near-singleton (one training instance per distinct word) that the classifier
can't reliably learn to select — confirmed the correct tree already existed in the trained model
(`trees.apply` produces the right answer), the classifier just doesn't predict it at inference.
Plain (non-hyphenated) capitalised words are unaffected (`Buku`→`buku`, `Jakarta`→`jakarta` already
generalise fine — they share trees with hundreds of other simple downcasings). Fixed with a small
deterministic safety-net component, **`id_lemma_case_fix`** (`scripts/id_lemma_case_fix.py`,
appended after the lemmatizer like `clause_parser`): overrides the lemma from a
`FORM.lower()`→`LEMMA` table harvested from the training treebank by `build_id_lemma_lut.py`
(scoped to hyphenated forms only, 398 entries, embedded as a dict literal in the module — a couple
of GSD gold rows with a stray XPOS/Morf fragment leaked into LEMMA were filtered out), but **only**
when the lemmatizer's prediction equals the raw surface form (the observed failure signature) and
the token is simple initial-cap (`text[1:] == text[1:].lower()`) — so it never touches an
already-correct prediction. `package_lemma.sh id` now runs `add_id_lemma_case_fix.py` before
packaging (mirrors `add_clause_parser.py` for lzh/sa); verified in the built wheel
(`id_sud_gsd-0.1.0`) via a clean venv install, not just the loose `training_id_lemma` dir.
Separately noticed but **not fixed** (out of scope, pre-existing): the raw-text tokeniser
inconsistently splits some capitalised hyphenated reduplications (`Argumen-argumen` → 3 tokens)
that lowercase forms tokenise as one — a tokeniser-boundary issue, not a lemma/morph one.

### SUD-relation conformance (`normalise_reparandum.py`)

Two SUD-relation audits against the guidelines: (1) `conj` is correctly **chained** (each conjunct →
the previous, `cc` → the conjunct it precedes) in every treebank, and no transform disturbs it —
nothing to fix. (2) `appos` is never emitted bare; apposition is the sanctioned `conj:appos` (46260).
(3) The UD relation **`reparandum`** (disfluency/repair) survived un-converted in a few upstream SUD
releases; SUD renders it as **`conj:dicto`** (sibling of `conj:coord`/`conj:appos`). `normalise_reparandum.py`
rewrites it (696 total across all derived files; distinct instances la 32 / yue 165 / zh 2 — ITTB /
Cantonese-HK / Chinese-GSD+GSDSimp) → `conj:dicto`, **DEPREL column only** (`reparandum` is also a Latin gerundive word form — FORM/LEMMA
untouched). It is a **pure label rename** (head/attachment unchanged). The affected corpora were
reconverted and **la/yue/zh bases retrained** (then morph + lemma) so the released models emit
`conj:dicto`. Other non-official UD carry-overs (`mod@poss` family, `@unmarked/@desc/@predet/@preconj`,
`compound@prt`) were left as-is by user decision (they'd need broad base retrains; `@lmod/@tmod` and the
other language-specific semantic subtypes are legitimate SUD conventions the pipeline relies on).

## Tokeniser–treebank matching (`retokenize.py`, `coarsen_id.py`, `train_pkuseg_zh.py`)

`gold_preproc` sidesteps the tokeniser/treebank mismatch for *evaluation*; this layer makes the
tokeniser and treebank actually **agree**, so the parsers work on raw text (raw-eval `tok` is now
0.941 zh / 0.999 id / 1.000 ko, up from a Korean collapse to ~LAS 30). Direction is chosen per
language by whether the treebank tokenisation is a deterministic function of the text:

- **zh — bend the tokeniser to the treebank.** No spaces ⇒ word segmentation is statistical,
  never lossless. `train_pkuseg_zh.py` trains a pkuseg model on GSDSimp (`spacy_pkuseg.train`;
  `train_iter` >20 doesn't help, fine-tuning from `spacy_ontonotes` barely helps; a GSD word-type
  **user dictionary** adds ~+0.04). Best word-F1 ~0.88 vs jieba 0.80. NB the ~0.94 pkuseg reports
  is the lenient *cut-point* F, not word-level F. Wired into `config_zh.cfg`.
- **ko — retokenise the treebank (finer).** `retokenize.py --lang ko` splits each eojeol into its
  mecab morphemes (matching `KoreanTokenizer` exactly) and builds eojeol-internal structure
  **functional-head** per the mSUD standard: the case particle (ADP) / verbal ending (AUX) heads,
  the lexical stem is `comp:obj`/`comp:aux`. This lands `udep`/`comp:obl`/`mod` on the case
  particle — parallel to the adpositions in the other languages. Lossless + reversible (asserts
  per-sentence round-trip). Evidence for functional-head: the native `mSUD_Nenets-Tundra` treebank
  (case suffix = ADP heads its noun via `comp:obj`; verbal suffix = AUX is the clause root).
  **Lemmas** are carried onto the morphemes: the eojeol lemma is `+`-separated and aligns 1:1 with
  the mecab morphemes (`잡스는`→`잡스+는`), so each morpheme takes its part (else falls back to the
  surface form). Without this the retok corpus had 100 % `_` lemmas and the lemmatiser was vacuous.
- **id — coarsen the treebank.** Enclitics (`-nya/-lah/…`) are lexically ambiguous (`-lah` is a
  clitic 73× but inside whole words like `adalah`/`salah` 1723×) and not rule-separable, so
  `coarsen_id.py` merges each MWT range (host+enclitic) into one whitespace token, which the rule
  tokeniser reproduces deterministically. token-F1 vs spaCy 0.955→0.989. The merged token keeps the
  **host (representative) token's lemma** (`penghuninya`→`penghuni`; the clitic is not part of the
  lemma) — likewise needed so the lemmatiser isn't vacuous (`coarsen_id` had hardcoded lemma `_`).
- **en — leave as is.** `Tokenizer.v1` already matches EWT at the rule ceiling (F1 0.991);
  hyphen/slash tweaks both regress (EWT is internally inconsistent). A useful negative result.

`retokenize.py` also has a general char-span align + reproject path (merge/split/crossing +
cycle/root repair) used for zh boundary disagreements; for ko every block is a clean 1→m split.
**Reversibility invariant**: new tokens are surface substrings + carry `SpaceAfter=No`, so
concatenation reproduces the text. Only **en** and **ko-eojeol** are deterministically matchable;
**zh** and **id** stay statistical, so keep `--gold-preproc` for fair parser comparison. The
relabel decisions are tokenisation-agnostic, so `relabel_retrain_retok.sh` relabels at the
original tokenisation (cached) and **transfers** the labels through these transforms (which
preserve deprels) rather than re-relabelling the retokenised data.

## Pipeline architecture (`scripts/`)

Data flows: download SUD `.tgz` → extract to `assets*/` → merge/convert → `corpus*/*.spacy` →
train → relabel `udep` → retrain. Naming convention: English artifacts are unsuffixed
(`corpus/`, `training/`, `metrics.json`); other languages use `_<lang>` suffixes; relabeled
variants use `_rl`/`_relabeled`; improved (contrastive-prompt) reruns use `_rl2`.
Matched-tokenisation variants add `_simp` (zh pkuseg/GSDSimp), `_coarse` (id enclitic-merged),
`_retok` (ko morphemes); these compose with `_rl` (e.g. `training_ko_retok_rl/`,
`metrics_zh_simp_rl_gp.json`). For matched models, `metrics_*_{gp,raw}.json` hold the
gold-preproc and raw end-to-end evaluations.

- **`disambiguate_pp.py`** — foundation module imported everywhere. `parse_conllu`,
  `descendants` (dependency subtree), `render` (subtree → surface text, trims edge punct),
  and `query` (the canonical qwen3:8b call: `think:false`, `temperature:0`, normalized
  one-word answer). Other scripts load it via `importlib.util.spec_from_file_location`.
- **`build_gold.py`** (English) / **`lang_gold.py`** (zh/ko/id) — build the *confident* comp/mod
  benchmark from `udep` cases that are unambiguous: COMPLEMENT = verb lexically selects the
  adposition (curated `(verb, adp)` frame lists); MODIFIER = temporal/causal adposition or
  temporal-object. Writes `gold_*.jsonl`. (SUD's own committed labels are too sparse/noisy,
  which is why the gold is rule-built.) Note the temporal-object override: a frame with a
  year/temporal object → modifier (e.g. "believe in 1999").
- **`eval_prompts.py`** (English) / **`lang_bench.py`** (zh/ko/id) — benchmark prompt variants
  against the gold. Prompts are **static prefix (definitions + few-shot) + short variable
  suffix (the sentence)** so Ollama reuses the cached prefix KV (~4× speedup); keep them this
  way. `eval_prompts.PREFIXES["fewshot12_def"]` is the canonical English prompt.
- **`zh_bench.py` / `id_bench.py` / `en_bench.py`** — curated same-adposition contrastive
  few-shot (e.g. 在/于 place-vs-time, di/pada selected-vs-temporal). `en_errors.py` does
  error analysis to drive these.
- **`relabel.py`** (English) / **`lang_relabel.py`** (zh/ko/id) — apply the chosen prompt to
  the full in-scope `udep` set, rewriting `udep`→`comp:obl`/`mod` in CoNLL-U. Resumable via
  on-disk `relabel_cache*.jsonl` (every model decision flushed). `lang_relabel` uses the
  confident **rule first**, model only for the genuinely ambiguous remainder; `CHOSEN` /
  `EXTRA_SHOTS` hold the winning per-language prompt. Block-based rewriter preserves the file
  byte-for-byte except target deprel cells — verify round-trip before long runs.
- **`relabel_ext.py`** (all four langs) — **extended scope** beyond ADP-of-VERB, writing separate
  `*.relabeled_ext.conllu` (baselines untouched), cache `relabel_cache_ext_<lang>.jsonl` (seeded
  from the baseline caches so verb decisions aren't re-queried). Adds: ADP dependents of
  **NOUN/PROPN/ADJ** heads; clausal verb PPs (the no-VERB-in-subtree filter is dropped for VERB
  heads only); **participial** complex prepositions (`according/based/following`→mod); a Korean
  **case-suffix rule** (`ko_case_label`: particle off the rightmost head-final eojeol → mod for
  locative/temporal/comitative, comp for dative 에게 and selecting frames, model for 로/으로/topic/
  bare); zh 的/之 associative PART→mod; ko ADV-of-VERB→mod. **Partitives (NUM/DET/PRON heads) stay
  `udep`** (SUD's documented default — `nmod`→`udep`; user decision). Reuses the baseline
  per-language prompt — `eval_prompts.suffix()` names the head word generically, so it reads
  naturally for noun/adjective heads. Retrain with **`relabel_retrain_ext.sh`** → `corpus_*_ext` /
  `training_*_ext` / `metrics_*_ext.json` (printed base vs verb-rl vs ext, LAS + comp:obl F).
- **`udep_audit.py` / `udep_probe.py` / `hard_examples.py`** — analysis behind the extended scope:
  profile every `udep` by head/dep POS (in-scope vs out); committed comp/mod base rates per head
  POS; the Korean case-particle calibration that produced `KO_MOD_CASES`/`KO_COMP_CASES`
  (`udep_probe.py --ko-case`); and a sampler for the residue left as `udep`.

## Key empirical findings (so you don't re-derive them)

- Relabeling `udep`→binary comp/mod lowers headline **LAS by ~1–2** in every language (the
  binary is harder than the noncommittal label) while **UAS is unchanged** (only labels change).
  The metric that reflects disambiguation quality is per-label **`comp:obl` F**.
- Few-shot composition only slides the precision/recall frontier; it can't beat a good prompt.
  English plateaus at ~0.91–0.93 on qwen3:8b; gains there came from **auditing the gold**, not
  more examples.
- Value scales with how genuinely ambiguous the adpositional system is: high for English /
  Indonesian (prepositional), and — *at the verb-ADP scope* — near-vacuous for Korean
  (postpositional, `udep` adpositions ~96% temporal/causal modifiers). **The extended scope
  overturns the Korean conclusion** (see below): Korean's `comp:obl` signal lives on bare
  case-marked NOUN dependents of verbs, not on the few ADP tokens.
- The relabel signal survives the tokeniser matching: on the matched tokenisation, relabeling
  moves **`comp:obl` F** by id +17 / zh +10 / ko −12 (gold-preproc) — the same ambiguity-scaling
  story, with the Korean drop being the near-vacuous case (relabeling only adds a few hard
  `comp:obl` instances that dilute the class). Headline LAS still falls ~1–2 everywhere.
- Matching the tokeniser to the treebank is usually better than re-tokenising the treebank, and
  only needed at all when you want a *different granularity* than the treebank has (the ko
  morpheme case). A trained segmenter is **not** lossless — only deterministic tokenisations
  (en rules, ko-eojeol whitespace) can be matched exactly.
- **Extended scope (`relabel_ext.py`, plain tokenisation + gold-preproc):** disambiguating
  `udep` beyond verb-headed adpositions (noun/propn/adj heads, clausal verb PPs, participials,
  Korean case-marked NOUN dependents) lifts per-label **`comp:obl` F** further than the verb-only
  relabel — base → verb-rl → **ext**: id 0.463→0.565→**0.703**, ko 0.169→0.247→**0.386**,
  zh 0.190→0.307→**0.356** — with headline **LAS flat** (±0.01). English regresses slightly
  (comp:obl F 0.740→0.730): it already had a large, well-disambiguated verb `comp:obl` set, so
  noun/adjective heads dilute the class. **Korean is *not* near-vacuous** once the case suffix on
  noun dependents is used (the verb-ADP-only view missed where its signal lives). Caveat: each
  relabel rewrites the *test* gold too, so `comp:obl` F has a moving denominator (same caveat as
  base-vs-rl). Partitives are left `udep` by design.
- **Six more languages (fa/ar/la/sa/lzh/ja; `*_new.sh` drivers, configs `config_<lang>.cfg`).**
  Treebanks: Persian-PerDT, Arabic-PADT, Latin-ITTB+PROIEL+Perseus (merged; see the Latin section
  below for the Perseus addition), Sanskrit-Vedic,
  Classical_Chinese-Kyoto, Japanese-GSD. Per-language relabel model (Phase-3 benchmark, English
  prompt unless noted): fa/sa/lzh→qwen3:8b, ar/la→gemma4, **ja→qwen3 + native-Japanese prompt**
  (`OLLAMA_MODEL` env selects it; `disambiguate_pp.MODEL` reads it). `comp:obl` F base→verb-rl→ext:
  **fa 0.705→0.815→0.794, ja 0.000→0.720→0.688, ar 0.617→0.659→0.634, la 0.678→0.691→0.684,
  lzh 0.716→0.659→0.664, sa 0.404→—→0.352** (LAS within ~1 throughout). The thesis holds across
  language types: relabelling **helps genuinely-ambiguous prepositional systems** (fa/ja/ar/la) and
  **hurts the near-vacuous/model-limited ones** (lzh: `udep` coverbs ~mostly modifiers, model 0.70;
  sa: case-based, model ~chance on the Ins/Acc residue), and ext dilutes `comp:obl` F where the
  verb set is already strong (fa/en/ja/ar/la — の/noun-heads). **Japanese GSD commits *no*
  `comp:obl`** (all particle deps left `udep`), so the relabel synthesises the class from scratch
  (F 0→0.72) — the cleanest demonstration of the LLM adding new annotation.
  - **Two `udep` families.** Prepositional (fa/ar/la/lzh/ja — the adposition/particle is the ADP
    head of the NP) use the verb-frame gold; case-based (sa) uses the dependent's morphological
    **Case** (parallel to Korean). Associative genitive → mod, like zh 的: **lzh 之** and **ja の**
    (relabel_ext buckets `lzh_zhi`/`ja_no`, deterministic).
  - **Classical Chinese coverb rule** (`LZH_LOC_COMP_VCLASS`/`lzh_coverb_label` in lang_gold;
    `lzh_coverb` bucket in relabel_ext). The bulk of lzh's coverb signal does **not** live on plain
    `udep` (which both relabel pipelines scope to) — it lives on the **subtyped** `udep@lmod`
    (locative, ~3029) and `udep@tmod` (temporal, ~105) ADP<-VERB deps, which the plain-`udep` scope
    never reached. relabel_ext now brings them in and decides them from the annotators' own semantic
    category + the head verb's class (XPOS field 3): **@tmod → mod** (WHEN adjunct); **@lmod →
    comp:obl** only under a locus-selecting verb class (移動 motion / 姿勢 posture / 設置 placement /
    存在 existence / 生物 birth-death), else **mod** (circumstantial locative). Object FEATS `Case=Tem`/
    `Case=Loc` is the same signal for any *plain*-`udep` coverb (via `classify`). This commits ~815
    of the test coverbs (`udep` 1288→473) and **nearly doubles the comp:obl class** (test 182→355,
    incl. the locative-complement construction the LLM relabel had entirely missed): comp:obl F
    base→verb-rl→**ext 0.716→0.685→0.701** (with the frame rule below), **precision 0.72**, **LAS
    flat 0.789→0.790**, mod F unchanged. So lzh is near-vacuous only on the *plain* `udep` residue;
    the locative complements
    are a real, learnable comp:obl class (same lesson as Korean's case-marked NOUN deps).
  - **lzh plain-`udep` 於 routing (object semantic class).** After Loc/Tem are ruled, the residue
    splits ~evenly person 958 / non-person 912. The treebank commits **0 comp:obl and 0 mod on
    於+person** (maximally ambiguous: recipient-dative vs comparison vs passive-agent) — only the LLM
    can adjudicate it. **於+non-person** *is* committed (84:54, ~61% comp), so a verb-frame rule fits.
    The default `_derive_comp_frames` (minc=8/thresh=.85) yields *no* lzh frames (too sparse), so
    `COMP_FRAMES["lzh"]` is derived loosely (**minc=2/thresh=.70 → ~15 frames**: 至於/達於/在於/異於/
    甚於/長於/怒於…), committing ~132 non-person comps by rule. So the LLM is scoped to where it is the
    only tool — **於+person and the non-frame non-person residue** — while frames + Loc/Tem + the
    @lmod/@tmod subtypes carry everything decidable by rule. No re-querying needed: the rule
    intercepts cases *before* the cache; person 於 has no committed gold so its LLM decisions are
    inherently unvalidatable (no benchmark possible).
  - **Sanskrit case rule** (`SA_MOD_CASES` in lang_gold; `sa_case_label` in relabel_ext): recipients
    are **dative** (confirmed in-treebank: dā/prayam+Dat), not locative — the locative-of-locus is
    the Vedic ritual `hu` "offer into fire-LOC", which SUD leaves `udep`/mod. So Loc/Abl/Voc/Nom→mod,
    recipient datives→comp via the (verb,Case) frames; **blanket Dat→comp is avoided** (the
    dative-of-purpose is adjunctival); Ins/Acc/Gen/Dat-of-purpose → model.
  - **Tokenisers.** fa/ar/la/sa = rule tokeniser + gold_preproc. **lzh has no spaCy module**:
    `scripts/lzh_tokenizer.py` registers a custom `lzh` language + char tokeniser (one Han char =
    one token, deterministic), loaded via `spacy ... --code`; the shipped wheel bundles it. ja =
    SudachiPy + gold_preproc. fa/ja run on raw text (TOK 99.1/99.4); **sa & lzh need pre-segmented
    sentences** (Vedic/Kyoto carry no in-text sentence boundaries — raw LAS collapses to ~41/~48).
    `scripts/clause_parser.py` (lzh + sa, last pipe) recovers per-sentence parses on punctuated
    editions. A **sentence** is the span between two sentence-final marks; within it
    the content tokens are concatenated **with sentence-medial marks removed** and parsed as one doc,
    then every mark is reattached as `punct`. Which marks are sentence-final is set by **`sent_scheme`**
    (`add_clause_parser.py --sent-scheme`). For lzh `sent_scheme=""` + empty `sent_punct`, so *every*
    mark is sentence-final and each 句讀 unit is parsed in isolation (unchanged). **Sanskrit sets
    `sent_scheme="danda"`** — a **document-dependent** rule: `?`/`!` (+ an optional trailing **closing**
    straight/curly/angular quotation mark, a space before it allowed) *always* end a sentence; then, of
    the remaining marks, **only** the period ends a sentence if the text has periods (non-decimal),
    **else** only the double daṇḍa if the text has any double daṇḍa (single daṇḍas become medial),
    **else** the single daṇḍa. An OPENING quote after a final mark begins the next sentence (not pulled
    back). A daṇḍa may be `| || / //`, `‖`, or an Indic-script daṇḍa `।॥` (the sa tokenizer normalises
    `।`→`|` and every **double** daṇḍa `॥`/`||`/`।।`/`//`→`‖`; `_danda_kind` counts strokes, ≥2 ⇒
    double). A *medial* mark (a comma, a single daṇḍa
    when doubles are present) is pulled out but its comma-separated units are parsed together (the parser
    itself relates them — no fabricated `parataxis`) and it reattaches as a `punct` child of the **head
    of its left unit**; a sentence-final mark (and any trailing quote) attaches to the sentence root on
    its left. It also **normalises punctuation morphology** — Kyoto/Vedic carry almost no
    punctuation, so the tagger hallucinates content tags on it (？→名詞,糧食 "noun, food", 。→動詞,
    brackets even become ROOTs). Every punctuation token (Unicode P*, incl. quotation brackets) is
    forced to `pos=PUNCT` + a deterministic XPOS: the Kyoto `s,記号,{句点,読点,括弧開,括弧閉}` map for lzh,
    or the component's `punct_tag` config string flat (sa sets `punct_tag = "PUNCT"`, so the daṇḍa is
    not stamped with Japanese-tagset notation). gold-preproc eval bypasses clause_parser, so metrics
    are unaffected — this is purely a raw-inference fix. Repackage the lzh/sa wheels to ship it.
  - **Released (v0.1.0), all 6 + the original 4:** `fa_sud_perdt` (ext), `ja_sud_gsd` (ext),
    `ar_sud_padt` (ext), `la_sud_ittb_proiel_perseus` (ext), `sa_sud_vedic_ufal_csl` (base, **CSL-reverted** —
    accepts sandhied CSL, de-sandhies to clean wordforms; see below; re-clobbered at 0.1.0),
    `lzh_sud_kyoto` (**ext** —
    bundles `training_lzh_ext` + `clause_parser` with the punctuation-morphology fix; replaced the
    base wheel in-place at 0.1.0 via `gh release upload --clobber`). Wheels live on the GitHub
    Release, not committed (`dist/` gitignored). Rebuild a custom-code wheel with
    `spacy package <model> <out> --code scripts/lzh_tokenizer.py,scripts/clause_parser.py --build wheel`
    (add `clause_parser` to the model first; remember `pip install click` — spaCy imports it directly).
  - **Both Han scripts (`zh_sud_gsd_simp_trad`, `lzh_sud_kyoto`; `scripts/both_scripts_release.sh`).** Both
    models train on a traditional+simplified union. **zh was renamed `sud_gsdsimp`→`sud_gsd_simp_trad`** (old
    asset deleted): it trains on the two *real* treebanks for the same sentences — `SUD_Chinese-GSD`
    (original traditional) + `SUD_Chinese-GSDSimp` (simplified auto-conversion) — NOT an OpenCC
    re-traditionalisation (simplification is lossy/many-to-one). The ext relabel lives on GSDSimp;
    `transfer_relabel_gsd.py` overlays it onto aligned GSD tokens (udep-only + alignment guard;
    comp:obl/mod is script-independent). pkuseg is retrained on the union (`models/zh_gsdboth_pkuseg`),
    swapped into the bundle post-hoc (gold_preproc training is segmenter-agnostic). **lzh** has no
    simplified counterpart treebank, so its simplified half IS OpenCC `t2s` of Kyoto
    (`scripts/opencc_conllu.py`; char-level, length-preserving). gold-preproc: zh combined LAS 69.3 /
    comp:obl F 32.6 (simp 35.3, GSD-trad 29.9); lzh combined LAS 79.0 / comp:obl F 70.9 — both within
    ~0.2 LAS across scripts, LAS/disambiguation unchanged vs the single-script ext models.
  - **Cantonese (`yue_sud_hk`; `scripts/{split_yue,yue_tokenizer,train_yue,train_pkuseg_yue,bundle_yue_pkuseg}.py`).**
    Coverb/prepositional like zh/lzh: in-scope `udep` ADPs are coverbs (喺 at, 畀 dative, 到 goal, 由
    from, 根據 according-to), decided by the same verb-frame/temporal `lang_gold` rules + qwen3:8b
    (`CHOSEN["yue"]=SUD_DEF`; gold 30c/12m, model==gemma4). ext adds two deterministic buckets in
    relabel_ext: associative/genitive **嘅** (plain `udep` PART → mod, like zh 的/lzh 之/ja の) and the
    annotators' temporal subtype **`udep@tmod`** (而家/今日/嗰陣時 → mod, like lzh @tmod), plus bare
    NOUN-of-VERB (temporal-lemma→mod rule, else model) and ADP-of-NOUN/ADJ. comp:obl F base→verb-rl→
    **ext 0.308→0.261→0.348**, LAS ~flat (noisy: see below). **SUD_Cantonese-HK is test-only** (1004
    sents) → deterministic 80/10/10 round-robin split (`split_yue.py`, also copies empty XPOS←UPOS so
    the tagger predicts UPOS in `tag_`; `pos_` empty). No spaCy `yue` module → `yue_tokenizer.py`
    registers a custom `yue` language. **tok2vec is Mandarin-init by default**: `config_yue.cfg` bakes
    `init_tok2vec = zh_both_tok2vec.bin` (extracted from `training_zh_both/model-best` via
    `model.to_bytes()`; needs the `[pretraining]` component/layer block — spaCy's cross-lang `source=`
    is blocked by E150 vocab-lang). vs from-scratch: TAG +0.4–1.4, UAS +0.7/+1.2 (base/ext), baseline
    LAS +1.15; **comp:obl F within 100-sent noise**. **Raw tokeniser = pkuseg** (`yue.PkusegTokenizer.v1`,
    falls back to char tok): word-F1 0.95 vs char 0.63; **fine-tuning pkuseg from `zh_gsdboth` ties
    from-scratch (0.9474 vs 0.9472)** so the self-contained from-scratch model ships (userdict *hurt*,
    0.93). Released v0.1.0 = ext arm + pkuseg, packaged from `training_yue_ext_pkuseg` (swap via
    `bundle_yue_pkuseg.py`; meta requires `spacy-pkuseg`; no clause_parser — sentence-segmented).

## SUD-vs-UD feature survey (what else might be disambiguable like `udep`)

A survey of SUD phenomena beyond `udep` — what's genuinely SUD-specific, what's plain UD, and
which residues are actually worth a few-shot-LLM disambiguation pass like the comp/mod work above.

**Overall architecture.** SUD favours **functional heads** (adposition/complementizer/auxiliary as
head) over UD's content-word heads, and collapses UD's category-driven relation inventory
(`nsubj/csubj/obj/iobj/obl/xcomp/ccomp/amod/nmod/nummod/advmod/acl/advcl/aux/cop/case/mark`) into 3
macro-relations refined by function, not POS: `subj`, `mod`, and `comp` (`comp:obj/obl/pred/aux/
cleft`), plus the noncommittal `udep` (hypernym of mod/comp) and `unk` (root of the whole taxonomy,
completely unclear relations). `conj` uses a **string analysis** (each conjunct depends on the
*previous* one, not the first) and splits into `conj:coord` / `conj:appos` (paradigmatic-list
apposition; tighter apposition is `mod:appos`) / **`conj:dicto`** (repetition/repair — this project
already converts leftover upstream `reparandum` to `conj:dicto`, see `normalise_reparandum.py`;
SUD's `conj:dicto` makes the *first* element head, unlike UD's `reparandum`, which makes the repair
the head). Raising/control is handled by decoupling the surface relation from a separate
**`@`-suffixed deep-feature layer**: `@x` (shared/raised subject, implied by `comp:pred`/`comp:aux`),
`@y` (object-raising), `@pass` (passive), `@caus` (causative), `@agent` (demoted agent), `@expl`
(expletive `subj`, replacing UD's dedicated `expl` relation), `@scrap` (spoken disfluency/
incomplete constructions — not typos). Reported speech's current recommendation is a `Reported=Yes`
MISC feature + ordinary `comp:obj` (an older `parataxis:obj`/`parataxis@rep` is superseded but still
visible in some treebanks). Numerals can be `det` (round-trips to UD `nummod:det`); `orphan`,
`punct`, `goeswith` are carried over from UD unchanged.

**Why the three user-suggested candidates aren't good LLM-disambiguation targets here:**
- **Typo correction** (`Typo=Yes`/`CorrectForm=`, plus `CorrectSpaceAfter/Number/Tense/VerbForm=`)
  is a **plain UD convention, not SUD** — SUD's conversion leaves it untouched. Confirmed in this
  repo's own data (en EWT/GUM, small amounts in id/fa/la; zero in zh/ko/ar/ja/la-ITTB/PROIEL/lzh/
  sa/yue) — and wherever it appears it's already fully gold-annotated, so there's nothing ambiguous
  to resolve.
- **Subject raising** (`Subject=SubjRaising`/`ObjRaising` in MISC) is likewise already
  gold-annotated wherever the source treebank carries it (2000–7500+ instances in en/zh/id/fa/la;
  **zero** in ar/ko/ja) — not ambiguous. Also moot right now regardless: no `config_*.cfg` has a
  `morphologizer`, so MISC/FEATS aren't trained or evaluated at all currently (`spacy convert`
  preserves them into the `Doc`, but nothing scores them) — exploiting this would mean new
  plumbing, not LLM annotation.
- **Reported speech** (`Reported=Yes`, `@reported`/`@rep`) is real but vanishingly sparse in this
  project's actual treebanks — only 33 Latin `@reported`/`@rep` instances total (ITTB patristic
  scriptural citations, e.g. `dicit` governing a `comp:obj@reported` quoted clause). GUM has a
  richer `Discourse=attribution-positive/negative` layer (1650 instances) but GUM isn't part of the
  training pipeline (`en` trains on EWT only).

**What turned out to be good targets instead** (same shape as `udep`: SUD-specific, genuinely
ambiguous, and either already exploited productively for one language or clearly analogous):
- **Sanskrit's `udep@<subtype>` semantic-role tags** — see the Sanskrit section below (Track A,
  implemented: `comp:obl` F 0.352→0.396 over the case-only ext arm, still short of un-relabelled
  base).
- **`unk` audited and ruled out (negative finding).** `unk` is a second noncommittal relation,
  distinct from `udep`, largest in Japanese (7519 train tokens) and Arabic (4111). Systematic
  audit of the whole train split: **99.2%** of Japanese `unk` tokens are the bound continuation of
  an `Idiom=Yes`/`InIdiom=Yes` periphrastic copula/auxiliary chain (である "de-aru", てくれる
  "te-kureru", てくる "te-kuru"...) — the token is always adjacent to its head (99.4%, `id ==
  head_id+1`), and the real syntactic relation (root/mod/udep/comp:aux/...) is carried entirely by
  the FIRST idiom token; some chains are 3+ tokens (1600 cases where the head is *itself* `unk`).
  Arabic splits: **53%** the same idiom-chain pattern (e.g. `باسم` "in the name of", a complex
  preposition — the second morpheme is `InIdiom=Yes`), **47%** newswire dateline/formatting
  artifacts (date/wire-agency-abbreviation tokens attached to a placename root with no real
  relation, e.g. `برلين 15-7 (اف ب)`). In both languages `unk` correctly marks "this token carries
  no independent grammatical relation of its own" (structurally like UD's `fixed` for non-first
  MWE members, but broader) — there is no comp:obl/mod-style choice being deferred, so **it is not
  a good target for the udep-style few-shot pipeline**: nothing would be gained by relabelling it,
  since these tokens don't semantically have an oblique/modifier relation to assign in the first
  place. This mirrors the lzh plain-`udep`-residue precedent: a negative result reached by an
  audit-first approach, avoiding a wasted gold/bench/relabel build.

## Latin (`la_sud_ittb_proiel_perseus`): three treebanks, macrons, XPOS blanking

The released Latin model trains on a plain `cat` of three SUD Latin treebanks (each keeps its own
sent_ids): **ITTB + PROIEL + Perseus**. `scripts/add_perseus_la.sh` is the reproducible driver
(phases `merge|macron|relabel|train`); it composes the macron and ext-relabel pipelines below.

- **Perseus splits.** Perseus ships only train + test (no dev), so it is added **train→train,
  test→test**; dev stays ITTB+PROIEL. Source: `grew.fr/download/SUD_2.18/SUD_Latin-Perseus.tgz`
  (ITTB in `assets_la/SUD_Latin-ITTB`, PROIEL in `assets_la2/SUD_Latin-PROIEL`).
- **XPOS blanking (non-obvious).** The three treebanks use mutually-incompatible XPOS tagsets (field
  5): ITTB Index-Thomisticus codes, PROIEL 2-letter, Perseus 9-position morphology. ITTB+PROIEL
  already mixed two and coped (TAG ~92), but Perseus's sparse fine tagset on ~1.3k sents tags at
  ~34% and tanks the combined TAG/LAS. Fix: **blank Perseus XPOS** (`scripts/blank_perseus_xpos.py`,
  field 5→`_` on the Perseus tail of each split; folded into `add_perseus_la.sh do_merge`). UPOS,
  FORM and dependencies are kept, so Perseus still trains the parser — only the tagger stays coherent.
  Blanking is orthogonal to the macron (FORM) and relabel (DEPREL) transforms.
- **Results (gold-preproc, ext+macron union = release).** Apples-to-apples on the ITTB+PROIEL test:
  LAS 77.7→**78.3**, UAS 83.1→83.8, `comp:obl` F ~69 (Perseus *improves* the original domain).
  Perseus-only test LAS ~54.6 (classical poetry — genuinely hard). Combined-test headline LAS 73.9 /
  comp:obl F 65.2 is lower only because the test now includes Perseus. ext relabel uses
  `OLLAMA_MODEL=gemma4:latest`; macrons via the Docker macroniser (see `macronise_la.py`).
- **Macrons.** One union parser handles plain + macronised input (`scripts/train_la_ext_macron.sh`
  trains on plain-ext ∪ macron-ext; `macronise_la.py` restores macrons via the Alatius Docker
  macroniser, `transfer_macrons.py` composes the FORM transform onto the ext deprels).
- **Licence: CC BY-NC-SA (NonCommercial).** All three sources are NC (ITTB BY-NC-SA 3.0, PROIEL
  BY-NC-SA, Perseus BY-NC-SA 2.5) — the only NonCommercial released model. See `NOTICE.md`.

## Sanskrit sandhied-CSL representation (`sa_sud_vedic_ufal_csl`)

The released Sanskrit model **replaces `sa_sud_vedic`**. It **accepts sandhied text in
Clay-Sanskrit-Library (CSL) conventions** (the Vedic treebank is natively *pausa*/unsandhied; UFAL
is classical Pañcatantra prose) but parses **de-sandhied wordforms normalised toward the pre-pausal
(pausa) form**: the tokeniser undoes the notation-marked sandhi (vowel coalescence + avagraha) AND
the deterministic, place-preserving external sandhi (visarga, `-s`/`-r`→`-ḥ`, voiced-stop
neutralisation, anusvara), and the parser is trained on those normalised forms — cleaner than the
sandhied surface, so it parses better (test-gp LAS 54.03 vs 53.5 sandhied-surface). The pipeline is
(1) build the sandhied CSL representation, then (2) revert the sandhi (`desandhi_csl`) for both
training and inference. Representation, built once and shared by both treebanks:

- **`scripts/external_sandhi.py`** — forward classical external-sandhi engine (`join_pair`): vowel
  coalescence (savarṇa/guṇa incl. `a+ṛ→` word2 `r`, vṛddhi, yaṇ, ayādi **glide-preserving** — `e/o+V→
  ay/av`, `ai/au+V→āy/āv`, NOT bare hiatus, so the junction stays reversible), visarga (`-aḥ→
  -o/-aś/-as`, `-iḥ→-ir`, `-āḥ→-ā`, final `-s`=visarga), `m→ṃ`, `-t→-d/-c/-j`, `-n→-ṃs/ñ/nn`,
  `t+ś→cch`, stop voicing; `internal=True` suppresses external-only rules (the `-n→-nn`
  gemination) for bound junctions. NB **no gold sandhied text exists**, so this is rule-based
  *generation* validated by the round-trip + textbook unit tests, not against a reference.
- **`scripts/apply_vedic_sandhi.py`** — applies it to the Vedic treebank within each sentence
  (`assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{train,dev,test}.sandhi.conllu`): compounds AND
  verb-**preverbs** (upasarga whose `head==` the following VERB; excludes tmesis) hyphen-joined
  with internal sandhi; the privative **`a-/an-`** sandhi'd + hyphen-joined (no gemination);
  pragṛhya duals, elided `_`, and sentence edges left in pausa. `generate()` chains sandhi
  **sequentially left-to-right** (carrying each word's evolving surface into the next junction) —
  needed for **single-character words**, notably the emphatic particle **`u`**: computing junctions
  independently mishandled it (`atha u āhuḥ` → `ath' u āhuḥ`, the `u` left uncoalesced); sequencing
  yields the correct `ath' ô āhuḥ` (= atho), which reverts cleanly to `atha u`. (Earlier versions
  bailed on these via an overlap guard; the fix touched 92 dev/train/test tokens, 90 of them `u`,
  with train/test reverted forms unchanged so the released model is unaffected.)
- **`scripts/sa_csl_prep.py`** (UFAL) — alignment-based CSL: transliterate Devanagari→IAST,
  re-segment MWTs (compounds hyphen-joined, external sandhi space-resegmented into surface forms,
  vowel coalescence marked); hard cases hand-corrected via `sa_ufal_csl_overrides.tsv`; typographic
  double quotes → **guillemets `«»`** (CSL direct-speech mark) before normalise.
- **Compound members carry NO join marker; MWTs recover from the `Compound` FEAT + `n-m` range.**
  Compound/preverb/privative members are emitted as **clean wordforms** (`śuka`/`sāri`/`kṛśānāṃ`), not
  the pipe-joined `śuka|sāri|kṛśānāṃ` used earlier. `scripts/strip_pipe_sa.py` removed the trailing
  compound-join `|` from **plain-token FORMs only** (len>1, guarded against the daṇḍa `|`/`||`), leaving
  the `n-m` MWT **surface** line (`śuka|sāri|kṛśānāṃ`) and `# text` intact so every grouping stays
  recoverable; the `Compound=Yes` FEAT marks *samāsa* members (the privative `a-/an-`/preverbs are
  PART/ADV and carry **no** `Compound` feat, so those recover from the range line / `SpaceAfter=No`
  adjacency only). The morphologiser predicts `Compound=Yes`, so a live-output samāsa MWT = a
  `Compound=Yes` run + the following head token. Stripped **8326/1044/1035** markers (train/dev/test);
  that strip-only base ran **LAS 54.11 (dev) / 53.83 (test gp) / TAG 88.5 / UAS 66.9** — ~−0.5 LAS vs
  the pipe arm (the parser embed reads NORM/PREFIX/SUFFIX/SHAPE, not FEATS, so removing the visible
  member cue costs a little and the feat gives nothing back). **Superseded** by the pre-pausal
  normalisation base below (test gp LAS 54.03 > 53.83). `backup_sa_prestrip/`
  holds the pre-strip (pipe) CoNLL-U; `backup_sa_prepipe/` the earlier hyphen version;
  `scripts/hyphen_to_pipe_sa.py` converts the compound join hyphen→pipe (still used by
  `rebuild_sa_csl_rev.sh`).
- **`scripts/sa_tokenizer.py`** — reproduces the tokenisation (hyphen-split keeping the join on the
  left member; daṇḍa/`||` run-grouping; `circumflix` U+0302 NOT stripped as it's a coalescence mark);
  **accepts BOTH a compound `|`** (CSL, word-internal `|`→`-` internally; a sentence daṇḍa `|` is
  never letter-followed) **and a legacy compound `-`**, and **straightens curly apostrophes/double-
  quotes → `' "`**; **reverses the CSL-marked sandhi** via `desandhi_csl` (see below); then **drops
  the compound-join marker** (a trailing internal `-` on a len>1 token is removed) so members are clean
  wordforms matching the training data, with membership carried by `SpaceAfter=No` adjacency. A lone
  dash stays `-`.
- **Source offsets (`doc._.src_spans` / `token._.src_span` / `doc._.src_text`).** This tokeniser is the
  only one in the project that *rewrites* what it reads, so `doc.text` is NOT the input and a token
  form generally is not a substring of it (`śaśa-bhṛto` → `śaśa` + `bhṛtaḥ`) — `token.idx` indexes the
  reconstructed string and is useless to a caller wanting to decorate the input. Every token therefore
  also carries the **half-open character span of the raw input it came from**, as spaCy extensions
  registered at import of the `--code` module (guarded with `has_extension`, since loading two models
  in one process imports it twice). It is **purely additive: the token list is byte-identical**, which
  is what let the wheel be repackaged with no retrain (verified over 27 420 corpus `# text` lines,
  0 differences; and the rebuilt wheel differs from the previous asset in exactly `sa_tokenizer.py`,
  `clause_parser.py` and the dist-info `RECORD`). Mechanism: `normalise` is not length-preserving
  (Devanagari→IAST is many-to-many, an accent mark vanishes), but **every token boundary this tokeniser
  can produce falls on a character that `normalise` maps 1:1** — `split()` cuts at whitespace, `_SPLIT`
  at `_PUNCT`, `_HYPH` at `-` — so `_normalise_aligned` segments the input at exactly those three
  classes ("anchors"), normalises each segment on its own, and aligns segment-to-segment; a maximal run
  of non-anchors is at most one token (a Devanagari word has no interior cut point), so per-segment IS
  per-token. Anchors are tested AFTER `_STRAIGHTEN` (which maps the curly quotes — themselves in
  `_PUNCT` — onto `'`/`"`, which are not) and the Indic daṇḍas are anchors, so each is transliterated
  alone (`।`→`|`, `॥`→`||`). The piecewise result is **checked** against `normalise(text)` and all spans
  are dropped (None) if they disagree, so an exotic input can cost the offsets but can never change the
  tokens. Downstream the spans survive because `desandhi_csl` and the join-marker strip both preserve
  the token COUNT (the forms change under the spans, which is the point), and because `clause_parser`
  copies them onto the doc it rebuilds. A token gets `None` rather than a guess whenever its normalised
  range does not land on segment boundaries at both ends; in practice that is never (208 567/208 567
  corpus tokens spanned), and the spans **tile the input exactly** — the only characters in no span are
  the spaces between tokens, and a compound member owns its join hyphen (`śaśa` ← `śaśa-`). Empty input
  now yields an empty `Doc`; it used to raise `E031` on the `words = [norm or text]` fallback.
- **`desandhi_csl` (`scripts/sa_tokenizer.py`) + `scripts/revert_csl_sandhi.py`** — normalises each
  word toward its **pre-pausal (pausa) form**, so the parser sees ONE canonical wordform regardless
  of the following context (less surface sparsity). `desandhi_csl(tokens)` walks the ordered token
  list and undoes, in six stages: **(0)** on the raw surface, the dropped visarga — `-a`/`-ā` in
  HIATUS before a vowel → `-aḥ`/`-āḥ` (a genuine `-a`/`-ā` + V would have coalesced *and been marked*,
  so an unmarked one is a lost visarga), `-o` before avagraha / a voiced consonant → `-aḥ`
  (`namo 'stu`→`namaḥ astu`, `vatso vi-`→`vatsaḥ`), the **ayādi glide** before a vowel → the mid
  vowel/diphthong (`-ay/-av/-āy/-āv`→`-e/-o/-ai/-au`: `tay i-`→`te`, `tāv a-`→`tau`), and **yaṇ**
  before a vowel → the **short** vowel (`-Cy/-Cv`→`-Ci/-Cu`: `ity a-`→`iti`, `tanv a-`→`tanu`,
  `dadātv a-`→`dadātu`; the i/ī, u/ū length is lost so short is the default — `iti`, not `itī`), plus a
  **bare glide** token `v`/`y` (the emphatic `u`/`i` reduced before a vowel) → `u`/`i`. NB the following
  vowel that triggered the glide/visarga may be **hidden**: when the next word is a single-vowel particle
  (preverb `ā`, emphatic `u`) that itself coalesces FORWARD, its vowel survives only as the mark `'`/`"`
  (`nayatu ā agram`→`nayatv " âgram`) — and stage 0 runs *before* the marks are undone, so `rvow` also
  counts a following `'`/`"` or a bare glide `y`/`v` as a vowel (`vai u X`→`vāy v X`, both reverted).
  Validated over the whole Vedic gold `Unsandhied`: 112 changed / 111→gold / 0 regressions, residual stray
  `-v`/`-y` = 0. **(0.5)** the **guṇa of a following vocalic ṛ/ḷ** (`_rev_guna_r`, the reverse of the
  `-a/-ā + ṛ- → -ar` rule `external_sandhi` has always generated): word2's initial `r`/`l` **before a
  consonant** → `ṛ`/`ḷ` (`ca rṣiḥ`→`ca ṛṣiḥ`, `etayā rcā`→`etayā ṛcā`, `vāmadevya- rco`→`ṛcaḥ`). Word1
  keeps its own vowel at this junction, so — unlike coalescence — **nothing is marked** and the only cue
  sits on word2; it is unambiguous because no native Sanskrit word may begin `r`/`l` + consonant. Runs
  AFTER stage 0, which must still see that word2 as consonant-initial: an *unreduced* `ṛ-` after `-a`
  means a dropped visarga (`-aḥ + ṛ- → -a ṛ-`, so `tata ṛṣiḥ`→`tataḥ ṛṣiḥ`), while the reduced `r-`
  shows word1's `-a` is genuine. 195 corpus tokens (Vedic train 155 / dev 26 / test 14) were previously
  left as `rṣiḥ`/`rcā`; recovery vs the gold pausa rises 94.698→94.785 % (train), 94.343→94.431 (dev),
  94.403→94.461 (test), with **0** tokens moving away from gold. **(1)** the notation-marked
  **vowel coalescence** (the left word's `'`/`"` + the right word's circumflex/macron mark
  `â ê î ô û / ē ō / âi âu` encode both original vowels) and **avagraha** (leading `'`→`a`);
  **(1.5)** the CONTEXT-SENSITIVE sibilant/palatal junctions, gated by a small **gold-derived lexicon**
  of genuine consonant-final / ch-initial stems (`_rev_sibilant_and_c`; `_SIB_FINAL` 17 / `_C_FINAL` 19
  / `_J_FINAL` 22 / `_L_FINAL` 1 / `_CH_INITIAL` 90, harvested from Vedic gold): word-final `-ś` before
  c/ch and `-ṣ` before ṭ/ṭh → visarga `-ḥ` (`kratuś ca`→`kratuḥ ca`) unless a genuine `-ś`/`-ṣ` stem
  (`diś`, `haviṣ`); word-final `-c`/`-j`/`-l` before their trigger → `-t` (`tac ca`→`tat ca`,
  `taj jal-`→`tat jal-`) unless a genuine stem (`vāc`, `rāj`, the `-añc` directionals — `-ñc` never
  arises from t+c so it is structurally genuine); and the `ch` of a `-c ch-` junction → `ś`
  (`paṭhec chiva`→`paṭhet śiva`) unless a genuine ch-word (`chāyā`, `chandas`);
  **(2)** the DETERMINISTIC final-consonant reductions — word-final `-s`/`-r` (after a vowel) →
  visarga `-ḥ` (`tatas`→`tataḥ`, `punar`→`punaḥ`, `agnir`→`agniḥ`); voiced stop `-d`/`-g`/`-b` →
  `-t`/`-k`/`-p` (place preserved: `tad`→`tat`, `id`→`it`); anusvara `-ṃ` before a non-sibilant
  consonant → `-m`; gemination `-nn` → `-n`. **(3)** the **law of finals** (avasāna), a per-stem map
  `_LAW_OF_FINALS` (57 entries, Whitney §141-2 + gold) that normalises a genuine consonant-final stem
  to its pausa form: `-c`→`-k` / `-ñc`→`-ṅ` (`vāc`→`vāk`, `ṛc`→`ṛk`, `pratyañc`→`pratyaṅ`), the
  lexical `-ś` split (`diś`→`dik` but `viś`→`viṭ`), the lexical `-j` split (`rāj`→`rāṭ` but `yuj`→`yuk`),
  and `-ṣ`→`-ḥ` for `-s`-stems (`haviṣ`→`haviḥ`) / `-ṭ` for genuine `-ṣ` (`ṣaṣ`→`ṣaṭ`). Applied to
  **compound members too** (`prāc-`→`prāk-`, the join marker preserved), so each stem has one pre-pausal
  form regardless of position; the treebank is itself inconsistent (writes both `ṛc`/`ṛk`, `prāñc`/`prāṅ`,
  `haviṣ`/`haviḥ`), so this collapses each genuine stem to ONE canonical pausa form. The hapax `dadhṛṣ`
  is omitted (uncertain place). **(4)** daṇḍa normalisation (`_normalise_danda`): every DOUBLE daṇḍa
  (`||`/`//`/`॥`/`।।`) → the single char `‖` (U+2016); a single daṇḍa stays `|`. Runs here so the corpus
  build (`revert_csl_sandhi`) and the runtime tokeniser agree. Stage 0
  runs BEFORE stage 1 so a coalescence-derived
  hiatus (L ends in the elision mark) is never mistaken for a dropped visarga (L ends in a plain vowel).
  **Punctuation: non-coalescent sandhi applies straight across a sentence-MEDIAL mark.** A comma /
  semicolon / colon / dash / guillemet / bracket (`_MEDIAL_PUNCT`) is a typographic overlay a modern
  editor lays over a phonological chain that does not pause there, so stages 0, 1.5 and 2 take their
  neighbour *through* it (`_next_word`): `tataś, ca`→`tataḥ ca`, `vatso, vipra-`→`vatsaḥ`, `kiṃ, bhadre`→
  `kim`, `tay, iti`→`te`. A **single daṇḍa counts as medial too, when the sentence is closed by a DOUBLE
  daṇḍa** — there it is only a pāda / half-verse boundary, a metrical rather than a phonological break, so
  sandhi reads across it (`tataś | ca gataḥ ||`→`tataḥ`). The test is DOCUMENT-dependent, exactly as in
  `clause_parser`'s `sent_scheme = "danda"`, so `_next_word` evaluates it over the whole token list: a
  single daṇḍa is medial iff some double daṇḍa is present. The two **coalescent** reversions (stage 1's
  marked coalescence and stage 0.5's guṇa ṛ) stay **adjacency-only** — coalescence fuses the two vowels
  into one syllable, so no mark can sit inside it. Everything else — `. ? ! …`, a double daṇḍa, and a
  single daṇḍa in a text with no doubles (there it IS the sentence end) — is a **pause** (avasāna) and
  blocks every rule: the words flanking it already stand in pausa form. Verified over the Vedic sandhied
  corpus by inserting a mark at all 147 368 non-coalescent external junctions — a comma, and a single
  daṇḍa under a closing `||`, each leave **0 / 206 440** reverted forms different from the un-punctuated
  reversion, while a double daṇḍa (or a lone single one) blocks 42 031 reversions as intended.
  This also settles the **anusvara at a pause**, consistently with the earlier decision that `-ṃ` before a
  vowel or a pause is a GENUINE anusvara: it is now left alone before a pause daṇḍa too (`oṃ ‖`), where the
  old code counted every daṇḍa as an ordinary following consonant and reduced it to `om` while leaving a
  sentence-final `oṃ` — a fix in passing, and the reason the guṇa-ṛ rebuild moves **0** tokens away from
  gold. (Before a *medial* single daṇḍa the reduction does fire, on the real following word.)
  Residual risk, accepted: an edition that *resets to pausa* before its commas gets a spurious visarga at
  `-a/-ā , V` junctions (`rājā, ṛtvijam`→`rājāḥ`) — but in genuinely sandhied text that junction would
  read `rājā, rtvijam` (stage 0.5), so the two conventions stay distinguishable in practice.
  **NOT reverted — genuinely ambiguous even with the lexicon:** the remaining palatal/retroflex finals
  `-j`/`-h` and a word-final `-c`/`-ś`/`-ṣ` at pause or before a non-triggering segment (`diś`→dik but
  `viś`→viṭ), and `-ā`/`-a` before a voiced consonant (a dropped `-āḥ`/`-aḥ` is indistinguishable from a
  genuine final vowel there); these stay on the surface, as do the unpaired-mark fallbacks (unpaired
  trailing `'`/`"` → `a`/`ā`; leading circumflex → restore). Measured on Vedic (round-trip through
  `external_sandhi` against gold): the bare-hiatus visarga rule is **100 % clean** (5922/5922 genuine
  visarga — since ayādi keeps its glide, it no longer collides), the ayādi glide reversal round-trips
  exactly, and the **lexicon-gated sibilant/-c junctions are 100 % on the gold** (word2 1668/1668; the
  `-ś`/`-ṣ` and `-c` guards 1230+438, zero genuine-stem mangling). The only accepted collision is the
  ~0.8 % `-u`-stem vocative in `-o`, plus the yaṇ length default. That `-o` collision costs **95 train
  tokens** (`go`→`gaḥ` 21, `viṣṇo`→`viṣṇaḥ` 19, `atho`, `bho` …) and the medial-punctuation transparency
  exposes it a little more, since a CSL editor sets vocatives off with commas (`bho, brāhmaṇa`→`bhaḥ`);
  a gold-derived `_O_FINAL` guard (the same shape as `_SIB_FINAL`, 47 types / 239 tokens) would fix it,
  but it would drift the corpus, so it is **not** implemented — left as a candidate for the next rebuild. **This depends on the glide-preserving
  ayādi in `external_sandhi.py`** — if the engine reverted to bare-hiatus ayādi, `-a/-ā` hiatus would be
  ~33 % ayādi and the visarga rule wrong.
  `revert_csl_sandhi.py` applies the *same* `desandhi_csl` to the sandhied CoNLL-U
  (rewriting FORM + MWT-range surfaces, regenerating `# text`) → `*.csl_rev.conllu`; `scripts/rebuild_sa_csl_rev.sh`
  is the full corpus driver (revert → `hyphen_to_pipe_sa` → `strip_pipe_sa` → combine Vedic-train+UFAL
  → convert), so training data and the runtime tokeniser produce identical forms. **Outstanding after the
  punctuation + guṇa-ṛ refinement:** the corpora on disk (and hence the trained arms) still carry the old
  forms at the **195** guṇa-ṛ tokens — the drift is 0.10 % of train and each new form (`ṛṣiḥ` for `rṣiḥ`)
  is the one already dominant in training, so no retrain was done; `rebuild_sa_csl_rev.sh` + base→morph→
  lemma **was** run (2026-07-26), so corpus, model and tokeniser are back in step. The punctuation change
  drifts **0** corpus tokens (Vedic has no punctuation; the UFAL commas and medial daṇḍas sit at junctions
  with nothing to revert) — it is purely an inference-side fix, and the rebuild moved only the 195 guṇa-ṛ
  tokens. Wheel repackaged (`package_lemma.sh sa`) and re-uploaded to the v0.1.0 release with `--clobber`
  (12 020 101 bytes, sha256 `e66317f3…`). Useful check when a release is meant to be **code-only**: diff
  the wheel against the previous asset file by file — the intermediate code-only wheel differed in exactly
  two of 29 files (`sa_tokenizer.py` + dist-info `RECORD`), proving the weights were untouched. The
  **source-offset** release (see the `src_spans` bullet above) is the same shape: code-only, three of 29
  files differing (`sa_tokenizer.py`, `clause_parser.py`, `RECORD`), 12 023 374 bytes, sha256
  `967bfe31…`, re-clobbered over v0.1.0.
- Trained on **CSL-reverted Vedic-train + UFAL** (`corpus_sa_csl_rev/`, `config_sa.cfg`, `--code
  scripts/sa_tokenizer.py`) → `training_sa_csl_rev/`. Gold-preproc on the pre-pausal-normalised Vedic
  test (after the yaṇ + t-assimilation (c/j/l) + law-of-finals + daṇḍa-norm + hidden-vowel glide-follower
  + medial-punctuation + guṇa-ṛ additions — the **current released arm**, retrained 2026-07-26):
  **LAS 54.71 / UAS 67.63 / TAG 89.52 / POS 88.49 / morph 79.57 / lemma 86.60** (dev LAS 54.93 / UAS 68.19
  / TAG 89.67 / POS 88.88 / lemma 86.77). The preceding (pre-punctuation) arm ran LAS 54.80 / UAS 67.94 /
  TAG 89.77 / POS 88.62 / lemma 86.43 — i.e. flat within seed noise, on a test set whose own FORMs move
  with each revert convention; both are well up on the earlier ayādi-only arm (54.03), with the
  representation now linguistically
  correct (see the ayādi/yaṇ/law-of-finals notes below). The pre-pausal normalisation collapses each
  word's sandhi variants to one form; the numbers are not strictly comparable across
  arms since the test FORMs themselves change with each revert convention. (comp:obl F moves with its
  denominator and sa comp/mod is un-relabelled, so it is not the metric of interest here.) UFAL was
  put **into training**, not held out; its 36 bare-hiatus junctions (no `Unsandhied=` to disambiguate)
  keep the residual visarga/ayādi ambiguity, negligible against 21 k Vedic sentences. (`corpus_sa_sandhi/` + `training_sa_sandhi/` are the
  superseded sandhied-surface arm; `backup_sa_prestrip/`, the earlier partial-revert corpora.)
- **comp/mod stays un-relabelled.** `scripts/ufal_compmod_probe.py` confirmed on classical UFAL
  that the LLM is at chance on the case-marked Ins/Acc/Gen residue (0.43 vs 0.82 majority), same as
  Vedic — structural (Sanskrit is case-based, not prepositional).
- **`udep@<subtype>` extension is a real (partial) fix, still not enough to release.** Beyond the
  bare-Case `sa_case` bucket (`relabel_ext.py`), the Vedic annotators separately tag `udep` with
  ten semantic-role subtypes (`@instr/@goal/@lmod/@tmod/@source/@manner/@soc/@benef/@grad/@path`,
  ~8850 tokens total) that the pipeline previously never touched at all (the bucket only fired on
  bare `udep`). `scripts/sa_subtype_audit.py` found only `@manner` has in-treebank commit evidence
  (626 `mod@manner` / 0 `comp:obl@manner`); the rest were classified from Case-distribution +
  governing-verb evidence: 7 subtypes (`instr/lmod/tmod/source/manner/benef/grad`) are dominated by
  cases/semantics already established as circumstantial (`SA_MOD_CASES`, Dat/Gen-of-purpose,
  Abl-of-comparison) → **mod**; `goal`/`path` are >85% headed by motion/placement/ritual-offering
  verbs (i, gam, āgam, praviś, āruh, dhā, nidhā, hu, nī...) taking an Acc/Loc goal-of-motion or
  path-traversed argument, the paradigm selected oblique → **comp:obl**, no verb-class gating
  needed; `@soc` sampled as a genuine mix (ingredient-mixing instrumentals vs. true accompaniment)
  → left for the LLM (515 instances, ~54%/46% comp/mod — a real split, not a default). Retraining
  `training_sa_ext` on this (`sa_subtype_label` in `relabel_ext.py`) moved test-gp **comp:obl F
  0.352→0.396** over the prior case-only ext arm, LAS flat (0.557→0.562) — a genuine improvement
  demonstrating the annotator subtypes carry real signal the Case-only view missed, but still below
  the un-relabelled base's 0.404 (comp:obl F has a moving denominator across relabels, per the usual
  caveat) — **not enough to change the released (un-relabelled) model.**
- **Bundles `clause_parser`** (`punct_tag="PUNCT"`; `DEFAULT_PUNCT` already covers the daṇḍa
  ।॥ `|` `||` / . ? !), like the former `sa_sud_vedic`, for raw multi-clause inference; packaged
  with `spacy package … --code scripts/sa_tokenizer.py,scripts/clause_parser.py`. The shipped
  pipeline is `[tok2vec, tagger, parser, clause_parser]`.

### `Compound=Yes` as a tokeniser-supplied INPUT feature (sa only; +1.30 LAS)

The sa arm is the first to **read morphology as an input feature** rather than only predict it. The
tokeniser sets `Compound=Yes` deterministically and every component's embed lists **`MORPH`** among
its attrs. This is a deliberate divergence from the other ten arms' recipe (user decision), on the
expectation that other morphology-heavy languages (ar/la/fa/ko) will follow.

- **Why it exists.** Stripping the compound-join marker to leave clean wordforms (`strip_pipe_sa.py`,
  above) cost ~0.5 LAS, because the join marker had been *visible* to the parser inside the token
  form. This puts the cue back as a feature instead of as text: `sa_tokenizer.__call__` records which
  tokens carried a join marker before it strips them, and stamps the feat.
- **Deriving it (`_NON_COMPOUND_JOIN`, 23 types).** The CSL hyphen joins THREE things: samāsa members
  (`Compound=Yes`), verb preverbs (upasarga, ADV), and the privative `a-/an-` (PART). Only the first
  is `Compound=Yes`. The bare join marker predicts the feat at only **P 0.775 / R 0.713**; excluding
  the closed upasarga+privative list (harvested as the join-member types predominantly NOT Compound)
  lifts it to **P 0.9998 / R 0.9997** (TP 6463, FP 1, FN 2), with zero false exclusions in dev/test.
  `sam`/`su` each lose one genuine compound occurrence — the entire cost. The 2 598 gold
  `Compound=Yes` tokens the rule misses all have FORM `_` (elided; they exist only in the treebank).
- **Making it an input needs three pieces, and omitting any one breaks it silently.**
  (1) **`sud.CompoundCorpus.v1`** (`scripts/gold_tok_corpus.py`) — under `gold_preproc` the predicted
  doc is built from gold words and the tokeniser **never runs**, so the feature would be absent in
  training and present at inference. The reader copies **only** `Compound` from the reference;
  copying any other feat would be leakage, and this one is not, precisely because the tokeniser
  supplies the identical value at runtime. (2) **`clause_parser`** re-imposes the tokeniser's verdict
  (the morphologizer overwrites `token.morph`) and carries it into the sub-doc it builds for the
  per-clause re-parse, else the parser re-runs on input it never saw in training. (3) The configs
  swap `HashEmbedCNN` (which hard-codes NORM/PREFIX/SUFFIX/SHAPE) for the equivalent explicit
  `MultiHashEmbed` + `MaxoutWindowEncoder`, so `MORPH` can be listed at all.
- **Result (test gold-preproc, baseline → Compound arm; `training_sa_lemma3_noannot`):**
  **LAS 0.5471→0.5601 (+1.30), UAS 0.6763→0.6928 (+1.64), morph_acc 0.7957→0.8050, pos 0.8849→0.8896,
  tag 0.8952→0.8976, lemma 0.8660→0.8645 (−0.15, flat)**; `Compound` F **0.889→0.999**, and the gain
  propagates to Mood +0.009 / Tense +0.008 / Person +0.007 / Case +0.005 (Voice −0.081, smallest class).
- **Token input: the `sa_compound` fallback component** (`scripts/sa_tokenizer.py`, added FIRST in
  the pipeline). A `Doc` built WITHOUT the tokeniser (`Doc(vocab, words=...)`, pre-tokenised input,
  `spacy evaluate`) has no `Compound`, and the model then runs with one of its inputs deleted —
  measured **LAS 0.5601→0.5169** on token input, with no warning. `sa_compound` closes that: if the
  tokeniser ran (`doc._.compound_flags` set) it defers, else it re-derives the feat from token
  adjacency (no intervening space, minus `_NON_COMPOUND_JOIN` and punctuation junctions). **On real
  text it is exact — 19 584/19 584 tokens agree with the tokeniser, precision 1.0000.**
- **CAVEAT — a PARTIAL `Compound` is worse than none, so do NOT evaluate through the fallback.**
  The one thing adjacency cannot see is an **elided** compound member: the treebank writes those
  FORM `_` with a trailing space (282 in test; every unrecoverable token is a `_`). They cannot
  occur in real input, but on the treebank the gap is actively harmful — token input scores **LAS
  0.5169 (no feat) / 0.4826 (fallback) / 0.5601 (full feat)**. Training always had the feat on every
  compound member, so an *unmarked* member reads as positive evidence of "not a compound", which is
  worse than uniform absence. Marking every `_` is not a fix (only 81 % are compounds; it would
  destroy the precision-1.0 property). **Evaluate with `scripts/eval_sa_compound.py`** (supplies the
  feat from the reference via `CompoundCorpus`), NOT `spacy evaluate` and NOT via the fallback;
  `--plain` reproduces the broken measurement for comparison.

### NEGATIVE RESULT: do NOT widen sa PREFIX/SUFFIX (costs 2.9 LAS)

`PREFIX`/`SUFFIX` are plain entries in `lex_attr_getters` (`spacy/lang/lex_attrs.py`, `string[0]` and
`string[-3:]`), so a language may widen them — `sa_tokenizer` does it on `Sanskrit.Defaults`, which
affects **sa only** (verified: la/ar/fa/ko/en keep 1/3). `SA_PREFIX_LEN`/`SA_SUFFIX_LEN` override them
for ablations; they are NOT a runtime knob (a model trained at one width and loaded at another
degrades silently, with nothing in the config to catch it). **The shipped arm uses spaCy's 1/3.**

Widening to PREFIX 3 / SUFFIX 6 was tried and **regressed everything but the tagger**: vs the Compound
arm, LAS −2.9, morph_acc −3.8, lemma −3.7 (tag +0.16). Why the reasoning that motivated it was wrong:
the width was sized from the **form→lemma edit** (SUFFIX 6 covers 96.9 % of it, vs 80.8 % at 3), which
describes what a **lemmatiser** needs — but three of the four components want the short, widely-SHARED
inflectional ending, and at 6 characters, in a language whose median word is 5–6, the suffix is close
to word-identity (23 433 types vs NORM's 32 854; 64 % of tokens have the whole word as their
"suffix"), so it memorises instead of generalising. Sizing compounded it: 2000 rows for 23 433 types
in the small encoders = 11.7× collision, against 1.9× at length 3. **Affix width is a lexeme
attribute, so all four components in a language must share one value** — it cannot be tuned per
component without a custom embed layer that hashes `token.text[-k:]` at forward time.
NB the two arms differed in BOTH widths, so prefix vs suffix blame is **not** separated; a
single-variable run (suffix 4, prefix 1) is the informative one if this is ever revisited.

Also ruled out: **`annotating_components = ["morphologizer"]`** on the lemma config (so the lemmatizer
conditions on predicted FEATS instead of nothing) is **not** worth it — lemma_acc 0.8627 with vs
0.8645 without. Predicted `Case` at F 0.856 adds about as much noise as signal to an edit-tree
classifier that already has the whole form in `NORM`. Left at `[]`, matching the other ten arms.
