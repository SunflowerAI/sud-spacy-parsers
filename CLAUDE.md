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
- **id — coarsen the treebank.** Enclitics (`-nya/-lah/…`) are lexically ambiguous (`-lah` is a
  clitic 73× but inside whole words like `adalah`/`salah` 1723×) and not rule-separable, so
  `coarsen_id.py` merges each MWT range (host+enclitic) into one whitespace token, which the rule
  tokeniser reproduces deterministically. token-F1 vs spaCy 0.955→0.989.
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
