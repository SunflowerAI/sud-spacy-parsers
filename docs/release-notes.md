# Release notes — what changed in each wheel, and when

Wheels are published on the [GitHub Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases)
page, not in git. This page records what users actually have; how the release is *built and verified*
is [`packaging-and-release.md`](packaging-and-release.md).

## The versioning rule, and why `pip install -U` is not enough

**The 0.2.0 set is re-clobbered in place as layers land.** A wheel can be rebuilt and re-uploaded at
the same version, so `pip install -U` will **not** pull the new copy — the version is unchanged, and
pip has nothing to compare. Use `--force-reinstall` when a note below says a wheel was re-clobbered.

Four wheels took a version bump instead of a clobber (ja, ko, la, sa at **0.3.0**) because their
changes were large enough that silently leaving old copies installed would have been wrong.

The asset list, with the upload times that reveal a clobber, is one command:

```bash
gh release view v0.3.0 --json assets -q '.assets[] | "\(.name)  \(.updatedAt)"'
```

⚠ **Re-derive this page rather than trusting it.** It goes stale faster than anything else in the
repo — sa shipped 0.3.0 within a day of it last being written.

## Current state

| release | wheels |
|---|---|
| [`v0.3.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/v0.3.0) | `ja` `ko` `la` `sa` at 0.3.0; `ta` `te` new at 0.1.0; `en_gum` 0.3.0 uploaded 2026-10-28 01:26 UTC, sha256 `b9bf07c9…4b6b8`, re-verified by downloading the asset (entry below) |
| [`v0.2.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/v0.2.0) | `en` `en_gum` `ar` `fa` `id` `zh` `yue` `lzh` at 0.2.0 |
| [`vectors-v0.1.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/vectors-v0.1.0) | 13 aligned cross-lingual 128d vector assets — see [`aligned-vectors.md`](aligned-vectors.md) |

## Change log

### 2026-10-28 — `en_gum` 0.3.0 (TinyBERT-frozen base arm) — UPLOADED 2026-10-28 (hash re-verified on the downloaded asset)

A new architecture and new weights, so this takes a **version bump rather than a clobber** (the ja/ko/la/sa
convention above): the 0.2.0 asset stays on `v0.2.0` and is the rollback. Uploaded to the existing
[`v0.3.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/v0.3.0) release (new asset name, so no
`--clobber`); `pip install -U` cannot pull it either way, since wheels are installed from release URLs.

**What changed.** `tok2vec` is now a **frozen TinyBERT-4L-312D** (`huawei-noah/TinyBERT_General_4L_312D`, uncased)
with a learned softmax mix over all five hidden states and one learned scalar, first word-piece per word,
concatenated with the previous 96-wide hash-embed CNN (listener width 408). `parser` is retrained on it. Everything
else in the pipeline keeps its place: `[tokenizer, tok2vec, parser, morphologizer, lemmatizer, tagger, sud_subject,
sud_shared, sud_reported_rule, sud_idiom]`.

**No torch at inference.** The wheel's encoder is `sud.TinyBertPlusHashNumpy.v1` (`scripts/sud_tinybert_infer.py`):
a numpy forward pass of the 4 BERT layers and a pure-python WordPiece tokenizer. Weights, the 30 522-entry vocab,
the BERT hyper-parameters, the mix logits and gamma are **sealed in `tok2vec/model`** (63 MB); the config names no
path (standing hazard 4). The training module `sud_tinybert_embed.py` (torch) is untouched and is not in the wheel.
`scripts/seal_tinybert_tok2vec.py` converts a trained arm and refuses to write unless the twin matches torch.

**Equivalence, measured** (`scripts/check_tinybert_equiv.py`, torch arm against sealed arm, 302 test documents plus
ten stress documents: 900- and 2500-word documents that must chunk at 256 pieces, the empty document, whitespace-only
tokens, a literal `[SEP]`, CJK, accents, 255/256/257-token runs):
max |torch − numpy| on the tok2vec output **8.3e-6**; full-pipeline analysis (head, deprel, tag, POS, lemma, morph,
sentence start, `sud_misc`) identical on **47 682 / 47 682 tokens**. The WordPiece port was fuzzed against
`BertTokenizerFast` on all 29 419 word types in train+test plus 30 000 random Unicode strings: **0 mismatches**
(the fuzz found two places where the fast tokenizer differs from the textbook BERT rules: unassigned code points
are kept rather than dropped, and a literal `[SEP]` inside a word is id 102).

**What was rebuilt, and why** (`CORPUS_TAG=vnrulings6 bash scripts/build_en_gum_chain.sh tinybert`, ~47 min).
The morphologiser, lemmatiser and XPOS tagger each have their own encoder and train on gold, so they do not read the
base: re-trained, they come out **byte-identical** to the 0.2.0 asset (`cmp`), which is why tag, POS, morph and lemma
scores are *identical* to 0.2.0 (tag 94.45) — the base arm's own 94.74 tagger is not what ships (the conditioned
XPOS tagger replaces it, as before). What does read the base is `sud_subject`/`sud_shared` (they annotate from its
parse, standing hazard 5), so those were re-trained against it; `sud_reported`/`sud_idiom` are rules and were
re-measured. Wheel versus 0.2.0, file by file: `config.cfg`, `meta.json`, `README.md`, `tok2vec/model`,
`parser/model`, `sud_subject/model`, `sud_shared/model` differ; `tagger`, `morphologizer`, `lemmatizer`, tokenizer,
vocab are byte-identical; `NOTICE/` added.

**Full deployed pipeline, both wheels measured in clean venvs from `/tmp`** (0.2.0 downloaded from the release;
it reproduces `metrics_release_en_gum*.json` exactly). Test split of `corpus_en_gum_sud_vnrulings6` for
`--gold-preproc`, of `corpus_en_gum_vnrulings6` (udep-resolved) for raw:

| | 0.2.0 gp | 0.3.0 gp | 0.2.0 raw | 0.3.0 raw |
|---|---|---|---|---|
| LAS | 82.90 | **84.98** | 81.71 | **83.28** |
| UAS | 87.13 | 88.66 | 85.79 | 86.93 |
| sentence F | 90.87 | 89.26 | 76.47 | 79.10 |
| TAG / POS / lemma | 94.45 / 94.66 / 96.24 | identical | 94.45 / 94.57 / 95.50 | identical |
| `comp:obl` F / `mod` F | 70.7 / 76.6 | 74.0 / 78.5 | 71.1 / 76.5 | 74.4 / 77.6 |

The one cost is gold-preproc sentence F (−1.6); raw sentence F rises 2.6. Files: `metrics/en_gum/metrics_en_gum_{020dl,030}_{gp,raw}.json`.

**MISC layer re-measured** (`scripts/eval_misc_en_gum.py`, gold tokens, test; 0.2.0 base → 0.3.0 base). No ship
decision reverses. Subject (trained) F 78.43 → 78.13 (rule 63.71); Shared (trained) 61.12 → 63.06 (rule 53.75);
Reported (rule) 63.01 → 66.67 on n = 42 (trained 31.01); Idiom 80.19 → 80.38 and InIdiom 80.00 → 79.82 (rule).
The Subject and Idiom changes are inside noise; the Reported figure rests on 42 tokens.

**Serverless sizing** (clean venvs, NumpyOps, laptop under other load, so latency is a range): wheel 22.9 → **76.3 MB**;
site-packages 160 → **215 MB**; cold load 0.15–0.4 s for both; peak RSS 155 → **331 MB** (5 200-token document:
355 → 594 MB); p50 per single sentence 2.9–5.6 ms → 4.7–10 ms; batched gold-preproc throughput 12.1k → 4.2k words/s.
Fits a 512 MB function; the 63 MB tok2vec is float32 (fp16 would halve it; not done, since it breaks the 1e-6 equivalence).

**Caveats, stated plainly.**
- **SINGLE SEED.** The TinyBERT arm is seed 0 only. The 0.2.0 recipe has three seeds (raw LAS 81.71 / 81.76 / 81.91,
  sd ≈ 0.1), so +1.6 LAS is about 15 of its standard deviations, but the TinyBERT arm's own seed spread is unmeasured.
  Sentence F has a seed sd of about 0.8, so the 2.6-point raw gain is only ~3 sd of the old arm and uncertain.
- The `sud_subject`/`sud_shared` retraining is also single-seed.
- `en_sud_ewt` (EWT only) is NOT updated; it stays on the hash-CNN base.
- **Licence.** The encoder and vocab are Apache-2.0 (`NOTICE/` in the model directory carries the licence text and
  notice; `meta.json` lists the source). The only licence statement found is the Apache-2.0 `LICENSE` in the
  upstream TinyBERT project directory; the Hugging Face model card carries no licence field. The vocabulary is BERT-base-uncased's.

**Commands run** (all from the repo root, `.venv/bin/python`): `ln -s training_en_gum_tinybert_hash_frozen_mix_s0 training_en_gum_tinybert`;
`CORPUS_TAG=vnrulings6 bash scripts/build_en_gum_chain.sh tinybert` (train morph, lemma, xpos, sud_xpos, xposwarm, then
`graft_pipe.py`/`graft_xpos_tagger.py`: parse unchanged 8585/8585, donor tags reproduced 8585/8585);
`bash scripts/package_sud.sh en_gum` (seals, adds the Reported rule and Idiom, packages as 0.3.0);
`.venv/bin/python scripts/check_tinybert_equiv.py training_en_gum_sud_xw_tinybert build_sud/work_en_gum.seal corpus_en_gum_sud_tinybert/test.spacy`;
`scripts/eval_misc_en_gum.py <arm> --rules` for both arms; and in clean venvs, from `/tmp`,
`python -I -m spacy evaluate en_sud_ewt_gum <test.spacy> [--gold-preproc] --output ...` for both wheels.

### 2026-10-26, 08:40 UTC — `en_gum` re-clobbered at 0.2.0 (extended `comp:obl`/`mod` rulings)

Trained on the corpus relabelled by the extended VerbNet-class rulings (`docs/udep-relabel.md`,
"Extensions to the 2026-10-26 rulings"). Rebuilt with `scripts/build_en_gum_chain.sh vnrulings6`.
Only `tok2vec`, `parser`, `sud_shared` and `sud_subject` (plus config, metadata and strings) differ
from the previous asset; morphologiser, lemmatiser, tagger, tokeniser and vocab weights are
byte-identical, and tag, POS and lemma scores are identical. Verified on the DOWNLOADED wheel
(sha256 `e3a104fb…1ed6`, 22 945 697 bytes), loaded in a clean venv from a directory unrelated to the repo.

Base arm, three seeds per corpus (`scripts/seed_check_en_gum.sh`; the test gold differs between the
two corpora, so these are trends): LAS 81.11 → **81.79** (higher on every seed), UAS 85.92 → 85.79,
sentence F 77.79 → 76.98, `comp:obl` F 70.1 → 70.5, `mod` F 75.3 → 76.1. The sentence F and UAS
differences are inside the seed spread (sd about 0.8 and 0.3). A single-seed comparison had shown
−2.6 sentence F; the old corpus's seed 0 was a lucky draw.

Known leaks left in the labels: selected PPs kept as `mod` (`rely on`), locative adjuncts promoted
by G2/Wiktionary, and `from` on extent verbs. `metrics/release/metrics_release_en_gum.json` is now the `--gold-preproc` scoring, like every other
language's release file, measured on the DOWNLOADED wheel (test split of `corpus_en_gum_sud_vnrulings6`,
the corpus its own config names): LAS 82.90, UAS 87.13, tag 94.45, sentence F 90.87. The RAW scoring is
`metrics_release_en_gum_raw.json` (LAS 81.71, UAS 85.79, POS 94.57, lemma 95.50, sentence F 76.47,
`comp:obl` F 71.1, `mod` F 76.5) and is the figure for running text, since gold pre-segmentation hides
how well the parser starts sentences (standing hazard 11). The previous file (LAS 81.86, UAS 86.83,
sentence F 91.99) was most likely `--gold-preproc` too, though its setup was not recorded. Neither wheel's training config sets `gold_preproc = true`
(a reader option with no effect at inference). Rollback copy of the previous asset: `release_backup_en_gum_v0.2.0/`
(untracked).

### 2026-08-23 — `sa` re-clobbered at 0.3.0

The `Reported` rule, re-tuned against the base it actually ships on: test F 48.76 → **53.05** as the
wheel runs. A **code-only** change — all six weight files are byte-identical to the previous asset,
verified out of the downloaded wheel. See [`sud-misc-layer.md`](sud-misc-layer.md).

### 2026-08-22, 19:47 UTC — `zh`, `id`, `lzh` (0.2.0) and `ta` (0.1.0) re-clobbered

The `SEG_BATCH` memory cap in `char_seg_tokenizer.py` and `ta_tokenizer.py`. A **source-only**
change: every model byte is unchanged. Each wheel was rebuilt by unpacking the released asset,
editing the bundled file and repacking, so `RECORD` and those two files are the only entries that
differ, and all four reproduce their previous token stream and full-pipeline parse digest exactly.

The defect it fixes is worth reading even if you never hit it: a tokeniser that batched its whole
input into one `predict` call cost 10–14 kB per character of the calling string. Every metric in this
repo is computed sentence by sentence, so it looked healthy for four languages and was invisible
until someone handed it a book. `te` was not touched — its lookup splitter runs no segmenter.

### 2026-08-22, earlier — `zh` re-clobbered at 0.2.0

The traditional jieba dictionary replaced the `t2s`-the-text channel. **Only the tokeniser changed**
— every model weight is byte-identical to the previous asset, verified by hashing them out of the
downloaded wheel — so the gold-preproc results are untouched and only the raw figures moved. See
[`results-notes.md`](results-notes.md).

### 2026-08-19 — `lzh` and `yue` re-clobbered; `ta` and `te` first released

`ta_sud_ttb_mwtt` and `te_sud_mtg` released at 0.1.0, both CC BY-NC-SA 3.0. Verified after upload by
hashing `parser/model` and `tok2vec/model` out of the *downloaded* assets, then installing from the
public release URL into a clean target with `scripts/` off `sys.path`. See
[`dravidian.md`](dravidian.md).

### 2026-08-08 — ⚠ `zh` shipped unable to segment

The wheel published on 2026-08-08 **could not segment at all**: it shipped with no
`tokenizer/segmenter/` directory and returned each input string as a single token. It has been
rebuilt and re-uploaded at the same version.

**If you installed `zh_sud_gsd` before 2026-08-09, reinstall with `--force-reinstall`** — `pip
install -U` will not replace it, since the version is unchanged. Every model weight is byte-identical
between the two builds, so no score moved: the gold-preproc figures never ran the tokeniser and were
correct throughout.

The lesson is in [`packaging-and-release.md`](packaging-and-release.md): a component that silently
loses an input must refuse to load, and the model to verify is the **reloaded** one, never the
in-memory one.

## Which release figures were stale, and on what

`metrics/release/*.json` is measured on the arm each wheel ships, identified by hashing
`parser/model` out of the **downloaded** wheel. A training directory of the right name is not
evidence. Re-measured 2026-08-23 against the downloaded assets. **Ten of the fourteen wheels were
quoting a figure from an arm they no longer ship.**

| file | was | is | why |
|---|---|---|---|
| `metrics_release_la.json` | LAS 71.72 | **73.23** | described the pre-lemma-vector arm; the 0.3.0 wheel is `training_la_lemvec_sud` |
| `metrics_release_la_ittbproiel.json` | LAS 75.90 | **77.67** | same |
| `metrics_release_la_perseus.json` | LAS 53.53 | **53.91** | same |
| `metrics_release_sa.json` | LAS 37.35 | **48.54** | described the 0.1.0 wheel; two generations of retraining have landed since |
| `metrics_release_ta.json` | *(absent)* | LAS 59.73 | ta was released without one |
| `metrics_release_te.json` | *(absent)* | LAS 69.06 | te was released without one |

`en`, `en_gum`, `id`, `zh`, `ja` and `ko` were re-verified and reproduce their files **exactly**,
which is what validates the method for the six re-measured above.

**`ar`, `fa`, `lzh` and `yue` were stale for the same reason, and it is the one this repo keeps
paying for: the arm was re-released and the metric was not re-measured.** Every one of the four
release files was written by `7191f11` on 2026-08-12, and every one of the four wheels was rebuilt
after that date on a *different arm*:

| | published (08-12) | the arm actually shipping | re-measured |
|---|---|---|---|
| `ar` | 83.67 / 77.34 / 62.90 | `training_ar_vocal_sud_idiom` — the vocalisation-augmented arm adopted by `ea1886f` (08-14) | **83.05 / 76.76 / 62.78** |
| `fa` | 90.61 / 87.18 / 79.20 | `training_fa_vocal_sud_xpos` — same commit | **90.98 / 86.29 / 79.81** |
| `lzh` | 82.92 / 77.20 / 66.47 | `training_lzh_seg_sud_xw` — the sentence-segmenting arm, after `0d49e18` (08-19) restored the annotators' `@tmod`/`@lmod` | **81.98 / 76.46 / 67.11** |
| `yue` | 72.37 / 64.51 / 46.15 | `training_yue_sud_xpos` — after `05539b9` (08-19) kept yue's subtypes and fixed three packaging defaults that named the wrong arm | **75.22 / 67.29 / 52.17** |

Each wheel's `parser/model` was hashed out of the downloaded asset and found **byte-identical to the
arm `package_sud.sh` names**, so no model is stale and nothing needed retraining — only the numbers
were wrong. Each was then re-evaluated on the test split of the corpus its own `config.cfg` names
for train and dev, which is what the earlier attempt got wrong: `fa` on `corpus_fa_ext` rather than
`corpus_fa_sud` reads LAS 87.28 instead of 86.29, and picking the corpus by name rather than by
asking the arm is how a one-point error gets into a table.

The two subtype restorations move `comp:obl` F most (`yue` +6.0, `lzh` +0.6), which is expected:
they change the label inventory the metric is computed over. **Rare labels pay first, and they are
also the first to move when the gold moves.**

⚠ **`ja` must be scored through `scripts/eval_ja_infl.py --reader infltag`.** It reads a tokeniser-supplied input channel that
`spacy evaluate --gold-preproc` does not build, and scoring it with the stock reader reports LAS
72.06 instead of 90.04 — a measurement of the model with one of its inputs deleted.
