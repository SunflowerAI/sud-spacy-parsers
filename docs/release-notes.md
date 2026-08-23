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
| [`v0.3.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/v0.3.0) | `ja` `ko` `la` `sa` at 0.3.0; `ta` `te` new at 0.1.0 |
| [`v0.2.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/v0.2.0) | `en` `en_gum` `ar` `fa` `id` `zh` `yue` `lzh` at 0.2.0 |
| [`vectors-v0.1.0`](https://github.com/SunflowerAI/sud-spacy-parsers/releases/tag/vectors-v0.1.0) | 13 aligned cross-lingual 128d vector assets — see [`aligned-vectors.md`](aligned-vectors.md) |

## Change log

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
evidence. Re-measured 2026-08-23 against the downloaded v0.3.0 assets:

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

⚠ **`ar`, `fa`, `lzh` and `yue` could not be re-derived exactly**, and the gap is not explained. Each
released parser was found byte-identical to an arm still in the training tree, so none of them is a
newer generation than the file describes — but no local test corpus reproduces the published figure:

| | published | best local re-derivation | on |
|---|---|---|---|
| `ar` | 83.67 / 77.34 / 62.90 | 83.05 / 76.76 / 62.78 | `corpus_ar_sud/test.spacy` |
| `fa` | 90.61 / 87.18 / 79.20 | 90.98 / 87.28 / 79.81 | `corpus_fa_ext/…relabeled_ext.spacy` |
| `lzh` | 82.92 / 77.20 / 66.47 | 81.98 / 76.46 / 67.11 | `corpus_lzh_trad/…rulemerged.spacy` |
| `yue` | 72.37 / 64.51 / 46.15 | 75.22 / 67.29 / 52.17 | `corpus_yue_sud/test.spacy` |

The deltas run in both directions and are ≤3 points, which is the signature of a **regenerated test
corpus** rather than a changed model — these four have all been through relabel or normalisation
passes (`reparandum` → `conj:dicto`, the `udep`-ruled residue commits) that rewrite gold. The four
release files are left as published, since they were measured on the arm at release time; the README
quotes them. Resolving this means finding which corpus generation each was measured on, and it is
the obvious next job for anyone touching these four.

⚠ **`ja` must be scored through `scripts/eval_ja_infl.py --reader infltag`.** It reads a tokeniser-supplied input channel that
`spacy evaluate --gold-preproc` does not build, and scoring it with the stock reader reports LAS
72.06 instead of 90.04 — a measurement of the model with one of its inputs deleted.
