# Packaging and release

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

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
  `spacy>=3.8.14` and hit an ImportError on every load; zh USED to declare `jieba>=0.42.1` (it now
  vendors it, see the vendoring note below), yue
  `spacy-pkuseg`, and **ar `camel-tools>=1.5.2`** — its tokeniser raises at LOAD time, so before this
  a plain `pip install ar_sud_padt` produced a model that could not be opened. Per-language
  requirements now live in `stamp_model_meta.py`, which already runs for every arm at packaging.
  The `camel_data -i …` download still has to be run by hand: a data fetch is not expressible as a
  pip dependency, so this reduces the missing pieces from two to one rather than to none.
- **zh VENDORS jieba, and declares no jieba requirement** (`scripts/vendor_jieba.py`, run from
  `package_sud.sh`'s zh branch between `add_zh_script.py` and `pkg`). The pip distribution is 42 MB
  of which the BMES channel reaches ~6 MB: `posseg`, `lac_small` and `analyse` are never imported —
  established by reading `sys.modules` after `zh_jieba_feature.jieba_codes`, not by inspection. The
  reachable subset ships inside the model dir as `vendor/jieba/`, which needs no setup.py change
  because `spacy package` copies the model dir wholesale and its `list_files` walks it recursively.
  **Wheel 11.8 → 13.9 MB; any install of it drops 36 MB** (site-packages ~190 → 155 MB, which is what
  brings it inside a 250 MB serverless budget). `zh_jieba_feature._import_jieba` prefers an INSTALLED
  jieba and falls back to the vendored tree, so training is unaffected and a user's own copy wins.
  `scripts/slim_jieba.py` does the same pruning to a deployment tree for anything not vendored; both
  share one allowlist so they cannot drift.
  ⚠ The vendor search must look ONLY in the module's own directory and one glob level
  (`<pkg>/<name>-<version>/vendor`). An earlier version walked up two parents and bound an unrelated
  `vendor/jieba` from a different tree — silently, because the contents happened to match.
  ⚠ jieba ships **no LICENSE file** (only `License: MIT` in its METADATA), so vendoring means writing
  the notice: `vendor/jieba/NOTICE` records version, upstream, author and that the files are
  unmodified copies. Redistribution without it would not satisfy MIT.
- **Do NOT prune `dict.txt`.** It is 4.8 MB of the surviving 6.4 MB and it IS the feature — the
  channel's value is vocabulary (the traditional-vs-`t2s` gap is entirely vocabulary), worth +4.42
  token F. Baking the channel into the weights was considered and rejected for the same reason.
- **Training-only imports must not be module-scope in a bundled file.** `sa_presegment` importing
  `sa_tokenizer`, and `sa_presegment_lex` importing `eval_samhita`, both broke the zh wheel.
- **A component that silently loses an input must refuse to load.** `bundle_zh_charseg.py` REFUSES
  to write a model whose saved `vocab.json` lacks the `jieba_source` marker — without it the wheel
  would load, run with one input deleted, and say nothing (the same silent degradation as sa's
  `Compound` on token input).
- **Verify in a clean `--target`/venv install**, not just the loose training directory.
- For a **code-only** re-release, diff the wheel against the previous asset file by file: the sa
  code-only wheels differed in exactly 2 and 3 of 29 files, proving the weights were untouched.

## Release audit, 2026-08-04 — and the lesson

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

## Serverless deployment, and the zh re-release of 2026-08-10

The target is a serverless function: one text per invocation, billed on duration, with a package
size ceiling (AWS Lambda 250 MB unzipped). Measured on the shipped `zh_sud_gsd`:

    clean install    155 MB site-packages (was ~190 before vendoring)
    cold model load  ~1.0 s        p50 per request  1.4 ms (AppleOps) / 3.65 ms (NumpyOps)
    peak RSS         ~425 MB       => needs a 512 MB function, realistically 1 GB

Three things this settles, each of them a bigger lever than anything in the model itself:

- **A torch dependency is disqualifying.** The CPU wheel is ~500 MB unzipped on its own, so
  `spacy-experimental`'s biaffine parser could never have shipped here whatever it scored. Only a
  pure-Thinc implementation is deployable at all, whatever its accuracy.
- **The BLAS backend is worth more than the parser.** A plain `pip install spacy` gets `NumpyOps`;
  `thinc-apple-ops` (the `spacy[apple]` extra) gets `AppleOps`, and that alone is **2.6x per
  request** on identical weights. Whatever the deployment platform, check which ops class actually
  resolves before optimising anything else — it dwarfs the differences between model architectures.
- **Batched throughput is the wrong metric here.** Nothing amortises across a single-text
  invocation, so per-request cost is roughly double what a 3000-doc batch suggests (the BiLSTM is
  1.24x batched and **1.77x** per request; beam 4 is 3x batched and 7x per request).

The 0.2.0 zh wheel was **re-released (clobbered) on 2026-08-10** carrying the vendored jieba. Diffed
against the previously published asset: 30 of 43 files byte-identical, 5 changed (two `meta.json`,
METADATA, RECORD, `zh_jieba_feature.py`), 8 added, none removed, **no weight or vocab file moved** —
so no published zh score changes. Verified by downloading the asset and by a clean venv with jieba
genuinely absent: identical full-parse digest on 500 test sentences. ⚠ Same version, so
`pip install -U` will NOT replace an older copy; `--force-reinstall` will.

⚠ **`find build_sud -name '*.whl'` is still the wrong upload command** and the build tree proves it:
eight wheels sit there, four at 0.1.0. Upload by NAME.

## The zh wheel that could not segment, 2026-08-09

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

## Metrics files: which is which, and which are stale

**`metrics_release_*.json` is the RELEASED arm; every other `metrics_*.json` is a development one.**
The distinction earns its keep, because several development files outlived the generation they
describe and the README was quoting them: the en row was a RAW run in a table declaring
gold-preproc (79.63/84.40 raw vs **81.33/86.26** gold-preproc), and ar/la/yue were still the `_ext`
arms from before the segmentation retrain the wheels actually contain (la 73.95 → **72.26**,
ar 78.45 → **77.34**, yue 65.64 → **64.51**, with `comp:obl` F moving as far as yue 26.7 → 46.2).
The release set was measured on the arm each wheel ships, identified by hashing `parser/model` out
of the DOWNLOADED wheel — a training directory of the right name is not evidence.

**Known-stale fields in the release set**, all of them TAG and all of them from the XPOS work
(`docs/xpos.md`); every other field in these files is unchanged and still correct, because every
other component is byte-identical:

- `metrics_release_la*.json` — holds the pre-normalisation TAG.
- `metrics_release_la_{ittbproiel,perseus}.json` — the per-slice TAG was not re-measured after the
  conditioned tagger was grafted.
