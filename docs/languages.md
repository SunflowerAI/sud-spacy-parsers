# Language notes: English, Indonesian, Korean

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

## English — TWO arms, one licence (`en_sud_ewt`, `en_sud_ewt_gum`)

`en_sud_ewt` (CC BY-SA 4.0, EWT only) is unchanged. `en_sud_ewt_gum` (**also CC BY-SA 4.0**, since
2026-08-17 — see below) adds the ten non-NonCommercial GUM genres — 340,324 train tokens, +66 % on
EWT. Built by `scripts/build_en_ewt_gum.sh` (steps `merge relabel fix verify filter corpus base`),
then the ordinary `train_morph → train_lemma → train_sud → package_sud` chain, which all take
`en_gum` as an arm name.

**Why two wheels — and why the licence stopped being the reason.** GUM's LICENSE says the treebank
is CC BY-NC-SA *and* that the NC comes from the individual sources; the second reading supports
filtering, the first offers the ANNOTATIONS under NC whatever the document — and annotations are
what a model absorbs. On the strict reading the merged wheel would be NC regardless of the filter,
so v0.2.0 shipped it that way and users chose. **Amir Zeldes settled it by email (2026-08-17): the
annotations are Georgetown's, under CC BY; the NC belongs only to the individual documents.** So
the filter is load-bearing, and the merged wheel is CC BY-SA 4.0 — SA from EWT and from the two
ShareAlike kept genres, not from GUM. What remains is CC BY **attribution**: cite GUM, link
<https://gucorpling.org/gum/>, credit the annotators, cite the text sources (`NOTICE.md`, and
`meta.json`'s `sources` inside the wheel). The two wheels now differ only in training data and
attribution surface, and en_gum is the stronger parser. ⚠ `NC_GENRES` in `build_en_ewt_gum.sh` is
now a LICENCE BOUNDARY, not a conservative default. **GUM's NC genres are FIVE** (essay, fiction,
letter, podcast, whow), not two.

**The relabel is free if the ORDER is right.** The original development corpus was EWT+GUM
concatenated, so `caches/relabel_cache*.jsonl` already holds every GUM decision — but the keys are
POSITIONAL (`path|sentence_index|token_id`). Relabel the unfiltered EWT-first concatenation, filter
the NC genres LAST: 34,461 targets at **zero** model calls. Filtering first shifts every later index
and throws away half the cache. `build_en_ewt_gum.sh` step 2 refuses to run if the dry run bills
anything. The Perseus XPOS trap does NOT apply — GUM's 46 tags are a strict subset of EWT's 49.

**`Reported` gold keys differently and that is what makes IT cheap** — `sent_id|comp_id`, not
positional — so `base_lang()` pointing en_gum at `caches/relabel_cache_reported_en.jsonl` makes the EWT
half free: of 565 residue decisions 394 hit, and all 171 misses were GUM. See `Reported` in
`docs/sud-misc-layer.md`; an arm name is not a language, and the two places that confused them
both failed silently.

**Apples-to-apples on the EWT-only test** (identical gold — the EWT half of the en_gum test is
byte-identical to it, 2077/2077 blocks; RAW end-to-end, not gold-preproc, so these are ~1.7 LAS
below the released figures in `metrics/release/metrics_release_en*.json` and are a comparison, not a headline):
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

## Indonesian

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

## Korean

**Korean has its own file now: `docs/korean.md`.** The eojeol tokenisation, the 34.5 % of test
tokens that are unseen strings, the analyser channel that reaches their stems, and the word-order
augmenter all live there. What stays here is the one thing that reads as a metric and is not:

The eojeol arm reads the original `assets_ko/SUD_Korean-GSD`, whose FEATS is 4.7 % populated — so its
`morph_acc` 95.36 is ~the base rate for predicting empty and says nothing (`morph_micro_r` 0.15 is
the honest figure). POS 83.05 / lemma 78.30 are real.

