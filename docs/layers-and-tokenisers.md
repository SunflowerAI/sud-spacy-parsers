# The layer stack and the tokenisers

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

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
- **SUD MISC layer** (`train_sud.sh`, `package_sud.sh`) — see `docs/sud-misc-layer.md`.

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
| **sa** | `sa_tokenizer.py` + CSLiser front end | see `docs/sanskrit.md` |
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

