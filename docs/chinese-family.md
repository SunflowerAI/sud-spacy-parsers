# Classical Chinese, Chinese, Cantonese

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

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

## What `clause_parser` is still for on lzh — and the arm that removes it

Since punctuation was restored, `clause_parser` has been doing only ONE of its three jobs on lzh.
Measured, not inferred: fed three punctuated 論語 sentences, `training_lzh_trad_sud_xw` tags all six
marks `PUNCT` with the right Kyoto 記号 XPOS and attaches every one as `punct` — so stripping marks
and normalising their morphology are both already redundant there, which is why the wheel ships
`--keep-marks`. But it returns **one sentence for three, with zero self-headed roots**: the whole
24-token string comes back as one connected tree. Segmentation is the only remaining job.

The cause is one line. `config_lzh.cfg` reads the corpus through `spacy.Corpus.v1` with
`gold_preproc = true`, so every training example is a single sentence and the parser never sees a
boundary to learn — while the corpora were ALREADY converted at 10 sentences per document, which
`gold_preproc` was throwing away. Its own log reads `sents_f 1.0000`, standing hazard 4 exactly.

`config_lzh_seg.cfg` changes only the reader, to `sud.GoldTokCorpus.v1`. On the test set, scored on
multi-sentence documents — the input users actually give it:

    released training_lzh_trad_sud_xw    LAS 61.05   UAS 67.04   SENTS_F  0.00   TAG 92.43
    training_lzh_seg                     LAS 74.48   UAS 80.14   SENTS_F 79.57   TAG 92.35
                                             +13.43                 +79.57

⚠ **It is a trade, not a free win.** Under `gold_preproc` — one sentence handed over, boundaries
free — the released arm is still AHEAD (LAS 76.57 vs 74.81), and `SENTS_F` there reads 87.75 rather
than 100 for the new arm, i.e. it sometimes splits a sentence that should not be split. The new arm
is better on realistic input and slightly worse when the boundaries are given.

⚠ **`seg` is a BASE recipe, so the whole stack is retrained on it**, not reused —
`config_lzh_seg_{morph,xposwarm,sud}.cfg`, in that order, then `graft_xpos_tagger.py` so the tagger
sits behind the morphologiser (`package_sud.sh` refuses an arm where it does not).
`scripts/queue_lzh_stack.sh` is the driver. The morph layer lands on `MORPH_ACC 91.12`, identical to
the released chain's, which is the freeze recipe behaving as designed — a dedicated encoder reading
no parse cannot be reached by a base change. The SUD MISC pipes are the ones that must be
RE-MEASURED rather than assumed to carry over: every one of them reads the base's own predictions
(standing hazard 5).

## Restoring punctuation to Kyoto, and relating the units (lzh)

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
  Released figures: zh LAS 69.01 / UAS 73.82 / comp:obl F 31.09, lzh 77.20 / 82.92 / 66.47
  (gold-preproc, `metrics_release_*.json`).

  ⚠ **The traditional-only zh arm shipped with NO sentencising at all, and gold-preproc hid it
  completely.** `zh_sud_gsd-0.2.0` returned ONE sentence per document however many full stops the
  input had (`metrics_release_zh_raw.json`: `sents_p/r/f = 0.0`), silently undoing the whole seg
  layer for this language; lzh was unaffected, since `clause_parser` splits it by rule. Cause:
  `retrain_zh_trad.sh` trained the base from `configs/config_zh.cfg` — the PRE-seg recipe
  (`spacy.Corpus.v1`, `gold_preproc = true`, `sents_f = 0.0`) — and its `for layer in seg morph
  lemma` loop then died on a missing `configs/config_zh_seg.cfg`, because `make_seg_config.py` takes
  a config PATH and was handed the string `zh`. The failure was swallowed by `|| true` (the 186-byte
  `train_zh_trad_seg.log` is the whole record of it), and morph/lemma were run two days later
  against the unfixed base, which they freeze.

  **`seg` is a BASE recipe, not a stackable frozen layer** — that is the lesson worth keeping. No
  layer above a `gold_preproc`-trained parser can teach it to START a sentence, and nothing in the
  ordinary metrics says so: under gold-preproc every dev example is already one sentence, so
  `SENTS_F` reads a reassuring 100.00 in the training log and 1.0000 in `metrics_release_zh.json`.
  Only the raw eval, or two sentences typed at the model, tells the truth.

  Retrained through `configs/config_zh_seg.cfg` (`sud.GoldTokCorpus.v1`, `sents_f = 0.05`), morph and
  lemma rebuilt on it by the freeze recipe (tok2vec/tagger/parser byte-identical up the chain).
  **Raw end-to-end on the traditional test, same fixed segmenter** (`models/zh_seg_jbdec_trad`, so
  only the parser moves — the "never quote zh raw LAS from one segmenter run" caveat does not
  apply): SENT F **0.00 → 98.50**, LAS 58.41 → **62.61**, UAS 62.47 → **66.98**, TAG 87.51 → 87.58.
  Gold-preproc is unmoved, as it must be: LAS 68.86 → 69.01, UAS 73.29 → 73.82, pos/lemma/morph
  identical to the decimal. The old base is kept as `training_zh_trad_preseg/`, and the driver now
  generates the seg config with the right argument, drops the `|| true`, and REFUSES to hand off to
  packaging unless a two-sentence input comes back as two sentences.

  **Re-released (clobbered) at 0.2.0 on 2026-08-11.** Verified the standing three ways: the
  published asset's sha256 matches the built wheel, every weight file hashed OUT OF THE DOWNLOADED
  wheel matches the arm it should (`tok2vec`/`tagger`/`parser` → `training_zh_trad`, `morphologizer`
  → `_morph`, `lemmatizer` → `_lemma`, so the freeze chain is intact), and a clean `--target`
  install segments. The tokeniser did NOT move: `token_acc` 0.9694 / strict token F 0.9242,
  re-measured, identical to the published figures. ⚠ Same version, so `pip install -U` will NOT
  replace an older copy — `--force-reinstall` will. ⚠ `spacy evaluate` prints NO `TOK` row for this
  arm, because token scoring comes from `nlp.tokenizer.score` and `ZhTradTokenizer` defines none;
  the raw file's `token_*` keys are merged in from `Scorer.score_tokenization` over the same model
  and corpus, and the gold-preproc file has none (they would be 1.0 by construction). The superseded both-scripts arms trained on the two
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

