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
  (gold-preproc, `metrics/release/*.json`).

  ⚠ **The traditional-only zh arm shipped with NO sentencising at all, and gold-preproc hid it
  completely.** `zh_sud_gsd-0.2.0` returned ONE sentence per document however many full stops the
  input had (`metrics/release/metrics_release_zh_raw.json`: `sents_p/r/f = 0.0`), silently undoing the whole seg
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
  `SENTS_F` reads a reassuring 100.00 in the training log and 1.0000 in `metrics/release/metrics_release_zh.json`.
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

  ⚠ **jieba is SIMPLIFIED and the arm is traditional, so the channel has to be made traditional
  somewhere.** Asking jieba about the traditional text directly scores boundary F 0.8931 against
  0.9237 — the whole gap is vocabulary, not the language. There are two places to fix it, and this
  repo has now built both:

      convert the TEXT       `--jieba-t2s`, superseded 2026-08-22   F 0.9236
      convert the DICTIONARY `--jieba-dict`, what 0.2.0 now ships   F 0.9237

  They score the same, and the second is the one that answers about the string being segmented.
  `t2s` is many-to-one, so converting the text hands jieba 干 for 乾, 幹 and 干 alike and every
  distinction the traditional orthography draws is invisible to the lookup; converting the
  dictionary (`build_jieba_trad_dict.py`, OpenCC **`s2tw`** — the same conversion `ZhTradTokenizer`
  applies to incoming simplified input, and GSD's own orthography, where plain `s2t` would write
  爲什麼 for 為什麼) leaves the text alone. ⚠ **The dictionary is only half of jieba.** Unknown runs
  go to `finalseg`, an HMM with per-CHARACTER emission probabilities estimated on simplified text,
  and leaving it on traditional characters costs 0.9237 → 0.9203 — as much as the dictionary
  itself is worth. It is asked about the `t2s` rendering and handed back the original characters,
  which is sound because the codes are per character and `t2s` preserves length (500/500 test
  sentences), with a length check per run falling back to the raw text.

  End to end the two regimes are a **wash**: ten training runs each, mean strict token F 0.9209
  (dictionary) against 0.9203 (text), sd ~0.003, 6/10 seeds favouring the dictionary — and the
  dev-selected run that shipped scores **0.9196**, below the 0.9242 the superseded arm was quoted
  at, which is a redraw within a 0.9167–0.9268 spread rather than a regression. It was swapped in
  for reading the script the model works in, not for its score — and, unlike the text
  conversion, it also holds up on Hong Kong-variant traditional input (F 0.9240 against 0.9233),
  which an s2tw dictionary was the obvious risk to.

  **`jieba_dict` is written into the segmenter's `vocab.json` and the dictionary itself is copied in
  beside the weights**, read back by `char_seg_tokenizer.load_segmenter` and `eval_zh_seg.py`, which
  REFUSE to load a model whose dictionary is missing rather than fall back on jieba's simplified one
  — a channel asked a different question at inference than at training is the `reads_spaces` trap,
  and this variant of it would come back on the wrong VOCABULARY with nothing raising. It costs the
  wheel nothing: `vendor_jieba.py` sees the model carrying its own dictionary and vendors jieba
  without `dict.txt`, 5.06 MB in against 5.07 MB out (wheel 15 088 470 → 15 111 219 bytes).
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


## Sentence joining: quoted spans and commas (lzh)

The lzh arm has **no `senter` pipe and no `clause_parser`** — the parser IS the senter, because
`sud.GoldTokCorpus.v1` feeds it multi-sentence documents and spaCy derives `doc.sents` from the
tree. So it inherits the Kyoto treebank's segmentation, and Kyoto segments at **句讀 units**: 5 944
of the 59 215 punctuation-restored training blocks have UNBALANCED 「」/『』 (the quote opens in one
gold sentence and closes in a later one), and **31.2 % of its blocks END at a pause mark**.

    block 3: 子曰：「不仁者不可以久處約，
    block 4: 不可以長處樂。
    block 5: 仁者安仁，
    block 6: 知者利仁。」

`scripts/sent_join.py` (`sent_join`, added by `add_sent_join.py`, LAST in the pipeline) imposes the
reading convention instead, under two switches: `quote_spans` — a balanced quoted span holds no
boundary and the closing mark stays inside it — and `pause_join` — no sentence ends at a pause mark
of any kind (，、；：,;:).

### What it attaches, and why it is not a guess

**Inside a quoted span, the speech verb gets ONE object and every further clause is `parataxis`.**
A verb has one object slot; the second, third and fourth clauses of a reported speech are juxtaposed
utterances, not additional objects of 曰. The first clause keeps whatever the parser gave it
(`comp:obj` of 曰, untouched), and every clause `sent_join` frees inside the span is attached to
THAT clause as `parataxis`:

    子曰「學而時習之，不亦說乎。有朋自遠方來，不亦樂乎。」
       學 comp:obj-> 曰      說 parataxis-> 學      有 parataxis-> 學      樂 parataxis-> 有

⚠ **AND THE FAN-OUT IS CAPPED AT WHAT GOLD ATTESTS (`max_same_dep`, default 2).** Kyoto's
attachment arities are strikingly tight and identical for both relations involved:

    children on one governor        1 child      2 children    3+
    comp:obj  (79 898 governors)    96.88 %       3.12 %       NEVER
    parataxis ( 9 527 governors)    98.08 %       1.92 %       NEVER

and `parataxis` is overwhelmingly FLAT on the first clause rather than chained (9 600 against 110).
So `sent_join` attaches flat until the anchor holds two, then hangs the next clause off the previous
one — a five-clause quote stays inside the attested space instead of giving one token four children.

⚠ **THE EVIDENCE FOR THE *ATTACHMENT POINT* IS THIN AND MUST NOT BE OVERSTATED.** It is tempting to
cite "1 743 of Kyoto's 1 755 balanced quoted spans have exactly ONE token whose head lies outside
the span". True, and nearly vacuous: almost all of those spans hold a SINGLE clause, and a
one-clause quote trivially has one attachment point. The statistic that actually bears on this is
spans containing an internal sentence-final mark, and there are **eighteen** — 8 attach the second
clause outside the span, 7 hold the block root, 3 attach inside. The broader cell it belongs to (any
clause after a full stop whose previous unit head is a VERB, n=155) is genuinely undecided
(`parataxis` 34 %, `comp:obj` 32 %, `conj:coord` 19 %) and `build_lzh_sent_joins.py` rejects it
outright. **The relation here is therefore settled on the analysis, not on the counts** — the counts
are too thin to settle anything, and the one thing they do settle is the arity ceiling above.
The 91.2 %-`parataxis`-for-曰 figure below is a different configuration again:
`cross_unit_rules.py` relating two 句讀 units ACROSS a block boundary.

⚠ A trap worth keeping: an earlier version attached every freed clause to the speech verb with the
span's own relation, giving 曰 **four** `comp:obj` children on a four-clause quote. **No headline
metric moved** — going from that to the capped `parataxis` analysis is worth ~0.03 UAS. It was
caught by reading the emitted tree, and by asking what arities the treebank actually contains.

**Everywhere else the relation is conditioned on the UPOS of the previous unit's head**, harvested
by `scripts/build_lzh_sent_joins.py`:

    after a PAUSE mark   VERB -> comp:obj  69% of 6545     NOUN  -> conj:coord 87% of 1092
                         PROPN-> conj:coord 95% of  552
    after a FINAL mark   PROPN-> conj:coord 98% of  266    NOUN  -> conj:coord 90% of   96
                         SYM  -> comp:obj  100% of   58    NUM   -> conj:coord 95% of   37

⚠ **TWO BARS, BECAUSE ONE CANNOT SERVE BOTH ENDS OF THIS DISTRIBUTION**: ≥ 80 % on ≥ 20 examples, OR
≥ 50 % on ≥ 500. `cross_unit_rules.py` uses ≥ 90 % on ≥ 20, which is right *there* because a wrong
rule writes false structure into the TRAINING data; here a rule only labels an arc the parser did
not produce at all, so a 69 %-accurate label on the 6 545-example `pause`+VERB cell plainly beats a
default that is right 17 % of the time. That argument does not licence a 50 %-of-48 cell, hence the
second bar. Four cells clear neither and fall through to the `conj:coord` default rather than
memorising a plurality: `pause`+AUX (`comp:obj`, 47 % of 207), `pause`+PART (56 % of 48),
`pause`+NUM (50 % of 48), and — worth noting — **`final`+VERB, where the best relation is
`parataxis` at only 34 % of 155**. That cell is genuinely undecided, and a majority vote there would
have shipped a rule the data does not support.

### The cost, and why its sign flips between harnesses

Measured on the shipped 0.2.0 wheel over the traditional test set, one prediction pass scored four
ways (`Scorer.score_deps` over hand-built gold-token docs, so read the rows against each other and
not against the released 77.20):

    arm            10-sentence docs                --gold-preproc (the released harness)
                    UAS     LAS   SENTS_F           UAS     LAS   SENTS_F
    off           80.51   74.93     80.41         81.98   76.46     90.79
    quotes only   79.57   73.97     74.68         81.99   76.46     91.07
    pauses only   77.70   72.00     56.42         81.94   76.30   **95.12**
    both          77.34   71.66     55.08         81.95   76.30   **95.18**

⚠ **THE TWO HARNESSES DISAGREE ABOUT `pause_join`, AND BOTH ARE RIGHT.** Under `--gold-preproc`
every document IS one gold sentence, so the only error available is OVER-splitting, and refusing to
break at a comma buys **+4.39 SENTS_F for −0.16 LAS**. Over 10-sentence documents the model must
also FIND the boundaries between sentences — and 31.2 % of Kyoto's boundaries sit at a pause mark,
so the same rule throws away a quarter of them. The rule is a reading convention imposed at
inference, not treebank fidelity, and no single number describes it: **which way it scores is
decided by whether the harness makes the model find sentence boundaries at all.**

`LZH_SENT_JOIN=0` leaves the pipe out entirely; `LZH_SENT_JOIN_ARGS=--no-pause-join` keeps the quote
rule alone.

### Three implementation points that are load-bearing

- **`token.is_sent_start` is NOT writable on a parsed doc** (spaCy raises E043), and a sentence
  begins exactly where a token heads itself. So merging means giving the second sentence's ROOT a
  head in the first. That is done IN PLACE (`token.head = …`), which is why this component — unlike
  `clause_parser` — never rebuilds the `Doc`, and the standing "anything that rebuilds a Doc owns
  carrying EVERY annotation" trap does not arise: lemma, MORPH, NORM and every extension survive by
  construction.
- **Only balanced spans merge**, so a stray 「 cannot collapse a text into one document-long
  sentence; `max_span` is a second guard on the same failure. Everything read off the tree —
  sentence roots, each span's external governor — is computed BEFORE the first merge, because a
  merge gives a root an external head and would otherwise make it look like the span's own first
  attachment on the next iteration.
- It composes with `clause_parser` rather than duplicating its sentencer: run last, it works on
  whatever tree it is handed, so the `LZH_CLAUSE_PARSER=1` arm gets the same behaviour.

## Why the lzh tagger reaches for PROPN — and what fixes it

**On the treebank's own test set it does not.** With gold tokens, PROPN comes out at 0.942× its gold
count, P 93.79 / R 88.30 — *under*-predicted. The problem is entirely off-treebank, and it has one
cause with two amplifiers.

**The cause is a corpus prior the encoder cannot override.** Kyoto is dominated by histories, so
PROPN is 8.5 % of its TOKENS but **37.5 % of its TYPES**, and 49.4 % of its hapax types:

    majority-UPOS == PROPN, among train types, by train frequency
      freq   1     3027 types   49.39%          freq  6-20    1562   29.26%
      freq   2     1270 types   46.06%          freq 21-100   1075   18.14%
      freq   3-5   1459 types   40.92%          freq 101+      636    8.18%

So a form the morphologiser cannot identify has a learned prior of roughly one-in-two PROPN — and
nothing to override it with. Its encoder is its own `HashEmbedCNN`, width 64, depth 3, **2 000 hash
rows for 9 029 training types**, `attrs = NORM/PREFIX/SUFFIX/SHAPE`, `pretrained_vectors = null`.
For a single Han character all four attrs are the same character, so the channel is NORM alone, and
an unfamiliar glyph lands on a colliding row.

**Amplifier 1 — every multi-character token.** 73.0 % of Kyoto's multi-character tokens are PROPN
(孔子 匈奴 五十). The trained character segmenter exists precisely to produce such tokens, so every
merge it makes — including every spurious one — lands on a 73 % PROPN prior. On 54 018 tokens of
kanripo the shipped wheel calls a multi-character token PROPN **70.9 %** of the time when the form
is treebank-unseen, against 3.55 % for a seen single character.

**Amplifier 2 — orthographic variants, which is where the token mass actually is.** Measured on
the test set, the failure population is sharp:

    slice (n of 34 233)                        UPOS acc   PROPN P   PROPN R
    ALL                                          93.13%    93.79%    88.30%
    form UNSEEN in train           (   392)      65.31%    80.90%    83.24%
    has a char absent from the treebank ( 195)   51.79%    39.13%    47.37%

⚠ **And those characters are mostly NOT rare words — they are ordinary ones in a different glyph.**
Of 42 M characters of kanripo, 3.38 % are absent from the treebank; of that mass **80 % sits in the
311 types that occur MORE than 500 times in kanripo**: 无 (146 685, = 無), 隂 (97 843, = 陰),
徳 (84 213, = 德), 逺 (27 903, = 遠). Each is a high-frequency function or content word that the
wheel tags PROPN because it has never seen that glyph.

⚠ **This is the type/token split NEGATIVE-RESULTS.md's kanripo-vector entry records, read the other
way round.** That entry found treebank-unseen TYPES have a median kanripo frequency of 4 and
concluded static vectors are informative only where the parser already copes. Both are true: by
types the unseen population is rare, by TOKENS it is a few hundred common variants. Which statistic
is the right one depends on the task — for parsing the treebank test set (1.15 % unseen tokens) it
was types; for tagging running text it is tokens. **Split by frequency, and then ask which weighting
the task uses.**

### The fix: an 異體字 map, applied at the TOKENISER

`scripts/build_lzh_variant_norm.py` builds the table; `sud.CharSegTokenizer.v1` applies it to the
input **before segmentation**, so 无 reaches every encoder as 無. Two sources, symbolic first:

    source                                     types   % of the out-of-treebank character mass
    every OpenCC character map in the venv         —                     25.8%
    Unihan variant fields, <=2 hops             1088                     47.9%
    + SikuBERT nearest neighbour, cos >= 0.55     90                    +14.8%   (union 62.7%)

**Unihan does two jobs.** `assets_unihan/Unihan_Variants.txt` was already in the tree for the radical
channel; its `kSemanticVariant`/`kZVariant`/`kTraditionalVariant`/`kSimplifiedVariant` fields resolve
half the mass with no model at all. And it is the **validator** for the other half: on the 406
characters where Unihan has a treebank-seen answer, SikuBERT's nearest neighbour picks the same
character **94.6 %** of the time — a precision estimate for the residue route that costs no hand
labelling. Where Unihan stops is the Japanese-style and Ming-edition forms kanripo is full of
(徳 逺 乗 懐 眀 曽 舎 聴 従 get no link at all), and a handful of its links are wrong (亲→榛, 栢→孛,
拼→秉, against SikuBERT's 親/柏/拚). Unihan is listed first anyway, so it wins ties; both members of
a variant set are treebank-seen, so the encoder gets a valid row either way.

⚠ **IT IS APPLIED AT ORTH, NOT AT NORM, AND THAT IS THE WHOLE DIFFERENCE.** `norm_` is one of the
encoder's four channels; PREFIX, SUFFIX and SHAPE are computed from ORTH and would still carry the
variant glyph — and for a single Han character all four are the same character, so a NORM-only fix
changes a quarter of the signal. Measured on 54 018 tokens of kanripo:

    baseline                          PROPN 5.52%      无 -> VERB:141 PROPN:98 NOUN:59
    a NORM-setting pipe               PROPN 5.48%      无 -> VERB:130 NOUN:75  PROPN:69
    the glyph rewritten (what ships)  PROPN 5.17%      無 -> VERB:364 ADV:47   PROPN:0

The NORM route moves several characters onto the right category outright (乗 NOUN→VERB, 别 NOUN→VERB,
従 NOUN→VERB, 曽 NOUN→ADV) but leaves most of the gain behind; rewriting the glyph takes 无's PROPN
count to **zero**. This is the `fa` recipe — normalise the orthography IN rather than train on every
spelling — and it also fixes the half a NORM pipe cannot reach at all: **the segmenter is itself a
character model trained on treebank orthography**, so a variant glyph is as unfamiliar to it as to
the tagger, and normalising after segmentation would leave that untouched.

**What it is worth, measured on the shipped wheel with the map bundled.** On the same kanripo
sample, restricted to the tokens whose SOURCE glyphs hold a character the treebank never showed —
the population the whole problem lives in:

    released 0.2.0 wheel        1680 such tokens   PROPN among them  17.56%
    + variant normalisation     1684 such tokens   PROPN among them   9.14%
                                                   (Kyoto's own gold PROPN share is 8.52%)

The PROPN rate on that population is **almost halved and lands on the treebank's own base rate**.
On the treebank's own test set, where those characters are genuinely often names, it is a gain in
the failure slice and a no-op everywhere else:

    slice (n of 34 233)                        UPOS acc          PROPN R
    ALL                                    93.13 -> 93.16    88.30 -> 88.37
    form UNSEEN in train           (   392) 65.31 -> 67.35    83.24 -> 84.39
    has a char absent from the treebank (195) 51.79 -> **55.90**  47.37 -> **57.89**

Four things make it safe to ship without a retrain:

- **No retrain is involved.** `gold_preproc` + `sud.GoldTokCorpus.v1` make the parser
  segmenter-agnostic, so what the tokeniser hands it may be changed freely. `bundle_lzh_variants.py
  --verify` asserts every file outside `tokenizer/` is byte-identical to the source arm.
- **It must be a no-op on treebank orthography**, or the released metrics stop describing the wheel.
  `--verify` reproduces the token stream AND the full-pipeline parse digest over 200 test texts
  before it will write anything.
- **The caller's own spelling is never discarded.** The map is strictly 1:1 by character — refused
  otherwise, because every `token.idx` depends on it — so `doc._.lzh_src_text` holds the input and
  `token._.lzh_src` slices it: `token.text` reads 無 where `token._.lzh_src` reads 无.
- **A model that declares a map it cannot find refuses to load.** `tokenizer/meta.json` records the
  regime and `from_disk` raises on a missing `variants.json`, rather than quietly segmenting the
  unnormalised text (CLAUDE.md standing hazards 8 and 11).

⚠ A trap the first version of the bundler hit, and worth keeping: it verified and wrote from the
SAME `nlp` object, and running text through a pipeline interns strings — so `vocab/strings.json`
came out different from the source arm for reasons that had nothing to do with the change. The fix
is to write from a freshly loaded model that has processed nothing. **Comparing every file rather
than a chosen list of weights is what caught it.**

### SikuBERT as a vector channel for the tagger

`scripts/build_lzh_sikubert_vectors.py` distils `SIKU-BERT/sikubert` (BERT-base, Apache-2.0,
pretrained on 四庫全書) into an ordinary static table: contextual last-hidden states averaged per
character over 4 M characters of **leak-free** kanripo, PCA 768 → 96 (73.4 % of variance), rows for
every character AND every treebank type (a multi-character type is the mean of its characters).
11 444 keys, 9 MB. `pretrained_vectors = true` then makes the morphologiser's `MultiHashEmbed`
**concatenate** a `StaticVectors` projection with its NORM/PREFIX/SUFFIX/SHAPE hash channels — the
literal "tok2vec ⊕ PCA'd SikuBERT" — and `scripts/train_lzh_sikuvec.sh` runs it against a **shuffled
control**: the same rows, the same shapes, the same parameter count, the type-to-row correspondence
destroyed. Three seeds each. Test set, `--gold-preproc`:

    slice                        control (mean±sd)   vectors (mean±sd)        Δ    per-seed Δ
    ALL                              93.22 ±0.12         93.47 ±0.17     +0.25   +0.48 +0.25 -0.00
    form UNSEEN in train             66.92 ±1.66         73.98 ±2.09    **+7.06**  +8.93 +6.38 +5.87
    multi-character token            92.49 ±0.73         93.66 ±0.70     +1.17   +2.46 +0.70 +0.35
    char absent from the treebank    53.68 ±1.80         67.01 ±2.14   **+13.33** +13.33 +10.77 +15.90
      — PROPN precision there        39.78 ±0.98         56.28 ±2.85   **+16.50** +19.26 +14.86 +15.38

**Read the first row and the fourth row as different answers, because they are.** The aggregate
+0.25 is roughly one and a half times its own spread and one seed gives exactly nothing — on the
headline this is a weak result, and seed 0's +0.48 alone would have been the kanripo-vector trap
repeated verbatim. The failure population is a different story: **the sign never flips and the
margin never drops below +5.87**, against a seed spread of ~2. A channel can be worth 13 points
where it is needed and 0.25 points on average, and the average is not the interesting number when
0.57 % of the test set is the population you built it for.

Three further things the arm establishes:

- **The control behaves as a control must.** 93.22 mean against the no-vector arm's 93.13 — the
  extra projection and the widened Maxout buy 0.09, which is what makes the +0.25 readable at all.
  Without it the arm would have looked like +0.34 over "no vectors" and that number would have been
  meaningless.
- ⚠ **THE CONCATENATION BEATS EITHER CHANNEL ALONE, AND THE MULTI-CHARACTER ROW IS THE PROOF.** A
  linear probe on frozen SikuBERT states is far WORSE than the shipped arm there (77.11 vs 93.43):
  mean-pooling a two-character name's subtokens loses the "this is one token" signal a NORM row
  carries. Concatenated, the same vectors are +1.17 over the control on that slice. The hash rows
  supply token identity, the vectors supply lexical knowledge, and neither alone has both — which
  is the argument for concatenating rather than replacing.
- **The freeze recipe held exactly.** `tok2vec`, `tagger` and `parser` come out byte-identical to
  `training_lzh_seg/model-best` in every arm, so TAG/UAS/LAS are unchanged to two decimals and
  spaCy's W113 ("source vectors are not identical to current pipeline vectors") is confirmed
  harmless here: a frozen component whose own embed has `include_static_vectors = false` cannot see
  the table at all.

⚠ **Read every SikuBERT figure here as an UPPER BOUND.** It is pretrained on 四庫全書 and Kyoto is
drawn from the same tradition, so the test text is very likely inside its pretraining corpus. That
is the kanripo situation again — "not label leakage, but fitted to the very text it is scored on" —
and unlike kanripo it cannot be fixed, because the pretraining run is someone else's.

### "Freeze the SikuBERT part and train the rest" — it already is, and here is what that buys

⚠ **THERE IS NOTHING LEFT TO FREEZE.** spaCy never updates `vocab.vectors`: the table is
byte-identical across all three seeds and to the source package it was initialised from. The only
trainable parameter that touches SikuBERT at all is `static_vectors`, the **96 × 64 = 6 144**
linear projection into the encoder's width — out of 509 661 trainable parameters in the
morphologiser. Freezing that too would leave a RANDOM 96→64 compression of every row, which
destroys a third of the dimensions for no reason; the concatenation is already
"frozen knowledge + trained residual".

**And the division of labour the idea is after is already happening — measurably.** Ablating the
channel at inference on the trained arm (every row zeroed, shapes and lookups intact, no retrain):

    slice                          vectors live   vectors ZEROED   held by the channel
    ALL                                   93.47            87.21             +6.26
    form UNSEEN in train                  73.98            62.67            +11.31
    char absent from the treebank         67.01            49.91            +17.09

⚠ **BUT RELIANCE IS NOT GAIN, AND THE CONTROL IS WHAT SEPARATES THEM.** The ablated arm falls to
87.21, far BELOW the shuffled control's 93.22 — a network trained WITH the channel offloads onto it
and allocates its hash capacity elsewhere, so removing it afterwards is catastrophic. A network
trained WITHOUT it reaches 93.22 on its own. The genuinely new part is the difference between those
two, +0.25 aggregate — and +13.33 on the population where the hash channels demonstrably cannot
substitute. **A 6.26-point ablation drop and a +0.25-point net gain are both true of the same arm.**
Read an ablation as "what the model came to depend on", never as "what the channel is worth".

### The parser, and why this test set cannot answer the question

**The parser does not read the vectors at all.** `pretrained_vectors = true` was set on the
morphologiser's OWN `HashEmbedCNN`; the shared `tok2vec` the parser listens to still has
`include_static_vectors = False`. The identical UAS/LAS across every arm above is a consequence of
that plus the freeze recipe, and is not evidence about the channel in either direction.

⚠ **AND THE PRE-FLIGHT ARITHMETIC SAYS THIS TEST SET COULD NOT MEASURE IT ANYWAY.** A channel moves
an aggregate metric only over the population it reaches, and a static vector reaches a parser
decision that the FORM does not already settle — i.e. rare and unseen types:

    test tokens (34 233)                       n      share
    form UNSEEN in train                     392      1.15%
    form seen <= 2 times                     751      2.19%
    holds a treebank-absent character        195      0.57%

    implied ceiling on aggregate LAS, if the channel were worth   +5     +10     +15  on the slice
      over unseen forms                                          0.06    0.11    0.17
      over forms seen twice or fewer                             0.11    0.22    0.33

against a seed spread of ~0.5 LAS on this arm family. **Even a spectacular per-token gain is
invisible**, which retro-explains the kanripo vector arm's +0.04 LAS mean far better than
"the vectors are empty where they are needed" does on its own: they were also aimed at 1 % of the
decisions. Running a parser-side arm on Kyoto would produce an unreadable number, and it is worth
knowing that before spending the base retrain (`seg` is a BASE recipe, so the whole layer stack
would follow it).

**Route 2 of that pair has now been RUN, and it fails** — see NEGATIVE-RESULTS.md, "SikuBERT
vectors as a PARSER channel, injected above the frozen encoder". `sud.StaticVecChannel.v1` gives the
parser `concatenate(Tok2VecListener(96), StaticVecChannel(96))` above a frozen, byte-identical
encoder — exactly the injection point that rescued the conditioned XPOS tagger. Three seeds against
the shuffled table: **every slice's mean delta is smaller than the spread of its own per-seed
deltas**, and the only consistent-sign slice is a small LOSS. The `unseen` row reads +1.63 and its
4.62-point seed swing is **eighteen tokens**.

⚠ **THE LESSON IS THAT IT IS THE TASK, NOT THE WIRING.** Same table, same repo, same three seeds:
+13.33 to the morphologiser on its failure slice with the sign never flipping, and nothing to the
parser at any frequency. A static type-level vector says what kind of lexeme this is — **for the
tagger that IS the output**, so it speaks straight to the decision; for the parser it is only a
mediator, and the category that mediates attachment is already in the encoder. Two independent
vector families (kanripo floret, SikuBERT) and two injection points (embed, above-encoder) now
agree: **this parser is not lexicon-limited.**

**And handing the parser the MORPHOLOGISER'S TRAINED ENCODER instead of the raw rows also fails**
(NEGATIVE-RESULTS.md, "The morphologiser's TRAINED encoder as a parser channel"): -0.54 LAS against
a donor trained on the shuffled table, same sign on all three seeds. The diagnosis is the one worth
carrying away, because two plausible explanations died first (the representation has NOT collapsed —
the real donor is +1.27 better at DEPREL; the parser has NOT been crowded out — it relies on the
better channel LESS). Conditioning is what explains it:

    DEPREL from the parser's own 96-d encoder ALONE      76.00 %
      + the real-vector donor      76.88 %   increment  +0.87
      + the shuffled-vector donor  76.78 %   increment  +0.77

The parser's own representation already beats either donor outright, and the better donor's whole
+1.27 advantage collapses to **+0.10** once conditioned on it. **A plain probe measures information
present; only a conditional probe measures information additional.**

So the remaining route is the other one, and it is an annotation problem rather than a modelling
one: **measure on text the metric can see it in.** The channel's value is out-of-domain, where the
unseen-type rate is not 1.15 % — on the kanripo sample it is 3.1 %, on a Ming edition or a
Japanese-hosted transcription higher again. lzh has raw gold sets for TOKENISATION
(`assets_lzh/*_gold.txt`) but none with gold dependencies, so nothing parser-side can be resolved
until some exist.

### What this means for shipping, and for the parser

**The 9 MB table is the cheap half and it is already justified**, on a 15 MB wheel, for +13.3 on the
population the arm actually fails on. **A fine-tuned SikuBERT tagger is the expensive half and is
untested**: it would very likely beat 93.47, and it costs a ~400 MB wheel plus a transformer forward
pass per document against a project whose premise is small CPU pipelines.

**As a parser input, a predicted POS label is still the wrong shape.** NEGATIVE-RESULTS.md settles
the shipped configuration: `spacy.Tagger.v2` is a linear softmax on the listener output and the
parser's first operation on the same output is also linear, so an XPOS the tagger predicts is
linearly present in the parser's own input by construction — 0.1475 bits of genuinely available
information bought −0.25 LAS. UPOS from the morphologiser escapes that particular argument (its
encoder is a separate `HashEmbedCNN`, not a listener) but is still a function of the same
characters. The SikuBERT table is the only channel here that passes both pre-flight checks —
genuinely new information, not decodable from an encoder-sharing head — and if it is being read at
all, the parser should read the 96-d vector, not a 15-way label hashed back down. Unmeasured; what
would decide it is exactly the split above, run on parse decisions rather than UPOS.


## What the Heart Sūtra found, and the two caps it forced (lzh 0.3.0)

Running the shipped arm on CBETA's 般若波羅蜜多心經 (T08n0251) turned up two defects that **no
treebank metric could have shown**, because Kyoto's blocks are single 句讀 units and too short to
contain either.

**1. A repeated opening quotation mark is a CONTINUATION, not a nesting level.** The sūtra has
「x5 against 」x2 — in every edition including CBETA's own, because a quotation running over several
paragraphs takes a fresh 「 at the start of each and one 」 at the end. English does the same with
continued speech. `balanced_spans` counted each 「 as a new level, saw an unclosed opener, and with
`open_quotes` swallowed **333 of the text's 371 tokens into one sentence**. A repeated opener of the
SAME KIND is now skipped rather than pushed; genuine same-kind nesting is not the Chinese convention
(an inner quote takes 『』), so this costs nothing.

⚠ It is worth being clear about what was NOT wrong: the text. The first instinct on seeing 「x5 」x2
is a corrupt copy, and three independent editions all have it.

**2. Even a correctly parsed quotation can be a whole discourse.** The sūtra's address to Śāriputra
is ONE 246-token quotation, and "a balanced quoted span holds no boundary" then yields a 254-token
sentence. Hence `max_span`, and `max_sent` for the comma-chaining equivalent.

**Both caps are derived from the treebank rather than chosen:**

    Kyoto balanced quoted spans   n=1719   median  8   p99 24   MAX 42   -> max_span 60
    Kyoto gold sentence lengths            p99    26   p99.9 37  max 123  -> max_sent 100

so neither can clip anything Kyoto attests. On the released harness they are free (LAS 76.39 /
SENTS_F 95.27 with and without); on multi-sentence input they help (LAS 71.35 -> 71.66). The sūtra
now comes out as 9 sentences, median 25, with 「舍利子！ standing as its own address and the 故知…
passage closing at its 」.
