# Sanskrit (`sa_sud_vedic_ufal_dcs`)

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

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

**Compound membership the annotator left implicit (`sa_compound_rule.py`).** A samāsa member is in
STEM form — the final member carries the compound's case, number and gender and every member before
it is morphologically bare — so inside a multiword token, a non-final member of a nominal-capable
word class (NOUN/PROPN/DET/NUM/ADJ/PRON, **or a participle**) carrying **no morphological features
proper** is a compound member whether or not the treebank says so. Six tokens across UFAL and DCS:
`dharma` in `dharmopārjitabhūrivibhavo`; `svasti-daḥ`, `bhāga-karaḥ` and `sarva-dehinām` in one DCS
nāmāvalī whose FEATS were never filled in; and the two participles `vṛkta-barhiṣam` (the Rigvedic
bahuvrīhi) and `svayaṃvara-āgatā-tyāgāt`. `dharma` is the one that proves the rule rather than
assuming it — the treebank DID record its membership, in the **XPOS** column, one field left of
where it belongs; `normalise_sa_xpos.py` cleaned that cell and deliberately left FEATS alone, so the
only record of it was deleted and the compound has been a member short ever since, rendered
`dharm' ôpārjita-…` (a separate coalesced word) instead of hyphen-joined. The two scripts now
compose: the rule puts the fact in FEATS, `normalise_sa_xpos.py` clears the stale XPOS copy.

Three things about it are load-bearing. **`VerbForm` is a POS SUBTYPE, not a morphological feature.**
It says which kind of word this is — participle, converb, finite verb — not how the word is
inflected, so a participle whose only FEATS is `VerbForm=Part` is morphologically bare and the rule
admits it; it is also the only thing identifying a participle at all, since UFAL and DCS both write
one as `VERB` + `VerbForm=Part` rather than giving it a UPOS. Counting it as morphology silently
costs both participles above — and `add_compound` has to KEEP it when it stamps, or the membership
is recorded by throwing the participle away.
**"No morphological features" is not the same test as "is a bare stem"** where morphology is
unannotated, so the rule also requires the multiword token's FINAL member to be able to END a
compound — a nominal or a participle, never a conjunction, a particle or a finite verb. That is the
same premise read from the other end, and it is what the numbers turn on: without it the rule fires
on 11 tokens of which only 6 are right, the other 5 being four `X-aḥ + ca` sandhi joins
(visarga-final nominatives with empty FEATS) and UFAL's `dūra` in `dūrādevāśṛṇot`, where the
orthographic word is `dūrāt eva aśṛṇot`. With it: 6 fired, 6 right.
**And it belongs on the SOURCE treebanks' own multiword tokens, never on the orthographic groups
`restructure_sa_csl.py` builds** — those group anything external sandhi fused, and a fused group can
perfectly well end in a nominal, so the guard does not save it: run downstream it stamps `kiṃca`
45 times over, in `kiṃc' âtr' ânnam` and friends.

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

**The daṇḍa IS the MISC separator, so its own gold cannot be read with `split("|")`
(`scripts/conllu_misc.py`).** `Unsandhied` lives in MISC, CoNLL-U separates MISC attributes with
`|`, and the Sanskrit daṇḍa is that same character — so a daṇḍa token's padapāṭha form is written
`Unsandhied=|`, and every naive reader in the sa chain returned the EMPTY STRING for it. Because
`Unsandhied` sorts last, the pipe that swallowed the value was the one immediately before the
newline. The failure was not symmetric with a missing value: `make_unsandhi_corpus.py` parks the
value in the LEMMA column, so an empty read wrote an **empty column 3** — a malformed row, not an
absent lemma — and scored the daṇḍa as a token "needing an edit", i.e. supervision to DELETE it.
`sa_csl_prep.set_misc` was worse in the other direction: it dropped the empty item on rewrite, so a
pass that only meant to set `SpaceAfter` turned `Unsandhied=|` into `Unsandhied=` permanently. An
empty MISC item is not legal UD, so it can only have come from a literal `|` ending the previous
value; `conllu_misc` folds it back and every sa reader and rewriter now goes through it. Verified
byte-identical on every sa/UFAL/DCS corpus in the repo — none of which carries a daṇḍa
`Unsandhied` today, which is exactly why nothing complained.

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


**Clause merging, and the first sa arm with punctuation** (`merge_sa_reparse.py`,
`build_sa_pmerged.sh`, `configs/config_sa_mp2_pmerged.cfg`). The treebank splits its sentences into
CLAUSE units — `sent_id` `…_1`, `…_2`, `…_3` off one base — and gives each its own root. Regrouping
them by base id is trivial; the question is what relation joins them, and the answer is *ask the
parser*: merge the units into one token sequence, re-parse the merged string, and if the re-parse
attaches unit *n*'s root into an EARLIER unit, take that arc, head and label together. Otherwise the
unit keeps its own root. On the training split that links 2 677 of 4 366 non-initial units and leaves
1 689 alone, choosing `conj:coord` 1 065 times, `parataxis` 852, `comp:obj` 221, `subj` 110, `mod`
103, `comp:obl` 87 — and the corpus comes out single-rooted 92.1 % of the time (train; dev 91.3,
test 89.9) rather than one root per clause. A constant-`parataxis` merge (`merge_sa_clauses.py`) came
first and is kept only as the thing this replaced: it asserts a relation the treebank never claimed.

The treebank realises no punctuation at all — 0 PUNCT tokens, 0 `punct` arcs — but RECORDS it, as
`Punctuation=fullStop` / `Punctuation=comma` in MISC on the token each mark follows. `--punct` turns
those records into tokens. Three things had to be got right, each of which failed first:

- **Which mark.** The mapping is EDITORIAL, matching what DCS is encoding: `fullStop` → double
  daṇḍa (verse end), `comma` → single daṇḍa (half-verse). A *syntactic* mapping was written first —
  demote a fullStop internal to a merged sentence to a single daṇḍa — and discarded, because then a
  double daṇḍa would never once appear mid-sentence in training while a user pastes verses whose
  marks fall where the metre puts them. That is standing hazard 10 arriving through the training
  data. Being faithful costs two harmless oddities: 28 dev sentences end in a single daṇḍa (their
  last recorded mark is a comma) and 229 end with no mark.
- **Which STRING.** Not `।`/`॥`. The sa tokeniser transliterates to CSL and folds every double
  daṇḍa to `‖` (U+2016), so Devanagari `।॥`, ASCII `|` `||` and CSL all arrive at the parser as
  `|` / `‖` — those two strings are the only ones it can ever be shown, so those are what the corpus
  carries.
- **Where it goes.** A mark recorded on a token INSIDE an MWT is emitted after the LAST member of
  that MWT: it follows the whole surface word, and an `n-m` range row may not span a token that is
  not one of its members. Emitting it in place put 45 marks inside a range.

One more trap on the way out: `make_norm_corpus.py` ran the sandhi transducer over the marks and
appended a visarga to 93 of them (`‖` → `‖ḥ`) — it had never met a punctuation token. It now skips
any token with no letter in it. `t.is_punct` is NOT the test: Unicode files the single daṇḍa `|`
under Sm, so it comes back False.

`config_sa_mp2_pmerged.cfg` trains the morph-first arm on this corpus through
`sud.GoldTokNormCorpus.v1` — whole multi-sentence documents with gold tokenisation, so the parser
learns where a sentence STARTS — and must therefore be evaluated WITHOUT `--gold-preproc`, which
would hand it the boundaries for free. If it works it removes `clause_parser`'s reason to exist for
sa: that component splits on daṇḍas and strips them precisely because the parser had never seen one.
