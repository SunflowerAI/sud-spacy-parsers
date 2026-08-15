# Vocalisation, orthographic augmentation, and the ar/fa arms

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

`ar_vocalise` and `fa_vocalise` restore the vowels the script omits; both were released 2026-08-14
(v0.2.0, clobbered). Same component shape as `la_macronise` -- purely
additive, `token._.vocalised` / `token._.vocalised_level` / `doc._.vocalised`, `doc.text` never
touched -- and the same degrade-don't-fail posture. `add_vocalise.py --lang {ar,fa}` is the shared
surgery; nothing reads the output, so no weight moves and every published figure stands.

**They are NOT the same proposition, and the difference is the data, not the language.**

## ar -- gold in the treebank, morphology does the work

Every PADT token carries `Vform=` in MISC (223 881 train tokens), so the table is a straight
extract. **The morphology is the whole difficulty**: a majority table keyed on the bare consonantal
skeleton tops out at **84.92 %**, +UPOS 86.90 %, +FEATS **98.55 %** -- because the final short vowel
is the case ending (iʿrāb), a syntactic fact. Two thirds of tokens sit under an ambiguous skeleton;
under (form, UPOS, FEATS) that falls to 8 %.

    PADT test, 28 264 tokens          WER      DER    WER-ce
    gold morphology, table only      .0951    .0481    .0505
    gold morphology + CAMeL          .0885    .0323    .0353
    PREDICTED morphology + CAMeL     .1386    .0562    .0609   <- deployment, 86.14 % exact

⚠ Do NOT read that WER against published diacritisers (Shakkala ~6.37): those are measured on
Tashkeela, a largely pre-vocalised classical corpus, not MSA news.

CAMeL Tools' `calima-msa` analyser is the fall-through for the 6.07 % of tokens whose skeleton train
never saw. It is **GPL v2, never bundled** -- fetched with `camel_data`, which ar needs for its
tokeniser anyway. ⚠ **CAMeL and PADT write different conventions and comparing them naively costs
24 points**: sukūn (PADT writes 39 in 223 881 tokens), the fatḥa PADT writes before a long ā,
tanwīn/alif order, and the article's hamzat waṣl. Oracle recall on out-of-table tokens is 13.63 %
raw and **37.68 %** normalised, so `canon`/`to_padt` are most of the fall-through's value. It barely
moves WER (+0.06) but cuts DER 17 % relative -- it gets an unknown word's vowels right and its case
ending wrong.

⚠ **`X`/PUNCT/SYM are harvested, not passed through.** PADT leaves most foreign material bare but
vocalises what it recognises (`برلين` -> `بَرلِين`), so neither answer alone is right;
table-then-identity scores **99.20 %** on test `X` against bare identity's 90.43 %. They take a
shorter ladder (UPOS-keyed rungs, then the token unchanged), skipping the skeleton-only rung whose
majority is taken over the other parts of speech.

## fa -- no gold anywhere, so coverage is the only honest number

**Nothing here annotates Persian vocalisation.** PerDT's `Translit=` is a mechanical romanisation of
the CONSONANTAL spelling (`nšst` for نشست). No `Ezafe` feature either -- checked SUD PerDT, upstream
UD PerDT and UD Seraji. So the table is RECONSTRUCTED by aligning a pronunciation lexicon onto
spellings (`fa_align.py`, a DP over graphemes x phonemes: Persian writes long vowels as letters and
leaves short ones unwritten, so every phoneme is either realised by a grapheme or inserted as a
diacritic on the one before). 97.87 % of Tihu aligns, **zero round-trip failures** -- the aligner
never alters the skeleton, which is the invariant that matters. It handles the silent و of `خواهر`
-> `خواهَر` and gemination `بچه` -> `بَچّه`.

Sources cascade **KaamelDict** (116 629 words, GPL, coverage) under **Tihu** (47 149, MIT-wrapped,
consistency -- it wins on overlap, being one editor's single scheme). Coverage on PerDT test, over
the 89.9 % of tokens that are not PUNCT/SYM/X/NUM: Tihu alone 72.80 % -> **94.30 %**. That is
COVERAGE, not accuracy; no figure here is scored against Persian gold because none exists.

⚠ **KaamelDict's homograph metadata is thinner than its README implies.** Of 116 629 entries only
3 261 carry more than one pronunciation and only **580** carry any POS, and the lists are not always
aligned 1:1 (شکر: four readings, two tags). The `prob` column contradicts the README on the very
word the README uses -- در reads `[10.0, 90.0]` against `[dar, dorr]`, making the rare literary
*dorr* the likely one. So POS is used only where it aligns one-to-one (**129 usable readings**),
`prob` is never used to pick, and the rest falls back to Tihu or the first reading.

**Ezāfe is the one part the pipeline really decides**, and it IS measurable. The ezāfe -e linking a
head to its modifier is syntactic, so no dictionary supplies it and the parse can. Gold is derived
from the one place Persian is FORCED to write it -- after a vowel-final stem, as ی or ٔ --
(`build_fa_ezafe_rules.py`), giving a subset where presence and absence are both evidence:

    host  rel   dep      n     ezāfe written        base rate, no following dependent: 12.5 %
    NOUN  mod   ADJ    5377      92.6 %   kept
    NOUN  mod   PROPN  1039      89.9 %   kept
    NOUN  mod   NOUN   5778      85.0 %
    NOUN  mod   VERB     52       1.9 %   a relative clause takes no ezāfe -- the rule working
    NOUN  udep  ADP    1046      25.7 %

⚠ **The observability test is on the FORM's ending, not on `form == lemma + ی`.** The latter looks
right and is badly wrong -- it only catches uninflected stems, so `اعضای` (lemma عضو) and `نیروهای`
were scored as counter-examples although they plainly carry the mark. Fixing it moved the headline
`mod` cell from 54.1 % to 85.4 %. The kasra is added ONLY to a consonant-final host: on a vowel-final
one the ezāfe is a LETTER, and writing it would change the skeleton.

**The two fa data files have different provenance and therefore different fates.** The LEXICON is
GPL and can never travel in a CC BY-SA wheel; the EZĀFE RULES are derived from PerDT, which is
CC BY-SA 4.0 -- the wheel's own licence and its own training data. **So the shipped wheel is bare of
the lexicon and still inserts ezāfe out of the box**, and `__call__` treats rules-without-lexicon as
data rather than as "no data".

## `ar_vocalise` returns the CALLER's spelling (user decision, 2026-08-14)

A vocalisation may only ADD marks: stripping them from the output must give back exactly what was
passed in. `la_macronise` holds that by contract and `fa_align` is verified to zero round-trip
failures; **ar_vocalise shipped without it**, and 5.41 % of PADT test tokens came back RESPELLED --
836 with a hamza the caller had not written (`الانباء` -> `اَلأَنبَاءِ`), 410 with digits
transliterated (`8` -> `٨`), and **282 that were simply the wrong word** (`دوامة` -> `دَوامُهُ`).
The cause: PADT's `Vform` is an orthographic NORMALISATION of the running text, not merely a
pointing of it, and the component treated it as only a pointing. `reproject()` enforces the
invariant -- same skeleton is taken as-is; one differing only by hamza carrier or digit form has
its marks transferred positionally onto the CALLER's letters; anything else is a different word and
is refused, so the caller gets its input back rather than a confident wrong answer. **0 of 28 264
skeleton changes.**

⚠ **MEASURE IT FOLD-BLIND, or you will punish the component for honouring its contract.** Scoring
the caller's spelling letter-for-letter against a gold that normalises orthography understates it
by ~4.7 points (81.73 % against a fair **85.71 %**). `eval_ar_vocalise.py` now compares hamza- and
digit-blind. The real cost of the guard is 86.14 -> 85.71 = **0.43**, all of it the 289 tokens where
every candidate was a different word and the component now declines to guess. Same shape as
`la_macronise`'s `_PARADIGM`: measured agreement falls, behaviour improves.

**The analyser rung is ranked by LEXEME FREQUENCY.** calima lists readings in its database's own
order, often the rarer lexeme first: `للمدرسة` gave `لِلمُدَرِّسَة` "for the teacher" ahead of
`لِلمَدْرَسَة` "for the school", and the code took the first survivor. The `LEX` table (7 370 PADT
lemma counts; calima's `lex` and PADT's LEMMA are both vocalised lemmas in the same convention)
ranks them. +0.04 MB. Released 2026-08-14 (v0.2.0, clobbered; verified by download).

## Idiom: trained beats the rule in ARABIC ONLY (measured on test, four languages)

The `Idiom`/`InIdiom` rule is EXACT on gold trees and loses end-to-end purely to upstream error --
and it is a CONJUNCTION, so those errors multiply. Decomposing ar's 240 gold heads on predicted
annotation: rule fires 50.4 %, **both inputs missing 30.8 %**, ExtPos alone 12.5 %, no `unk` 6.2 %.
So 43.3 % fail on `ExtPos`. Meanwhile **97.5 % of test gold idiom heads have a lemma seen as an
idiom head in train** -- signal the rule reads none of. A lexicon alone will NOT do it: the
idiom-head lemmas are the commonest function words (بِ 9.7 %, مِن 9.3 %, فِي 3.1 % of their
occurrences), so a >50 %-dominance lexical rule gets P 87.1 % / **R 11.2 %**. The lemma says which
tokens COULD head an idiom; only context says when -- which is what a classifier is for.

**CONFIRMED ON TEST.** Trained with `--encoder structural` (reads DEP, **LEMMA**, POS, MORPH),
ar test, same augmented base: Idiom **65.59** v the rule's 61.89, InIdiom **66.11** v 62.30 --
recall 59.17 v 50.42 against precision 73.58 v 80.13. The dev result was not checkpoint-selection
optimism; the trade is the same shape on both splits. ⚠ It still does not reach the 67.30 the rule
scored on the UN-augmented base, so training recovers most but not all of what the vocalisation
augmentation cost this layer.

⚠ **No harness had to be built, and the earlier 0.00 was my own error.** `eval_sud_idiom.py`
already scores a trained pipe: `get_misc` reads `Token._.sud_misc`, which BOTH the rule and
`sud_tagger` write, and the script only injects the rule when no `sud_idiom` pipe is present. So
`--model training_<l>_vocal_idiom/model-best` scores the trained pipes and `--model
training_<l>_vocal_sud/model-best` scores the rule on the same base. `spacy evaluate` was simply
the wrong tool.

Original dev figures, ar, same augmented
base: Idiom **74.84** v the rule's 70.73, InIdiom **75.05** v 71.01. Recall is where it comes from
(71.60 v 59.67) at ~8 points of precision. ⚠ DEV ONLY, and `model-best` was selected on it, so this
is optimistic; **`spacy evaluate` reports 0.00 for these pipes on a converted test corpus even with
the gold present** (its scorer setup differs from the training loop's), so a test figure needs a
harness like `eval_sud_shared.py` rather than the CLI. Not shipped. Data exists for ja (13 320),
lzh (2 452), sa (1 741); fa (258) and la stay rule-based.

### The sweep: ar ships trained, ja/lzh/sa keep the rule

All four idiom-annotating languages with enough data were trained with `--encoder structural` and
scored end-to-end on TEST against the rule on the SAME base (`eval_sud_idiom.py`, gold tokens):

| | rule Idiom | trained | Δ | rule InIdiom | trained | Δ |
|---|---|---|---|---|---|---|
| **ar** | 61.89 | **65.59** | **+3.70** | 62.30 | **66.11** | **+3.81** |
| ja | **96.88** | 96.71 | −0.17 | **96.32** | 95.40 | −0.92 |
| lzh | 75.68 | 76.05 | +0.37 | **85.54** | 73.91 | **−11.63** |
| sa | **76.99** | 76.60 | −0.39 | 80.48 | **81.30** | +0.82 |

**Only Arabic gains, and TWO measured factors predict the whole table.** Neither alone does:

    lang   rule inputs BOTH present   train Idiom heads   Δ Idiom
    ar               50.4 %                  1 993        +3.70
    ja               95.7 %                  5 861        -0.17
    lzh              49.5 %                  1 050        +0.37
    sa               40.8 %                    838        -0.39

**ja has the data but no headroom**: its parser supplies `ExtPos` and `unk` on 95.7 % of gold idiom
heads, so the rule is already at 96.88 and there is nothing for a classifier to recover -- the same
fact `NEGATIVE-RESULTS.md` records from the other side (99.2 % of ja `unk` tokens are idiom-chain
continuations). **lzh and sa have the headroom but half ar's data** (1 050 / 838 heads against
1 993), and a structural encoder cannot learn from what it has not seen. ar is the only arm where
both conditions hold.

⚠ **lzh's InIdiom is the sharpest negative in the table (−11.63), and it is instructive.** The
rule's `InIdiom` is not a classification but a CHAIN WALK -- follow consecutive `unk` links up to a
head bearing `ExtPos` -- which is exact given `unk`, and lzh's parser supplies `unk` on 65.9 % of
heads. So the rule scores 85.54 on InIdiom, HIGHER than its own Idiom 75.68. A trained pipe has to
rediscover transitivity from 1 226 examples, and does not. Where a rule expresses a transitive
closure rather than a local decision, prefer it unless the data is abundant.

⚠ **`grep -c 'Idiom=Yes'` OVERCOUNTS: it also matches `InIdiom=Yes`.** An earlier draft of this
section quoted ja at 13 320 and sa at 1 741 on that basis, which are the two keys summed. Use
`grep -oE '(^|\|)Idiom=Yes'`.

**`ja` had no `src_conllu` entry** until this sweep, because it ships only the rule and so never
needed `train_sud.sh`. Added, pointing at `.udep_ruled` -- the released generation for ja (802
committed cells), not the plain `.relabeled_ext` files, which are a generation behind.

## Why the augmentation cost the Idiom rule, and why de-augmenting cannot fix it

The rule reads two of the base's own predictions, so the question "can we recover it by normalising
the input?" has a precise answer: **no, because the input was never augmented.** The Idiom
comparison feeds the plain treebank, 98.5 % of whose tokens carry no diacritic at all, to both
arms -- there is nothing to strip. The loss is inside the morphologiser's weights. On that
identical bare text:

    base                                 ExtPos     unk    BOTH
    un-augmented (training_ar_sud)        66.7 %  65.8 %  59.2 %
    augmented (training_ar_vocal_sud)     56.7 %  62.9 %  50.4 %

**`ExtPos` prediction fell 10 points while the headline metrics ROSE** (bare TAG +0.41, LEMMA
+2.21, LAS identical). That is the lesson worth keeping: **augmentation costs are not uniform
across labels.** Overall `morph_acc` is dominated by common features -- Number, Case, Gender -- which
are robust to spelling; `ExtPos` is rare, and rare labels pay for the capacity spent on spelling
invariance first. Any ship decision in the MISC layer rests on rare labels, which is exactly why
CLAUDE.md requires re-measuring them after a base change rather than trusting the headline.

The fix is therefore not to touch the input but to stop making the layer depend on a degraded
prediction -- i.e. to train it, which recovers 61.89 -> 64.98 of the 67.30.

## Normalise at the boundary, or augment? Both — and which one depends on the AXIS

The cheaper alternative to augmentation is to normalise the INPUT instead of teaching the model to
read it, which is what zh already does (`ZhTradTokenizer` converts simplified in, `zh_script`
converts back out) rather than training on two scripts. Measured against the augmented arms, on the
same variant corpora:

    ARABIC, fully pointed input          LAS     TAG   LEMMA
    released, raw                      18.50   19.77   36.46
    released + strip diacritics        72.92   89.62   90.32
    AUGMENTED, raw                     73.71   94.45   94.54   <- best
    augmented + strip                  72.94   90.03   92.53

    PERSIAN, Arabic letterforms          LAS     TAG   LEMMA
    released, raw                      57.55   79.22   62.73
    released + normalise               86.92   95.96   98.48
    augmented, raw                     86.09   95.65   94.60
    AUGMENTED + normalise              87.02   96.04   98.42   <- best

    PERSIAN, all axes (incl. ZWNJ)       LAS
    released + normalise               82.65
    augmented, raw                     85.64
    AUGMENTED + normalise              86.75   <- best

**Three rules fall out, and the third is the one worth keeping.**

1. **Normalise whatever is REVERSIBLE and carries no information.** Persian's `ی`/`ي` and `ک`/`ك`
   are an Arabic-keyboard artefact with no linguistic content, and mapping them back is exact: it
   recovers 86.92 LAS from 57.55 with the RELEASED arm and no retraining at all.
2. **Do NOT normalise away information.** Stripping Arabic diacritics costs the augmented arm
   **0.77 LAS and 4.42 TAG** (73.71 -> 72.94, 94.45 -> 90.03), because the marks it is deleting are
   the case endings. Arabic vocalisation is syntax; Persian vocalisation is not (Persian has no
   case system), which is why stripping *helps* fa (+0.63 LAS) and *hurts* ar.
3. **Augment for what cannot be normalised away.** Dropping a ZWNJ is IRREVERSIBLE -- `میرود` gives
   no way to know where the joiner was without a lexicon -- so on the all-axes row normalisation
   alone reaches only 82.65 against the augmented arm's 85.64. This is the axis that justifies the
   augmentation for fa on its own.

**They compose**, and for fa the combination is best on every axis (87.02 / 86.75). ⚠ For ar it is
NOT: normalising before an augmented arm actively destroys the signal the augmentation exists to
read. The general form -- *normalise the information-free axes at the boundary, augment for the
information-bearing and the irreversible ones* -- is the rule, and neither half subsumes the other.

**IMPLEMENTED AND RELEASED 2026-08-14** as `sud.FaNormTokenizer.v1` (`scripts/fa_normalise.py`,
swapped in by `add_fa_normaliser.py`). ⚠ `doc.text` IS the normalised text -- deliberately the
OPPOSITE contract to `ar_vocalise`, which must give the caller's spelling back, because a vocaliser
adds to what it was given whereas a tokeniser's job here is to hand the model the spelling it was
trained on. The caller's original is kept on `doc.user_data["fa_source_text"]`, with
`doc.user_data["fa_normalised"]` recording whether anything changed. Pure surgery: no weight file
moved, and `--verify` checks the RELOADED model, since assigning `nlp.tokenizer` does not update
the config.

## Vocalisation augmentation: one copy, resampled every epoch (ADOPTED 2026-08-14)

`ar_vocalise`/`fa_vocalise` make the models WRITE the vowels. This makes them READ them, which is
the other half and turned out to be the bigger defect. Measured on the SAME trees with only the
FORM column rewritten (`make_ar_variant_conllu.py` / `make_fa_variant_conllu.py`, scored by
`eval_ar_variants.py`), the released arms collapse on text they were never shown:

    ar   bare 72.92 LAS   shadda-only 63.72   half-pointed 44.81   fully pointed 18.50
    fa   bare 87.18 LAS   no-ZWNJ     82.93   Arabic ی/ک  57.55   fully pointed 33.28

ar's spread is **54.42**, which is to the decimal what Latin had before its own augmentation (54.4).
fa's is **64.40**, and its worst row is not vocalisation at all: `ی`/`ي` and `ک`/`ك` are what an
ARABIC KEYBOARD produces, so that is a large share of real Persian text rather than an exotic
edition, and it cost 29.6 LAS.

**TWO DIRECTIONS, forced by the data, and this is the thing to understand before touching it.**
ar stores the corpus FULLY POINTED (`make_ar_vocalised_corpus.py` sets FORM = PADT's gold `Vform`)
and the augmenter only ever REMOVES marks -- a strict superset, exactly as Latin stores the
macronised copy and strips, so the bare spelling is derived and can never drift from the pointed
one. `fold(strip(Vform))` reproduces the treebank's own FORM on **97.50 %** of 223 881 train tokens;
the two folds are hamza (the vocalised column RESTORES the hamza running text omits) and
Arabic-Indic vs ASCII digits (9 765 tokens), and both are sampled axes rather than errors. fa has no
vocalised gold anywhere, so its corpus stays as the treebank writes it and the augmenter ADDS marks
from the same reconstructed table and the same syntactically-derived ezāfe rules `fa_vocalise` ships
against. ⚠ That asymmetry has a consequence: the ar augmentation is exact, the fa one is only as
good as the reconstruction -- mitigated because the parser is being taught to IGNORE these marks,
so a wrong vowel is noise in the input, not a corrupted label.

**Results, full chains (base → morph → lemma), gold-preproc, same trees throughout:**

| | ar released | **ar vocal** | fa released | **fa vocal** |
|---|---|---|---|---|
| bare LAS | 72.92 | **72.92** | 87.18 | **87.28** |
| bare TAG | 89.62 | **90.03** | 96.19 | **96.25** |
| bare LEMMA | 90.33 | **92.54** | 98.71 | 98.65 |
| fully pointed LAS | 18.50 | **73.71** | 33.28 | **86.39** |
| Arabic letterforms LAS | — | — | 57.55 | **86.09** |
| **LAS spread** | **54.42** | **1.42** | **64.40** | **2.04** |

**This is a better bargain than Latin's, and the difference is instructive.** Latin paid ~0.5 LAS
and 2.7 TAG on plain text to bring its spread 54.4 → 7.0. Arabic pays NOTHING -- bare LAS identical
to the decimal, TAG +0.41, LEMMA **+2.21** -- and Persian pays nothing either (LAS +0.10, POS −0.13
and LEMMA −0.06, both inside noise). The reason is that Latin's axes (breves, ligatures) are
perturbations that add no information, whereas the vocalisation IS information: on fully pointed
text the ar arm now BEATS its own bare score (LAS 73.71, TAG 94.45, LEMMA 94.54), which is what
should happen once a model can read marks that disambiguate. As in Latin, the LEMMATISER gains most
-- edit trees are literal string edits, so it was the component least able to generalise across
spellings on its own.

⚠ One row to read carefully: ar `final` (case-ending-only pointing) has vocal LEMMA **80.72** against
~93 on every other vocalised row. That is the edit-tree lemmatiser's weak spot by construction -- a
diacritic on the LAST character changes exactly the part of the string an edit tree keys on.

**Mechanics, all inherited from the Latin recipe and all load-bearing.** `max_epochs = -1` (at `0`
spaCy lists the corpus ONCE and a corpus-level augmenter samples a single style per document for the
whole run -- the run looks normal and trains on one fixed perturbation), `shuffle = true` on the
reader (because `-1` turns off the loop's own shuffling), and `init_aug_labels.py` because
`init_nlp` initialises from `islice(train_corpus(nlp), 100)`. As in Latin the LEMMATISER is the one
that really needs the label passes: its labels are properties of the FORM, so `كتاب` and `كِتاب` are
different trees, and a missing tree does NOT raise -- `get_loss` maps it to label 0 and the token is
quietly taught the wrong edit. **Dev stays BARE and un-augmented for both**, so `model-best` is
chosen on the spelling the arm is judged on rather than drifting toward whatever the augmenter
sampled. Freeze recipe verified: tok2vec/tagger/parser (and morphologizer at the lemma layer) come
out byte-identical up both chains.

Driver `train_vocal.sh` (`corpus | variants | labels | base | morph | lemma | eval`);
`metrics_{ar,fa}_variants.json` hold the tables.

**ADOPTED 2026-08-14 (user decision), released at v0.2.0 (clobbered).** `package_sud.sh` now names
`training_{ar,fa}_vocal_sud_xpos` by default; `AR_BASE`/`FA_BASE` get the old arms back. Adoption
was NOT a repackage -- the whole storey above the base had to be rebuilt on it:

  * **The conditioned tagger had to be retrained**, not carried over. Grafting the released
    `_xposwarm` donor would have put a tagger that has only ever seen bare text into a pipeline
    whose entire point is reading pointed text, and the tagger is the component most sensitive to
    spelling. New donors are warm-started from the augmented arm's OWN tagger and trained through
    the augmenter; `XPOS_SRC_ARM` overrides the source arm.
  * **The SUD MISC layer had to be retrained too**, because it reads the base's own predictions --
    `sud_shared`'s coordination mask is a fact about that parser. `SUD_SRC_MODEL`,
    `SUD_BASE_CONFIG`, `SUD_CORPUS` and `SUD_AUGMENT` were added for this. ⚠ ar needs
    `SUD_CORPUS=corpus_ar_vocal_sud`: its augmenter only REMOVES marks, so fed the ordinary bare
    corpus it can only ever produce bare text and the augmentation silently does nothing. fa needs
    no override, its augmenter ADDS marks. `Vform` survives `hoist_sud_gold.py`, which is what
    makes the vocalised SUD corpus buildable at all.

**Every ship decision re-measured, and one real cost found.** Shared: ar trained 52.17 v rule 51.70
(still trained, but the margin narrowed from ~2.0 -- re-check after any further base change), fa
trained 68.59 v rule 58.51. Reported: ar rule 73.49 v trained 34.78, fa trained 58.33 v rule 23.53
(fa IMPROVED, 46.15 -> 58.33). **⚠ ar's IDIOM layer costs 5.41 F: 67.30 -> 61.89 end-to-end**,
precision UP 78.0 -> 80.1 and recall DOWN 59.2 -> 50.4, measured like-for-like with the same script
on both arms. That is the standing pattern for a rule that is a CONJUNCTION of two of the base's own
predictions, and it is the price of the orthographic robustness. fa's Idiom is unchanged (72.73,
n=6). Verified by downloading both published assets (sha256 identical to what was built) and loading
them from a clean `--target` install on bare, fully pointed and Arabic-letterform input.
⚠ Same version, so `pip install -U` will NOT replace an older copy; `--force-reinstall` will.

## The ar licence was wrong, and correcting it is what let the table ship

SUD_Arabic-PADT is **CC BY-NC-SA 3.0** ("distributed under the same license terms as PADT 1.0"), and
`ar_sud_padt` had been falling through to the `CC BY-SA 4.0` default since v0.1.0. A survey of every
`assets_*/` found ar to be the **only arm where the declaration and the training data disagreed**
(PADT and the three Latin treebanks are the only NC sources; la was already declared). Corrected to
`CC BY-NC-SA 4.0` -- 4.0 like la, whose sources are likewise 3.0, since ShareAlike permits licensing
an adaptation under the later version. That is what makes bundling the `Vform` table legitimate:
the table and the parser absorb the same annotation. **la stays `--no-lut` for the opposite reason**
-- its data is Morpheus, CC BY-**SA**, and bundling that into an NC wheel would impose exactly the
restriction ShareAlike forbids.

## ⚠ The driver's defaults named PRE-GRAFT arms, and it nearly shipped ar backwards

All twelve v0.2.0 wheels ship the warm-started tagger MOVED behind the morphologiser
(`graft_xpos_tagger.py`). That release grafted per arm through `SUD_BASE` and kept a `_sud_xpos`
directory only for **en_gum** and **la** -- so `package_sud.sh`'s combined `en|fa|yue|ar|id)` pattern
sent five arms to bases whose tagger predates the graft. Repackaging ar therefore rebuilt the
PREVIOUS tagger generation (89.71 -> 89.44). It built, loaded and parsed perfectly; **only a
file-by-file diff against the DOWNLOADED asset caught it** (`tagger/model`, `tagger/cfg`,
`vocab/strings.json` moved). Fourth time this lesson has been paid for, after lzh three times.

Fixed three ways: ar and fa now name `training_<l>_sud_xpos` by default (`AR_BASE`/`FA_BASE` get the
old arm back), and **`pkg()` REFUSES to package any arm whose pipeline has `tagger` before
`morphologizer`** -- the cheap invariant that encodes the whole thing. Note `graft_xpos_tagger.py`
writes the model at the path you give it while the drivers expect `<arm>/model-best`, so nest it.

**en, yue and id caught up 2026-08-15**, so all twelve arms now have a grafted `_sud_xpos` directory
and the driver names it (`EN_BASE`/`YUE_BASE`/`ID_BASE` get the pre-graft arm back). Each was rebuilt
from `training_<l>_sud` + `training_<l>_xposwarm`, verified on `corpus_<l>_sud/test.spacy`: parse
unchanged en 8 585/8 585, yue 1 261/1 261, id 11 756/11 756, tags matching the donor on every token,
`tag_acc` 0.9287 -> 0.9325 / 0.9286 -> 0.9313 / 0.9264 -> 0.9288.

**Nothing was re-released, because nothing needed to be** -- these arms RECONSTRUCT what v0.2.0 already
ships rather than making a new generation, and that is the claim worth checking rather than asserting.
Every weight file in each arm -- the five base components and the `sud_*` pipes alike -- is
byte-identical to the one hashed out of the DOWNLOADED wheel. Repackaged at 0.2.0 from the new
defaults, **yue comes out byte-for-byte identical in all 37 files**, and en/id differ in exactly
`vocab/strings.json` plus the `RECORD` that hashes it.

⚠ That `strings.json` diff is the same false alarm this file has recorded before, and this time it
has a cause worth naming: the differing strings are morph bundles carrying `SudShared=`/`SudReported=`
against the published wheel's bare `Shared=`, i.e. the hoisted-gold prefix from whichever corpus the
graft's verification pass happened to read. Incidental interning, 0 label strings lost, `parser/moves`
and every `*/cfg` byte-identical, and all three load from a clean `--target` install and render tags
without E018.

⚠ Found on the way: **`package_sud.sh` was still defaulting to `VERSION=0.1.0`** long after all
twelve wheels went to 0.2.0, so a bare call built wheels named a generation behind and said nothing.
Default bumped to 0.2.0, `VERSION=` still overrides -- the same fix as a default that names the right
arm, for the same reason.

Both wheels verified the standing three ways: the published asset's sha256 matches the built wheel,
no weight file moved against the previous asset (ar 29 identical / fa 31, 0 weights), and a clean
`--target` install runs. ⚠ Same version, so `pip install -U` will NOT replace an older copy;
`--force-reinstall` will. ⚠ Two diffs that look alarming and are not: `vocab/strings.json` shrinks
(incidental corpus tokens; **no label string is lost and `parser/moves` is byte-identical**, both
checked), and fa's `AUX`/`CONJ`/`PART` tagger labels are absent from `strings.json` in the PUBLISHED
wheel too -- they are restored from `tagger/cfg` at load, verified rendering without E018.

