# Tamil and Telugu (`ta_sud_ttb_mwtt`, `te_sud_mtg`)

Extracted from `CLAUDE.md` so the main guide stays short — the same reason `NEGATIVE-RESULTS.md`
exists. Read this before touching the area it covers.

Two Dravidian arms built on the Latin/Sanskrit recipe: **parse off LEMMA and DECOMPOSED
MORPHOLOGY rather than surface forms alone**, plus word-order augmentation. Tamil gets both halves.
Telugu gets one, and the reason is the first thing on this page.

## The sizes, and what each treebank actually carries

| | sentences (train/dev/test) | words | LEMMA | FEATS | XPOS |
|---|---|---|---|---|---|
| SUD_Tamil-TTB | 400 / 80 / 120 | 6 329 / 1 263 / 1 989 | all | 18 categories | 9-position composite |
| SUD_Tamil-MWTT | **test only**, 534 | 2 584 | all | 16 categories | **none** |
| SUD_Telugu-MTG | 1 051 / 131 / 146 | 5 082 / 662 / 721 | **none** | **115 values in 6 465 tokens** | **= UPOS, verbatim** |

Latin trains on 586 604 tokens. These are two to three orders of magnitude smaller, and nearly
every deviation from the standard recipe below follows from that one fact.

**Licences.** TTB is CC BY-NC-SA 3.0 and MWTT is CC BY-SA 4.0, so a combined Tamil arm is
**NonCommercial** — which, as for `la`, blocks shipping any CC BY-SA-derived table (e.g. vectors
trained on Wikipedia) alongside it.

⚠ **MTG STATES ITS LICENCE THREE TIMES AND CONTRADICTS ITSELF**, and this is upstream in
UD_Telugu-MTG's own current README, not an artefact of the SUD conversion:

    LICENSE.txt              Creative Commons Attribution-ShareAlike 4.0 International
    README prose             "licensed under the terms of CC BY-NC-SA 3.0"
    README `License:` field  CC BY-SA 4.0

Two of the three say CC BY-SA 4.0 and one says NonCommercial. **The `te` wheel is declared
CC BY-NC-SA 3.0**, the more restrictive reading, because this project has already paid for the
other choice once: `ar` declares CC BY-SA 4.0 while PADT is CC BY-NC-SA 3.0, and `la` is the only
arm that got an NC declaration right first time. Under-declaring a restriction is the error that
cannot be undone by a later release; over-declaring one can. Worth raising upstream.

## ⚠ Telugu has no lemmas and no morphology, so it has no `lemvec` arm

This is a property of the treebank, not a decision. MTG's lemma column is empty on every token;
FEATS carries 92 `NumType=Card` and 56 SUD `Shared` and nothing else; XPOS is a **verbatim** copy of
UPOS, zero mismatches in 6 465 tokens. The recipe's two parser input channels have nothing to read.

**The library survey came back empty**, and it is recorded here so it is not repeated:

| candidate | what it is | verdict |
|---|---|---|
| IIIT-LTRC / anusāraka Telugu morph | 4.9 MB, **2001**, C + Perl + GDBM + "Perl-enabled vim", WX transliteration, build paths hardcoded to `/home/bharathi/…`. GPL. Claims 95 % coverage | archaeology, not a library |
| `apertium-tel` | the whole `.lexd` is **six noun roots** (dog, cow, name, tree, box, son), one case, one plural suffix | a stub |
| Indic NLP Library | unsupervised Morfessor segmentation — morpheme boundaries, no lemma, no features. GPL v3 | wrong output type |
| Stanza `te` | trained on MTG itself, so its lemmatiser has no data and returns `None` for most tokens | circular |

`scripts/llm_morph_annotate.py` is the remaining route — the local LLM this project already runs
for the `udep` relabelling — and it is **calibrated on Tamil**, which has gold lemmas and gold FEATS
for the identical task in the same family and the same annotation scheme.

### ⚠ MEASURED AND REJECTED: cross-lingual exemplars destroy the annotation

gemma4, 40 sentences per condition, exemplars always from a split disjoint from the scored one:

    condition                              LEMMA     FEATS P    R      F
    ta  0-shot                             63.21 %   0.653  0.403  0.499
    ta 12-shot, Tamil exemplars            68.03 %   0.846  0.700  0.766
    ml 12-shot, Malayalam exemplars        75.71 %   0.811  0.714  0.760
    ml 12-shot, TAMIL exemplars (cross)    48.11 %   0.471  0.469  0.470

**Few-shot is a large win in-language** (+4.8 lemma, +0.267 FEATS F over zero-shot; it fixes the
under-specification that made zero-shot useless — Person recall 0.101 → 0.591, Polarity 0.000 →
0.507). **Cross-lingual it collapses**: −27.6 lemma points and −0.290 F against the in-language
condition on the SAME target language, changing only where the twelve exemplars came from.

The mechanism is visible per feature, and it is contamination rather than mere degradation — the
exemplars transfer the SOURCE language's feature inventory onto the target:

    Animacy   in-language F 0.759   cross F 0.000  (0 TP, 183 FN — Tamil's exemplars rarely
                                                    show Animacy, so it stops emitting it, and
                                                    Malayalam marks it on every animate noun)
    Gender    7 gold instances      167 FALSE POSITIVES, P 0.029
    Person   10 gold instances      165 FALSE POSITIVES, P 0.041

Tamil marks Gender and Person on everything, so with Tamil exemplars the model invents them for
Malayalam, which does not.

**So Telugu gets no `lemvec` arm.** Telugu can only ever receive cross-lingual exemplars, and it is
*further* from Tamil than Malayalam is (South-Central against South Dravidian), so 0.470 F is an
upper bound on what it would get, not an estimate. An annotation channel at that quality would feed
the parser confident noise. The in-language numbers are what say this is a property of the
transfer and not of the model or the harness.

⚠ **MTG's empty lemma column is a live trap, not merely an absence.** `spacy convert
--converter conllu` does `lemmas.append(lemma)` with no special case and spaCy keeps CoNLL-U `_` as
a **literal string**. Verified on the unrepaired corpus: all 5 082 training tokens came out with
`token.lemma_ == "_"`. A lemmatiser trained on that learns `FORM -> "_"` for the whole language and
`spacy evaluate` reports a LEMMA score against it that looks like an ordinary number. This is
CLAUDE.md's recorded Sanskrit trap (5 043 tokens) arriving by a different route; `scripts/prep_te.py`
applies the prescribed remedy, **falling back to IDENTITY, never to `_`**.

## Telugu had NO multiword tokens, and now has some

MTG's own README states **"Word count: 6465, Token count: 6465"** — not one MWT range in the whole
treebank. That is not a fact about Telugu. Against Tamil TTB, which splits 9.67 % of its
orthographic words:

    ta TTB   AUX 6.6 %   comp:aux 608   mod@emph 228   PART 6.8 %
    te MTG   AUX 0.0 %   comp:aux   0   mod@emph   0   PART 0.4 %      in 6 465 tokens

Zero AUX in 6 465 tokens is an annotation policy; Telugu has auxiliary verbs. And MTG is
**inconsistent with itself**: the addressee particle `అండి` stands as its own `PART`/`discourse`
token once and is fused inside a verb five times.

The mechanism is Telugu's euphonic **enunciative `-u`**, which is added to consonant-final words and
ELIDES before a vowel-initial one, so the two get written as a single orthographic word. Because
Telugu is an abugida with the same structure as Tamil, `scripts/indic_sandhi.py` handles both — the
round trip is exact on **6 465 / 6 465** Telugu tokens as well as 13 043 / 13 043 Tamil ones.

`scripts/split_te_mwt.py` re-annotates. ⚠ **RE-ANNOTATING A TREEBANK IS A DIFFERENT ACT FROM
TRAINING ON ONE**, so every decision is read off MTG rather than supplied, and the script is held to
the 0.90 bar `docs/latin.md` sets for a rule that COMMITS an annotation. Five conditions, all
required: the orthography licenses the cut (left part ends in a virāma, right opens with an
independent vowel — the signature of elision); both halves are attested standalone with a ≥0.90
dominant UPOS; the relation between them is attested; the head part's UPOS equals W's own; and
re-joining reproduces the surface exactly.

⚠ **THE 0.90 BAR BELONGS PER CONSTRUCTION CLASS, NOT GLOBALLY — GATING ON THE GLOBAL NUMBER COMMITS
NOTHING.** Held out over every adjacent pair in the corpus the best rung reaches only 0.8909, and
the first version of this script therefore committed 2 splits. That pool is dominated by
(NOUN, VERB) pairs whose relation is genuinely ambiguous between `subj` and `comp:obj` — 138 against
237 — and **resolving that ambiguity is the parser's job, not this script's**. Conditioned on the
construction it separates completely:

    (DET, NOUN)  0.992 n=128      (NOUN, VERB) 0.594 n=646
    (ADV, VERB)  0.961 n=155      (PRON, VERB) 0.621 n=232
    (ADJ, NOUN)  0.927 n= 55      (VERB, VERB) 0.386 n=153
    (NUM, NOUN)  0.921 n= 63      (NOUN, NOUN) 0.359 n=103

So the gate is the finest supported class, and a closed-class dependent is its own class — which is
what "closed class" means, and what lets the `అండి` cases through on the treebank's own single
separate annotation of that particle. Three further guards, each found by something that went wrong:

- **The head part's UPOS must equal W's.** The category of a fused word is the category of its
  syntactic head, so a split that makes the head something W is not has misread the construction.
  This is what rejects `వాళ్ళని` — the accusative of `వాళ్ళు`, not `వాళ్ళు` + the quotative `అని` —
  and with it the case suffixes, the negative participles (`చెప్పని`, `చాలని`) and the
  nominalisations (`చెప్పేది`, `వెళ్ళేది`).
- **A reduplication is one word.** `ఎవరెవరు` "who all" and `అప్పుడప్పుడు` "now and then" pass every
  other test — both halves are of course attested, being the SAME word — and splitting them yields
  two identical tokens joined by an invented `mod`. Only an explicit `a == b` guard catches it.
- **The split's own two parts are not children to be redistributed.** Re-running the head-UPOS rule
  over the arc just decided overwrote it, and inflated the re-attachment count from 9 to 73.

**Committed: 20 splits, 8 distinct types** (15 train / 4 dev / 1 test), plus 9 children re-attached.

    చెప్పండి → చెప్పు + అండి   ×8      ఇళ్ళున్నాయి  → ఇళ్ళు + ఉన్నాయి
    వెళ్ళండి → వెళ్ళు + అండి   ×4      ఎవరున్నారు  → ఎవరు  + ఉన్నారు
    ఇవ్వండి  → ఇవ్వు  + అండి   ×3      డబ్బిచ్చేడు → డబ్బు  + ఇచ్చేడు
    రానప్పుడు → రాను + అప్పుడు         రేపనగా      → రేపు   + అనగా

Sentence 276, `మీ అన్నగారికి ఎన్ని ఇళ్ళున్నాయి ?`, is the case the whole exercise is about: the
subject was **inside the root token**, and `ఎన్ని` "how many" attached as `det` OF THE VERB. After
the split `ఇళ్ళు` is `subj`, `ఎన్ని` is `det` of `ఇళ్ళు`, and the dative `అన్నగారికి` stays on the
verb.

Validated end to end: sentence count unchanged, one root per sentence, no cycles, every head in
range, **every MWT range re-joins to its original surface**, the orthographic word sequence is
untouched and so `# text` still says what it said. The range line keeps the orthographic word's own
MISC (`Translit=`, `SpaceAfter=No`); the split parts get `_`, because inventing a transliteration
for a piece the annotators never transliterated would be manufacturing data.

⚠ **19 further candidates are orthographically licensed and left alone**, because no construction
class supports the relation at the bar — mostly (NOUN, VERB) and (PRON, VERB) fusions. They are
listed by `--dry-run` and are the right place for a human annotator to start; committing a guess
there would put a `subj`/`comp:obj` coin-toss into gold.

⚠ **THE TEST SET MOVED, so split and unsplit LAS are not like-for-like** (721 → 722 tokens). This
is the moving-denominator problem `docs/udep-relabel.md` records for each relabel, and the arms are
reported side by side with that stated rather than differenced. `training_te_nomwt_*` and
`corpus_te_nomwt/` keep the unsplit arm and its own gold for exactly this comparison;
`prep_te.py --no-mwt` rebuilds it.

## Two size deviations, both in `make_dravidian_config.py`

**`min_action_freq = 1`, not spaCy's default 30.** The default drops any parser action seen fewer
than 30 times, which is sized for Latin-scale corpora. Measured on these training sets it deletes:

    ta TTB (6 329 tokens)          7 of 19 labels
    ta TTB+MWTT (8 409 tokens)    19 of 33 labels
    te MTG (5 082 tokens)         14 of 29 labels

— `mod@relcl`, `parataxis`, `discourse`, `vocative`, `mod@cond`, `compound@redup` and the rest. The
parser then cannot **emit** them, so their recall is exactly zero and nothing says so: the label
simply never appears in the output.

**`tag_acc` drops out of checkpoint selection** (0.5 → 0.0, redistributed to `dep_las`). Telugu's
XPOS is UPOS, so weighting it twice `dep_las` would select a parser on a tagger's score for a task
with no content; on the combined Tamil arm 60.7 % of the MWTT half's tags are ones TTB never wrote.
The real XPOS tagger is a later layer, grafted by the freeze recipe reading UPOS+FEATS
(`docs/xpos.md`), which is why the released component order puts `tagger` behind `morphologizer`.

## Tamil: two treebanks that disagree, so both arms are kept

MWTT ships test-only and is carved 80/10/10 round-robin by `scripts/prep_ta.py`, the way
`split_yue.py` carves Cantonese-HK, then added train→train / dev→dev / test→test as
`add_perseus_la.sh` folds in Perseus. That roughly doubles the training data — but the two treebanks
**disagree about annotation, not merely about tagset**: MWTT writes `mod@poss` (28) where TTB writes
plain `mod` for the same genitive, `subj@nc` (46) where TTB writes `subj`, and subtyped
`udep@tmod`/`@lmod`/`@inst` where TTB writes bare `udep`. No map fixes that without deciding which
treebank is right, so `ttb` is kept as the control and `both` as the candidate, and the TTB test
slice is reported apart — the same way `docs/latin.md` prices Perseus.

### XPOS: MWTT rendered onto TTB's tagset, and why doing nothing looks fine

TTB carries a 9-character positional code and MWTT's column is `_`. That is **not** a hole the
tagger would skip: `spacy convert` does `tag = pos if tag == "_" else tag` and silently falls XPOS
back to UPOS, so a combined corpus gets 234 composite codes sitting beside 14 bare UPOS strings — a
mixed tagset of exactly the kind `docs/xpos.md` records for PROIEL-beside-ITTB.

The scheme was **mined rather than assumed** (`scripts/normalise_ta_xpos.py --report`). For every
position the builder scores each single feature and each PAIR by how well a majority map reproduces
the character, and keeps the best:

    pos 0  lexical    coarse POS       N V Z T J A U P R D C Q
    pos 1  lexical    POS subtype      N common, E proper, T{S,b,e,g,…} particle classes
    pos 2  Case                        1.0000
    pos 3  Tense x VerbForm            1.0000
    pos 4  Person                      1.0000
    pos 5  Number                      1.0000
    pos 6  Gender x Polite             0.9968
    pos 7  Voice                       1.0000
    pos 8  Polarity                    1.0000

Positions 3 and 6 came out as PAIRS, and keying either on its best single feature loses what the
second carries — nothing in the tagset documentation says so, which is why the search exists.

⚠ **THE LEXICAL LADDER'S ORDER IS MEASURED, NOT ASSUMED, AND THAT IS WORTH 18 POINTS.** Four rungs
are available — (UPOS, closed-class feats), (UPOS, LEMMA), (UPOS, form suffix), UPOS — and they do
**not** rank the same way at the two lexical positions:

    pos 0   closed 1.0000 > lemma 1.0000 > suffix 0.9995 > upos 0.9919
    pos 1   suffix 0.9765 > lemma 0.9292 > closed 0.7914 > upos 0.7292

Ordering them most-specific-first, as the first version did, let the 0.79 rung answer ahead of the
0.98 one. Whole-tag accuracy held out on TTB's own test went **72.10 % → 90.05 %** (dev 75.14 →
92.64) purely from sorting the ladder by what it measures. Same lesson as `la_macronise`'s key
ladder in `docs/latin.md`: a wrong answer that pre-empts the right one is worse than no answer.

## The Tamil tokeniser: an abugida makes rewriting into segmentation

TTB splits 835 orthographic words into 1 781 syntactic words, and **94.2 % of those splits rewrite
at the seam** rather than cutting cleanly:

    நிலையத்துக்குக்கான  ->  நிலையத்துக்குக்க்  +  ஆன
    துறைகளையும்         ->  துறைகளைய்         +  உம்

So the treebank's FORMs are not a segmentation of its own text, spaCy's rule tokeniser cannot
produce them, and `sud.CharSegTokenizer.v1` can only cut. **This is invisible in every parsing
figure** — `--gold-preproc` bypasses the tokeniser and `sud.GoldTokCorpus.v1` makes the parser
segmenter-agnostic — which is the blind spot `docs/lzh-tokenisation.md` records for 孔子.

The observation that dissolves it: Tamil is an abugida, so கா is not "k" then "ā" — it is one
akṣara spelling க் + ஆ, and the split points fall **inside** such characters. Decompose every
akṣara into consonant + virāma + independent vowel (`scripts/ta_sandhi.py`) and the rewriting
disappears:

    round trip `recompose(decompose(w)) == w`            13 043 / 13 043 tokens (100.00 %)
    gold parts a clean cut of the DECOMPOSED surface        842 / 878 ranges (95.90 %)
    gold parts a clean cut of the RAW surface                51 / 878 ranges  (5.8 %)

So the tokeniser is decompose → cut → recompose, and the existing trained character segmenter does
the middle step unchanged. The 4.1 % residue is a short **named** list, not a long tail:
gemination (வலிமிகல்: `கஷ்ட` + `படுகிறான்` → `கஷ்டப்படுகிறான்`), *u*-elision before a vowel
(`கொண்டு` + `இருக்கிறது` → `கொண்டிருக்கிறது`), ல்/ற் assimilation, and one suppletive `*இந்த` that is
a lemma rather than a spelling. `ta_sandhi.join_variants` recovers **29 of the 36**, leaving 7.

⚠ **Both sandhi rules must be stated in DECOMPOSED space.** Written on the composed string,
*u*-elision removes the vowel SIGN and silently leaves the consonant with its inherent அ —
`கொண்டு` + `இருக்கிறது` came out `கொண்டஇருக்கிறது`. That was the first version and it recovered
nothing at all.

⚠ **Gemination is offered, never chosen.** `join_variants` returns a LIST because வலி மிகும் /
மிகாது turns on the morphology and the lexeme, not on the phonology: plenty of the 842 clean ranges
have a hard-initial second part and do not geminate. A function that always geminated would break
them.

**Result** (`scripts/eval_ta_tokenizer.py`, strict token F on raw `# text`, 173 test sentences):

    tokenizer            correct    pred        P        R        F
    rule (baseline)         1797    2049   0.8770   0.8040   0.8389
    trained segmenter       2104    2232   0.9427   0.9414   0.9420

⚠ **The input regime travels with the weights** (`reads_decomposed` in the bundled
`ta_tokenizer.json`) and is **read back**. A segmenter trained on decomposed Tamil and handed
composed Tamil would not raise — it would meet a character inventory it half recognises and quietly
under-split. Standing hazard 10, which has already been paid twice.

⚠ **This tokeniser rewrites its input**, as Sanskrit's does: `doc.text` is the recomposed parts and
does not always equal the string handed in. A Doc from it must never be re-tokenised, and anything
rebuilding a Doc must build it from the WORDS.

## Word order: rigidly head-final, so the Latin transform is the wrong one

`scripts/dravidian_order.py`. Measured on the training corpora, as the share of dependents that
**precede** their head:

    deprel        ta        te
    subj       99.0 %    99.9 %
    comp       99.7 %    98.6 %
    udep       99.8 %    99.1 %
    mod        94.0 %    96.9 %
    det        98.3 %    96.1 %
    conj        0.0 %     0.0 %     a conjunct FOLLOWS what it coordinates with
    root is the last non-punctuation token in 93.7 % / 93.2 % of sentences

**Telugu is 99.9 % projective — one crossing sentence in 1 051.** Tamil has 18.0 %. So Latin's
central move, generating hyperbaton because a projective re-linearisation would hand the model a
corpus without discontinuity, applies to Tamil weakly and to Telugu **not at all**: `p_hyperbaton`
is 0.08 for `ta` and **0.0** for `te`, because inventing displacement in Telugu would be inventing
a construction the language does not have.

**The side of the head is read off the data and never assigned.** A child that was before its head
stays before it; one that was after stays after. Head-finality therefore falls out of the corpus
rather than being encoded as a rule, and the post-head phenomena the treebanks do have survive
without being listed. The check confirms it: Telugu's dependent-precedes-head rate comes out
**95.581 % → 95.581 %**, unchanged to three decimals.

**What is actually augmented is the preverbal field**, and it is genuinely free rather than merely
variable — subject before object runs 157/56 in Tamil (26 % OSV) and 198/59 in Telugu (23 %). With
400–1 000 training sentences a parser will otherwise memorise the orders it happened to see as
though they were rules.

⚠ `clause_only` **defaults to true and is an open question.** Scrambling is described as clausal and
prenominal order (DEM > NUM > ADJ > N) as rigid, but among nominal heads with two or more pre-head
children, 98 % (ta) / 71 % (te) sit in a label-set the corpus attests in more than one order. That
measure is coarse — it pools across heads and strips subtypes — so it licenses an experiment, not a
conclusion. Hence a knob rather than a decision.

⚠ **`max_epochs` must be `-1`, with `shuffle = true` and collected labels.** Same three-part hazard
as `docs/latin.md`, but the third part bites harder here: **the parser's own labels are a property
of the ORDER**, because a non-projective gold tree is pseudo-projectivised and the lifted arc picks
up a `||` suffix naming what it was lifted over. Under Latin's orthographic augmentation only the
lemmatiser's edit trees moved; here the parser's do. Collect over six passes with
`init_aug_labels.py` and read the coverage it prints.

⚠ **A permutation bug in HEAD does not raise.** It yields a well-formed `Example` carrying a
different tree, trains happily, and appears in no log and no metric.
`scripts/check_dravidian_order.py` asserts the arc set, every per-token annotation and every
sentence boundary survive. It computes the permutation from a second RNG rather than recovering it
by matching forms — recovery cannot tell a genuine arc bug from two identical tokens swapping, and
the first version of the check reported **13 false failures** for exactly that reason.

### Result: a bad trade for Dravidian, and the reason is that the transform is CORRECT

`scripts/make_dravidian_scrambled_conllu.py` renders the test set in each order — same trees, same
gold, only the string moves (173/173 sentences verified to preserve the tree) — and both arms are
scored across all of them. TTB arms, gold-preproc, LAS:

    arm                identity    order   order_free   order_nohyp    spread
    ttb_seg (base)      57.07      49.50     49.12         53.68        7.95
    ttb_order           54.07      50.71     49.44         53.01        4.63

So the augmentation **costs 3.00 LAS on natural order** and buys 1.21 on the scrambled order it was
trained for, 0.32 on `order_free`, and **loses** 0.67 on `order_nohyp`. Against Latin — which paid
0.5–0.8 LAS to take its spread from 54.4 to 7.0 — this is a far worse bargain.

⚠ **The reason is that the transform respects head-finality, which is the thing to get right.**
Latin's baseline collapsed on an unseen orthography (LAS 18.74 on a breve-marked edition), so there
were 54 points sitting there to be recovered. Tamil's baseline loses only 7.95 across every order
this module can generate, precisely because the module reads the side of the head off the data and
never assigns it — so a re-linearised Tamil sentence is still a head-final Tamil sentence. Having
correctly refused to destroy the positional signal, there is little left for the augmentation to
buy. **A negative result that follows from a design decision being right is not an argument for
reversing the decision**: a version that shuffled head position freely would show a bigger gain on
its own scrambled test and would be teaching the parser something false about Dravidian.

**Not shipped for Tamil on this evidence.** ⚠ Single seed, and `docs/latin.md` records that a
single-seed Latin delta below ~0.3 LAS says nothing and one below 0.8 can happen by chance — on a
400-sentence treebank the band is certainly wider. The 3.00-point cost is well outside any
plausible band; the 1.21-point gain is not, and should not be quoted as established.

## The `lemvec` arm: lemma as identity, not as geometry

`scripts/make_ta_lemvec_config.py`. Tamil gets the per-feature morphology channel exactly as Latin
does, and the lemma channel in its **identity** form — `LEMMA` added to the embed's `attrs` as its
own hash table — rather than as a distributional vector block. The reason is corpus size, not
preference: the vector block is PPMI+SVD over the training treebank's own lemmas, Latin has 529 809
lemma tokens to build it from and Tamil has 8 409. A co-occurrence matrix over eight thousand
tokens is noise.

That is not a consolation prize: the Sanskrit oracle grid measured gold lemma **identity**,
hash-embedded, at **+2.22 LAS**, and the vector block exists to test whether generalisation beyond
identity buys anything further — a question needing a corpus Tamil does not have. Building the
vectors from external raw Tamil text is the route back to it, and it is blocked on licensing for
the combined arm (CC BY-NC-SA against Wikipedia's CC BY-SA), the same conflict that keeps Morpheus
out of `la_macronise`.

The frozen morphologiser and lemmatiser move to the **front** of the parser, sourced and listed in
`annotating_components`, so the parser reads their PREDICTED output at training time as well as at
run time — what makes the arm shippable rather than an oracle. They can be moved because the freeze
recipe gave each its own `HashEmbedCNN`: neither is a listener and neither reads the parser.

**The feature list is read off the treebank**, not written down, and two categories are excluded by
name. **`Shared` is not morphology** — it is SUD's own coordination annotation and the TARGET of the
`sud_shared` pipe trained as a later layer, so feeding it to the parser would hand a downstream
layer's gold answer upstream. `PunctType` and `NumForm` govern no attachment.

The control (`--control`) replaces every `feats` table with one more differently-seeded `NORM` table
and the `LEMMA` table with another, keeping the architecture, the table count, the rows and the
Maxout width — so any gain over it is the two channels and not their parameters. A loose control is
what made the Latin `morphfirst` result unreadable for a generation (LAS 0.7256 against 0.7255).

### Result: it works, and the control is what makes that legible

Combined test, gold-preproc, LAS:

    arm      baseline   control   lemvec    lemvec - baseline   lemvec - control
    ttb        57.07     55.40    57.48          +0.41              +2.08
    both       59.38     58.39    59.73          +0.35              +1.34

**The channel is worth 1.3–2.1 LAS of INFORMATION** — that is the number the control isolates, and
it is the number this arm exists to produce. Its margin over the plain baseline is much smaller
(+0.35 to +0.41) because the tables also cost capacity, and how much they cost depends on how much
data there is to pay for them. Dev makes that visible:

    dev LAS   baseline   control   lemvec     capacity effect   information
    ttb (6 329 tokens)   0.6199   0.5852   0.6113      -3.47          +2.61
    both (8 409 tokens)  0.6143   0.6150   0.6246      +0.07          +0.96

At 6 329 tokens the extra tables cost 3.47 LAS on their own; at 8 409 they are **free** (the control
lands within 0.07 of the baseline) and the whole gain is information. ⚠ The two splits disagree in
sign on `ttb` (dev −0.86, test +0.41) — 80 dev sentences is not enough to settle a delta of that
size, and the test figure, on 173 sentences, is the one to quote.

**This is the INVERSE of Latin's decomposition**, which is worth stating plainly because the same
experiment on the same architecture gave the opposite reading: there, `docs/latin.md` records that
**half the +1.51 gain was the extra embedding rows** and the honest range for the information was
0.76–1.46. Here the rows are a liability at the smaller size and neutral at the larger, and
essentially all of the gain is information. Neither result is legible without the tight control.

**On the MWTT slice the gain is much larger** — `both` scores 70.47 baseline against **74.09**
lemvec (+3.62). MWTT is the out-of-domain half (grammar-book examples against TTB's news prose), and
a lemma channel generalises across domains where a form channel memorises.

## XPOS conditioning: neither language has anything to condition on

`docs/xpos.md`'s conditioned tagger reads UPOS+FEATS above the encoder, and `package_sud.sh`
refuses any arm whose `tagger` still precedes its `morphologizer`. Both Dravidian arms need the
MOVE and neither needs the CONDITIONING, and both halves of that are measured rather than assumed.

**Telugu**: XPOS is a verbatim copy of UPOS — **0 mismatches in 12 970 tokens**, before and after
the MWT split. Exactly `sa`'s situation: there is no tagset to condition on.

**Tamil**: the composite 9-position code looks like Latin's ITTB tags, which gained 13.8 points
from conditioning, so the expectation was that it would too. It does not.
`build_feats_inventory.py` on TTB+MWTT:

    8 409 tokens, H(XPOS) = 6.053 bits, H(XPOS | form) = 0.163 bits
    IG|form:  Gender 0.039   VerbForm 0.038   Polarity 0.030   Person 0.023   Case 0.019

**The FORM already determines the tag** — 0.163 bits of residual entropy — so UPOS+FEATS have
essentially nothing left to add, and every feature's information gain GIVEN the form is inside the
noise. That is the zh/id/ko result (0 features clearing the bar), not the Latin one, and the reason
is typological: Tamil is agglutinative, so the inflection the tag restates is sitting in the
suffix of the form, where a character-window encoder can already read it. Latin's gain comes from
SYNCRETISM — one form, many analyses — which Tamil largely lacks.

So the tagger is **moved without being retrained**, using the arm's own tagger as its own donor:

    graft_xpos_tagger.py training_ta_ttb_lemma/model-best training_ta_ttb_lemma/model-best OUT \
        --corpus corpus_ta/ta_ttb-sud-test.spacy
    ['tok2vec','tagger','parser','morphologizer','lemmatizer']
      -> ['tok2vec','parser','morphologizer','lemmatizer','tagger']
    parse unchanged: 1989/1989 tokens      tags match donor: 1989/1989 tokens

All three of the graft's checks still bite — shared components byte-identical, the parse reproduced
token for token, the tags reproduced token for token — so this is the ordinary graft with a donor
that happens to be the recipient, not a bypass. **No new exemption in `package_sud.sh`'s guard is
needed**, which is the right outcome: the guard tests pipeline ORDER, and the order really is fixed.

## The SUD MISC layer: trained, measured, and shipped nowhere

Both languages were put through `train_sud.sh` on the arm each wheel actually ships (`SUD_SRC_MODEL`
pointing at `training_ta_both_lemvec` / `training_te_lemma`, because this layer reads the base's own
predictions). The result is the ko/zh/yue/sa outcome, and the reason is the annotation, not the
recipe. What the treebanks contain, over ALL splits:

    ta   Subject 31   Idiom 2   InIdiom 2   Shared Yes 11 / No 60
    te   Subject  6   Idiom 0   InIdiom 0   Shared Yes 26 / No 30

Per split it is thinner still, and two features are **unmeasurable before anything is trained**:
ta's `Idiom` has 0 instances in dev and test, and te's `Subject` has 0 in both. ta's `Shared` has
**0 positive test instances**.

Test, gold tokens, everything else predicted:

    ta Subject   trained  P 83.33  R 62.50  F 71.43   gold  8   (rule identical)
    ta Subject   dev      P 50.00  R 16.67  F 25.00   gold  6   (rule identical)
    ta Shared    mask covers 0.00 % of gold over 42 candidate tokens
                 morphologiser P 100.00 R 18.75 F 31.58 | rule 0.00 | trained 0.00   gold 16
    te Shared    mask covers 0.00 % of gold over  0 candidate tokens — all arms 0.00  gold  6
    te Subject   gold 0 — nothing to score

⚠ **`Subject` LOOKS SHIPPABLE ON TEST AND IS NOT.** F 71.43 would be the best figure in the whole
released table (fa 67.7 is the current high), and it is computed over **eight gold instances**. Dev,
on six, gives F 25.00 from the same model. Pooling both splits — 14 gold, 8 predictions, 6 correct —
gives P 75.0 % with a **95 % confidence interval of [41 %, 93 %]**. The criterion in
`package_sud.sh` is a PRECISION FLOOR, "an annotation wrong more often than right is worse than
none", and an interval spanning 50 % cannot establish that it is cleared. So nothing ships. The
point estimate is not the finding; the width of the interval is.

**`Shared` fails for a structural reason worth recording.** The candidate mask — which
`docs/sud-misc-layer.md` calls the whole design — reaches **0.00 %** of Tamil's gold over 42
candidate tokens, and Telugu's test set contains **no coordination candidates at all**. The mask is
defined over conjuncts, so this is a fact about how little coordination these corpora have, and it
is a hard ceiling on rule and trained arm alike. Tamil's morphologiser meanwhile reaches P 100.00 /
F 31.58 predicting `Shared` inside its ordinary FEATS bundle — the zh situation exactly, where the
morphologiser wins uniquely — so its value is **left alone** rather than overwritten by a pipe that
scores 0.

The arms are on disk (`training_ta_sud`, `training_te_sud`) and the drivers are wired
(`src_conllu`, `eval_sud_subject.TEST`, `build_sud_shared_frames.TRAIN` all carry ta/te), so this
is a re-measurement away if either treebank grows. **Re-measure after any base retrain** — standing
hazard 5.

## The two wheels

**BUILT, NOT RELEASED.** `build_sud/ta` and `build_sud/te` hold wheels; nothing is on a GitHub
Release, and standing hazard 1 applies until it is.

| | `ta_sud_ttb_mwtt` 0.1.0 | `te_sud_mtg` 0.1.0 |
|---|---|---|
| arm | `training_ta_both_lemvec` (lemma + per-feature morphology) | `training_te_lemma`, split corpus |
| pipeline | `morphologizer, lemmatizer, tok2vec, parser, tagger` | `tok2vec, parser, morphologizer, lemmatizer, tagger` |
| tokeniser | `sud.TamilSandhiTokenizer.v1` (trained, akṣara-decomposed) | `sud.TeluguSplitTokenizer.v1` (lookup) |
| licence | CC BY-NC-SA 3.0 | CC BY-NC-SA 3.0 |
| size | 16.3 MB | 10.7 MB |

Neither ships a SUD MISC layer (`Idiom`/`Subject`/`Shared`). Both were TRAINED and MEASURED — see
the section above — and neither clears the precision floor on the evidence the treebanks provide:
ta `Subject` reaches P 75.0 % over eight predictions, a 95 % interval of [41 %, 93 %], and `Shared`
is capped at zero by a candidate mask that reaches no gold at all. Same outcome as ko/zh/yue/sa.

**The morphologiser and lemmatiser run FIRST in the ta wheel**, which looks wrong beside every other
arm and is not: the `lemvec` parser READS their predictions, so they must precede it. `la`'s lemma-
vector arm has the same order for the same reason. The packaging guard is satisfied because it tests
`tagger` against `morphologizer`, and the tagger is last.

⚠ **THE `te` TOKENISER EXISTS BECAUSE THE SPLIT CREATED AN INPUT-REGIME MISMATCH**, and it was
caught only by loading the built wheel. The arm trains through `sud.GoldTokCorpus.v1`, so nothing in
training exercises a tokeniser at all; with spaCy's rule tokeniser the wheel met `ఇళ్ళున్నాయి` as ONE
token having only ever seen `ఇళ్ళు` + `ఉన్నాయి`. Standing hazard 10, reached by re-annotating rather
than by retraining. It is a LOOKUP over the 8 committed types, not a model — eight types is a
lookup table however it is dressed — and it is harvested from the split treebank so it can never
claim a split the gold does not contain.

⚠ **`scripts/indic_sandhi.py` uses a PLAIN CLASS where the rest of the project uses `@dataclass`.**
spaCy loads a `--code` module by path without putting it in `sys.modules`, so `dataclasses` cannot
resolve its own string annotations and dies with `AttributeError: 'NoneType' object has no attribute
'__dict__'`. That fires at `spacy package` time, not at training time, so it is invisible until a
wheel is built — and the module has to load inside the wheel too.

**Verified by installing both wheels into a clean `--target` with `scripts/` OFF `sys.path`**, which
is how `la`'s missing `--code` entry (`E893`) was found:

    ta  சென்னை | அருகே | நிலம் | எடுக்கப் | படும் | . | அவர் | வந்தார் | .     9 tokens, 2 sentences
    te  మీ | అన్నగారికి | ఎన్ని | ఇళ్ళు | ఉన్నాయి | ? | చెప్పు | అండి | .        9 tokens, 2 sentences

Both split correctly from RAW text — `எடுக்கப்படும்` and `ఇళ్ళున్నాయి`/`చెప్పండి` — and both find
their own sentence boundaries.

## Drivers

    scripts/prep_ta.py          stage both treebanks, split MWTT, project XPOS
    scripts/prep_te.py          stage MTG, fall the empty lemma column back to IDENTITY
    scripts/train_ta.sh         prep | base | layers | lemvec | order | eval
    scripts/train_te.sh         prep | base | layers | order | eval
    scripts/train_ta_charseg.sh the tokeniser: decomposed pairs, train, strict token F
    scripts/llm_morph_annotate.py --score on ta (calibration); the te route is REJECTED, above
    scripts/split_te_mwt.py     --dry-run | --apply   the te MWT re-annotation
    scripts/indic_sandhi.py     akṣara decomposition, shared by ta and te
    scripts/graft_xpos_tagger.py  move the tagger behind the morphologiser (donor = recipient)
    scripts/bundle_ta_tokenizer.py  attach the trained ta segmenter, verified on a RELOAD
    scripts/te_tokenizer.py     the te lookup splitter; table harvested from the split treebank
    scripts/make_dravidian_scrambled_conllu.py  the word-order test renderings
