# Korean: the morphemes the eojeol hides

`ko_sud_gsd` has been the weakest arm in the set since it was built — released LAS 57, UAS 66,
TAG 73 — and the headline hid which half of the data it was losing. This file records where the
loss actually is, what was built to reach it, and what that measured.

Read this before touching `configs/config_ko_*`, `scripts/ko_analyser.py`, `scripts/sud_ko_embed.py`
or `scripts/ko_order.py`.

## The diagnosis: a third of the tokens are strings the parser has never seen

Korean orthography is eojeol-based — a stem with its particles and endings fused into one
whitespace-delimited token — and the treebank tokenises that way. So every stem-and-particle
combination is a fresh string: `잡스는`, `잡스가`, `잡스를`, `잡스에게`, `잡스의` are five unrelated
symbols to a hash embedding. `scripts/eval_ko_oov.py` splits every metric on whether the eojeol
appeared in training (released arm, test, gold sentences and gold tokens):

    seen  7 645 tok (65.5 %)   UAS 75.95   LAS 71.84   TAG 96.87
    OOV   4 032 tok (34.5 %)   UAS 52.16   LAS 38.10   TAG 27.50

**34.5 % of test tokens are unseen eojeol, and they parse 33.7 LAS below the rest.** That single
split is the whole deficit — the seen column is a perfectly ordinary parser.

It is not a vocabulary shortage. Cutting each token to its first morpheme drops the
out-of-vocabulary rate from 34.5 % to 12.4 %, because the stems are in the training data already,
sitting there unreachable behind a particle:

    key                                        types in train   covers all   covers OOV eojeol
    the eojeol itself (what the parser reads)          27 752        65.5 %             0.0 %
    first morpheme                                     10 569        90.4 %            72.3 %
    lexical join (stem + compound members)             16 651        82.0 %            47.8 %
    last morpheme                                       5 888        94.3 %            83.5 %

⚠ **The arm's own lemmatiser cannot supply the key.** It recovers the stem on 97.8 % of seen tokens
and **52.6 % of unseen ones** — it fails exactly where it would be needed, because an edit-tree
lemmatiser trained on 56 687 tokens has no more evidence about an unseen eojeol than the parser
does. This is why the channel reads an external analyser rather than an upstream component, and it
is the point at which the Korean problem stops resembling the Latin one.

### Why the Latin recipe does not transfer, and what replaces it

`docs/latin.md` records a lemma-vector-plus-per-feature-morphology channel worth +1.51 LAS. Neither
half survives the move to Korean:

* **There is no morphology to read.** The FEATS column of ko GSD carries a real morphological
  feature on **0.7 %** of tokens (against Latin's 67.4 %, Sanskrit's 80.3 %), which is why the
  released `morph_micro_r` is 0.15. The information is in XPOS instead — a composite Sejong tag,
  `NNG+JX+JKB`, 1 067 combinations over 45 atoms.
* **The lemma is not a lexeme.** It is the eojeol's own morpheme analysis, `잡스+는`, with 27 749
  types over 56 687 tokens and 39.1 % of tokens carrying a once-seen lemma. Latin's table is 11 794
  lemmas over 586 604 tokens at 0.7 %. A PPMI+SVD space fitted on the Korean counts would be noise.

What both halves were reaching for — a symbol shared across the inflected forms of one word — is
supplied in Korean by segmentation, not by a vector space. Hence a channel that reads a
morphological analyser.

## `sud.KoAnalyserEmbed.v1`: the analyser as a parser input

`scripts/sud_ko_embed.py` is `MultiHashEmbed` plus, per token, from `scripts/ko_analyser.py` at
runtime: a hash column on the **first morpheme** (the lexical key), a hash column on the **last
morpheme** (the functional key — the particle or ending that says what relation the token bears to
its head), and a multi-hot block over the analyser's tagset three ways (first tag, last tag, and the
bag of every tag in the eojeol), each with its own "the analyser said nothing" bit.

**The backend is mecab-ko, whose tagset IS the Sejong tagset the treebank annotates in**, so no
mapping stands between the analyser and the XPOS column. Its agreement with gold, on test:

    seen eojeol   first morpheme = gold first morpheme  92.1 %   full tag sequence 62.7 %
    OOV  eojeol   first morpheme = gold first morpheme  79.5 %   full tag sequence 60.2 %

⚠ **It is not the annotation pipeline coming back round.** UD Korean GSD's morpheme analysis was
produced automatically, so the obvious worry is that mecab-ko is simply reproducing the annotator
and the TAG figure is a copy rather than a prediction. Agreement rules that out: the analyser
reproduces the gold tag sequence on **62.7 %** of seen tokens, not on 99 % of them. And it says
nothing at all about attachment, which is what the parser is scored on.

⚠ **The objection this had to answer, and the falsifiable prediction it made.**
`scripts/sud_lex_embed.py` records why a per-form table is usually worthless: keyed on the form, it
is a FUNCTION of the form, so conditioning on (form, f(form)) is conditioning on the form — which
the parser already reads. That argument is airtight wherever the model has a trained representation
of the form, and it fails exactly where this channel aims: an unseen eojeol hashes to an untrained
row and carries nothing learnable, while its first morpheme is a different symbol with a trained row
behind it (61.3 % of OOV tokens land on a key seen at least twice in training). So the claim was
testable in a way a headline could not settle — **the gain had to sit on the OOV column**, and
`eval_ko_oov.py` prints that column for every arm.

⚠ **Runtime, not a shipped table.** `sud_analyser_embed.py` reached the same conclusion for Sanskrit
because a frozen extract missed 6.5 % of test tokens whose forms the analyser knew. Here the
argument is sharper: the tokens this channel exists for are BY DEFINITION absent from any
corpus-derived key set, so a frozen table would answer for every token except the ones that need
answering — and would load cleanly while scoring like its own capacity control.

⚠ **The backend is recorded in the model bytes and checked on load** (CLAUDE.md hazard 10). Two
analysers do not segment alike, so an arm trained against one and run against another reads a
channel it never saw. `ko_analyser.fingerprint()` travels in the extractor's `attrs`, and the
forward pass raises on a mismatch or an absent analyser rather than falling back to "unanalysed" for
every token. `scripts/check_ko_embed.py` asserts six properties before any arm is trained through
the layer, including that a no-channel build is byte-identical to `spacy.MultiHashEmbed.v2` and that
a mismatched fingerprint refuses.

**Deployment.** mecab-ko is a runtime dependency the released wheel does not currently have. On the
development machine the backend is Homebrew mecab-ko through `natto-py` (`MECAB_PATH`); the
shippable route is `pip install python-mecab-ko`, which vendors both the library and mecab-ko-dic,
and `ko_analyser` takes whichever is present. This is the same trade the sa arm made for vidyut: a
declared dependency and a one-off data fetch, in exchange for removing the train/deploy skew a
frozen table would carry.

## Word-order augmentation: built, and the case for it is weak

`scripts/ko_order.py` re-linearises the one degree of freedom Korean has. The measurements that
shaped it (`scripts/calibrate_ko_order.py`, on train):

* **Korean is head-final and that is not negotiable**: `mod` 96.6 %, `subj` 98.4 %, `comp:obj`
  97.1 % of dependents precede their head. The relations that do not — `flat`, `conj:coord`,
  `conj:appos` — are head-initial because SUD chains coordination, so permuting them would produce a
  different ANNOTATION rather than a different sentence. Both kinds are excluded.
* **Sibling order is the free dimension**, available at 27.1 % of heads — and it is NOT uniform:

      subj  before comp:obj   96.1 %          mod before subj    47.7 %
      mod   before comp:obj   78.3 %          mod before udep    59.1 %

  A uniform shuffle would teach `comp:obj` before `subj` at 50 % against an attested 3.9 %, i.e.
  spend most of its augmented data on orders Korean does not use. The augmenter samples from the
  corpus's own bigram distribution over sibling relations instead.
* **Non-projective sentences pass through untouched.** Rebuilding the string from the tree
  projectivises it, and a crossing arc is what spaCy's pseudo-projective encoding turns into a
  `||`-suffixed label — so projectivising the input silently rewrites the LABELS being trained on.
  15.1 % of sentences are affected.
* **No mark may change its absolute position**, and keeping each one in its sibling slot is not
  enough: swapping two subtrees of unequal length shifts everything between them, which strands a
  quotation mark against a word it does not belong to. Permutation is confined to punct-free
  regions. `scripts/check_ko_order.py` asserts on the marks and on the spacing.

**The prior is not encouraging.** Latin's word-order augmentation collapsed the LAS spread across
word orders from 17.44 to 8.38 and bought **+0.13 on natural order**; Sanskrit's bought +1.70 over
three seeds, most plausibly as small-data regularisation. Korean's measured order-sensitivity, on
the released arm (`scripts/eval_ko_scramble.py`):

    rendering   what moves                                    UAS      LAS
    identity    nothing                                     68.44    60.79
    attested    siblings resampled from the corpus bigrams  65.69    58.05
    uniform     siblings shuffled uniformly                 65.58    57.81

**−2.7 LAS, against Latin's −17.4.** So there is little robustness to buy; the argument that remains
is regularisation, and ko trains on 56 687 tokens, the smallest treebank in the set. Which of those
dominates is a measurement, not a prediction — `configs/config_ko_order.cfg` is the analyser arm
plus the augmenter and nothing else, so the three analyser seeds are its control.

⚠ An earlier unconstrained prototype put this figure at −5.1 LAS. It was measuring more than word
order: it moved marks and re-linearised non-projective sentences, so part of that number was
punctuation landing in the wrong place and pseudo-projective labels changing underneath the parser.
−2.7 is the figure for a permutation that changes nothing but the order of siblings.

## Results

(filled in by `bash scripts/train_ko_analyser.sh eval` and `… oov`)

## What is NOT addressed here

* **Sentence segmentation.** Every figure in this file is gold-sentences, gold-tokens, as
  `--gold-preproc` gives — comparable to the `metrics_ko_*_gp.json` set and not to a raw end-to-end
  run. `training_ko_eojeol_seg` is the arm that learns boundaries; CLAUDE.md hazard 4 applies
  unchanged.
* **The SUD MISC layer.** ko ships none — nothing cleared the precision floor — so no `Idiom` /
  `Subject` / `Shared` figure needs re-measuring after a base change here (standing hazard 5 is
  vacuous for this language, and only for this one).
* **The morpheme-tokenised arm** (`training_ko_retok_rl`, LAS 76 / TAG 95). Not comparable — a
  different tokenisation with a different denominator and many trivially-attached tokens — but it
  points the same way as everything above: the syntax of Korean lives on the morphemes the eojeol
  hides.
