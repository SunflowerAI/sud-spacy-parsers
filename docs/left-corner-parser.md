# The incremental left-corner beam parser for English (psycholinguistic modelling)

An arm that is **not a wheel**. It exists to produce two things per word — a **surprisal** and a
**memory load** — for fitting reading times, and it is a parser only incidentally. Nothing here
goes through `package_sud.sh`.

    scripts/lc_transitions.py   Noji & Miyao's left-corner transition system, its static oracle,
                                and the round-trip test that is this arm's correctness gate
    scripts/lc_features.py      oracle derivations -> per-step tensors; the shared vocabulary
    scripts/lc_model.py         the generative model and its trainer (two-stage, gold + silver)
    scripts/lc_surprisal.py     word-synchronous beam: surprisal, memory predictors, Viterbi parse
    scripts/fetch_en_selftrain.py, scripts/make_silver_conllu.py    the self-training corpus
    scripts/lc_provo.py, scripts/lc_rt_analysis.py                  Provo scoring and the regression
    train_en_lc.sh              the driver

## Why none of the existing machinery would do

**spaCy's parser is arc-eager, and arc-eager cannot be the model.** Its memory cost — the number of
connected components on the stack — *never rises* while the stack stays one connected component,
which is exactly what a centre-embedded clause produces. That is the one contrast the whole
enterprise turns on (Abney & Johnson 1991), so no amount of retraining spaCy reaches it. There is no
left-corner option in the transition system; it is Cython, and adding one means forking spaCy.

**spaCy's beam would not help even so.** Its scores are unnormalised discriminative scores. They
never normalise over what the next word *could have been*, so no function of them is a prefix
probability, so none of them is a surprisal. `NEGATIVE-RESULTS.md` records beam training losing 2.76
LAS on Latin and refuting its own premise — that finding stands, and it is about a **different
purchase**: a beam bought for accuracy. This one is bought for a marginal, and its success criterion
is how tight the marginal is, not LAS.

So: a standalone PyTorch model over the repo's CoNLL-U, in the spirit of `sud_arcfactored_parser.py`.

## The transition system, and the two things the paper leaves implicit

Six actions over a stack of **right spines**, each spine optionally carrying one **dummy node**
`x(λ)` that stands for a token not yet seen, with λ the subtrees already parked as its left
dependents. Shift-type and reduce-type actions **strictly alternate**.

    SHIFT       push <j>
    INSERT      <s|i|x(λ)>          -> <s|i|j>        + (i,j) + (j,k) for k in λ
    LEFT-PRED   <r, ...>            -> <x({r})>
    RIGHT-PRED  <r, ...>            -> <r, x({})>
    LEFT-COMP   <s|x(λ)> <r, ...>   -> <s|x(λ + {r})>
    RIGHT-COMP  <s|i|x(λ)> <r, ...> -> <s|i|r|x({})>  + (i,r) + (r,k) for k in λ

**Two rules as printed do not work, and neither failure announces itself.**

1. **Figure 2's RIGHT-COMP omits the arc `(i, r)`.** Its own oracle condition requires `(i, σ11)` to
   be gold, so the arc must be emitted. Leave it out and every subtree attached by composition comes
   out **headless** — the parser still runs, still returns a tree, and LAS is merely lower.
2. **"has one more dependent in β" means *at least* one for RIGHT-PRED and *exactly* one for
   RIGHT-COMP.** The paper writes the same phrase for both. Read both as "at least one" and the
   oracle deadlocks on the first head that has two right dependents and is not the root of its own
   stack element — and it deadlocks by **running out of buffer**, which reads like a
   non-projectivity problem rather than a rule problem. The asymmetry is forced by where each action
   leaves the node: RIGHT-PRED hangs the new slot off the spine **root**, which stays the root and
   can be given another slot later; RIGHT-COMP buries `r` one level down, so `r` never gets a second
   chance. The same reasoning explains INSERT's asymmetry, which the paper *does* state.

**The gate that catches both is a round-trip at 100 %**, not a metric. Every projective sentence's
oracle derivation is replayed through the rules and must reproduce the **labelled** gold tree
exactly. `scripts/lc_transitions.py <file>` runs it; anything below 100 % of projective sentences
means the oracle is wrong, never that the treebank is hard.

## What the system reproduces, and what it costs

Depth is **flat in length** for branching structures and **linear in embedding depth**:

| structure | length 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| right-branching | 2 | 2 | 2 | 2 |
| left-branching | 1 | 1 | 1 | 1 |

| centre-embedded | ×1 | ×2 | ×3 | ×4 | ×5 |
|---|---|---|---|---|---|
| depth | 1 | 2 | 3 | 4 | 5 |

On SUD English EWT+GUM train: **100.00 % of projective sentences derive and round-trip**, and
**99.34 % of configurations sit at depth ≤ 3** — Noji & Miyao's central claim, reproduced.

**The cost is non-projectivity: 6.29 % of train and 5.43 % of dev+test are outside the system** and
are dropped, as in the paper. ⚠ **Do not reach for pseudo-projective encoding to recover them.**
`NEGATIVE-RESULTS.md` already indicts that representation on accuracy grounds, and here it would be
worse than useless: the entire point of this arm is that stack depth *means* something, and a `||`
composite label turns the depth of a discontinuity into an artefact of the encoding.

## Labels are committed when a slot is opened

Every arc is labelled exactly once, by whichever reduce action puts the dependent into a slot:
LEFT-PRED and LEFT-COMP label the subtree they park as a left dependent, RIGHT-PRED labels the spine
root's next right dependent, RIGHT-COMP labels the one right dependent its own subtree has left.
Shift-type actions carry the word instead. This is not a convenience: a grammatical function is
committed **as soon as the slot is opened**, which is the incremental commitment a reading-time
model wants, rather than retroactively when the two ends meet.

## The model

Generative, so a prefix probability exists. A derivation is `w_1, r_1, w_2, ..., r_{n-1}, w_n`;
word steps choose a shift type then a word, reduce steps a name then a label. An LSTM over the
emitted history, plus the top two spines read fresh, feeds four heads. Output word embeddings are
tied to the input table — the point of tying being that self-training later adds vocabulary to
**one** table rather than two.

**The beam needs no resynchronisation.** Because shift and reduce alternate strictly, after *k*
words every hypothesis has taken exactly `2k-1` actions, so the beam is word-synchronous for free.
An RNNG needs Stern et al.'s machinery precisely because it lacks this property.

    surprisal(w_t) = -log2 [ Σ_i ρ_i Σ_r P(r|z_i) P(w_t|z_ir) ] + log2 Σ_i ρ_i

Mass lost to pruning inflates surprisal, so the reported figure is an **upper bound** that tightens
as the beam widens. It does, monotonically, and the size of the effect is the point (150 dev
sentences, gold-only model):

| beam | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| mean surprisal (bits) | 24.65 | 11.90 | 9.56 | 8.93 | 8.71 |
| UAS | 38.83 | 54.96 | 67.42 | 72.44 | 76.22 |
| LAS | 27.81 | 45.04 | 60.06 | 65.42 | 69.32 |

**Greedy decoding is not an option here, and that is the sharp contrast with the beam results in
`NEGATIVE-RESULTS.md`.** There, a beam bought nothing because the model was either non-autoregressive
(the zh segmenter, where per-position argmax already *is* the exact global MAP) or was scoring the
decisive actions low (Latin, where the beam emitted FEWER crossing arcs than greedy). Here the beam
is not searching for a better single path, it is **accumulating a marginal**, and beam 1 is a
one-derivation estimate of a sum over derivations — hence 24.65 bits against 8.71. Widening the beam
is buying probability mass, not search.

⚠ **The perplexities in `train_en_lc.log` are gold-path numbers** over the single oracle derivation.
They are an upper bound on the model's own surprisal, which marginalises over the beam. They are for
comparing training runs. Never quote one as the psycholinguistic quantity.

## Results

Two arms, both beam 10, on the SUD English EWT+GUM test set (3 014 sentences, 43 403 words):

| arm | vocab | dev OOV | UAS / LAS, projective only | UAS / LAS, all sentences |
|---|---|---|---|---|
| `lc_en_ewtgum` — gold only, 18 991 sentences | 12 458 | 7.03 % | 75.39 / 68.32 | 74.59 / 67.41 |
| `lc_en_selftrained` — + 120 000 silver, then fine-tuned on gold | 28 599 | 4.96 % | 77.11 / 70.78 | **76.32 / 69.88** |

**Report the all-sentences column.** Scoring only the projective gold sentences excludes exactly the
arcs this transition system cannot emit, and it flatters the parser by **0.80 UAS / 0.91 LAS** —
smaller than the 5.43 % non-projective rate suggests, because a non-projective sentence is mostly
projective arcs and the parser gets those. The self-training gain is the same either way
(+1.73 UAS / +2.47 LAS).

⚠ **About 2 % of test sentences finish with no terminal hypothesis on the beam** (42 540 of 43 403
arcs scored). Those sentences still yield surprisal and memory rows — the per-word frontier is
computed whether or not the derivation ever completes — but they contribute no parse. A per-word
metric cannot see this, which is why the arc counts are printed alongside the percentages.

Self-training is worth **+1.72 UAS / +2.46 LAS**, and it is bought with vocabulary rather than with
syntax: the silver trees come from a parser trained on the same gold treebank, so they contain no
construction the gold did not already have.

### Provo (84 readers, 55 passages, 61 072 rows after skips and complete cases)

ΔAIC against a baseline of word length, log frequency, cloze predictability, position and
sub-token count, all models on identical rows. More negative is better.

| measure | gold-only: surprisal / memory | self-trained: surprisal / memory |
|---|---|---|
| first fixation | −9.7 / −172.4 | −10.9 / **−181.0** |
| gaze duration | −4.5 / −594.8 | −12.4 / **−699.8** |
| go-past | −9.7 / −733.9 | −3.2 / **−923.4** |
| total time | −5.8 / −432.4 | −23.6 / **−547.6** |

**The memory predictors carry this arm and surprisal does not.** Stack depth and its companions beat
the baseline by 181 to 923 AIC on every measure, and improve with self-training on every measure.
Surprisal manages 3 to 24, and **the sign of its own coefficient is unstable across measures** —
positive for first fixation, negative for gaze and total. Read that as null, not as an inverse
effect.

### ⚠ Part of the surprisal effect is beam-truncation artefact

Widening the beam TIGHTENS the surprisal estimate and SHRINKS its reading-time effect
(`lc_en_selftrained`, identical rows):

| | mean surprisal | gaze ΔAIC | total ΔAIC | t surprisal (total) |
|---|---|---|---|---|
| beam 10 | 12.218 bits | −12.4 | −23.6 | −1.61 |
| beam 30 | 11.695 bits | −4.1 | −11.9 | −0.36 |

The mechanism is not noise attenuation, which would predict the opposite. A narrow beam loses more
probability mass on structurally hard sentences, so beam-10 "surprisal" is **partly a difficulty
measure in disguise** — which is what the memory predictors measure directly, and better. Any
surprisal figure from this arm must therefore name its beam width, and a beam chosen for speed will
overstate the effect. **Report at beam 30 or wider.**

⚠ **The model is an order of magnitude out of domain on Provo**: 12.2 bits/word against 8.87 on the
in-domain test set, i.e. roughly perplexity 4 700 against 470. Published Provo surprisal work uses
language models in the 50–150 range. Anything read off this arm's surprisal is read off a model
that does not know the register, and the self-training text (Gutenberg fiction) does not match
Provo's news and popular science.

### Eliminating the `<unk>` class DOES produce a surprisal effect

The prediction in the next section was WRONG, and the way it was wrong is the useful part. Replacing
the lumped unknown-word class with a character fallback (`CharDecoder`, `lc_en_char.pt`) — so
P(w) = P(`<unk>` | state) x P(spelling | state) and every rare word gets its own probability —
turns a null, sign-unstable surprisal term into a textbook one. Both arms at **beam 30**, identical
rows, same baseline:

| measure | closed ΔAIC | open ΔAIC | closed ms/bit (t) | open ms/bit (t) |
|---|---|---|---|---|
| first fixation | −14.8 | −9.4 | +0.24 (3.55) | +0.22 (3.62) |
| gaze duration | −4.1 | **−113.5** | **−0.21 (−2.10)** | **+0.94 (10.49)** |
| go-past | −5.6 | **−73.4** | +0.23 (1.58) | **+1.07 (8.35)** |
| total time | −11.9 | **−82.8** | **−0.04 (−0.36)** | **+0.99 (8.82)** |

**The sign is the result, not the ΔAIC.** The closed-vocabulary arm had two of four coefficients
pointing the wrong way — higher surprisal predicting FASTER reading. The open-vocabulary arm is
positive on all four at about 1 ms per bit, which is the range the reading-time literature reports.
About 5 % of tokens shared one probability, and those are precisely the high-surprisal words the
effect lives on; pricing them individually did not merely sharpen the estimate, it corrected it.

Two things go the other way and both belong in any write-up:

- **The memory predictors WEAKEN** (gaze −735.0 → −628.5, go-past −1049.3 → −709.3). Some of what
  they were absorbing was mis-estimated lexical surprisal — consistent with the beam-truncation
  finding above, where a badly estimated surprisal behaves as a difficulty proxy. Memory still
  dominates in absolute terms.
- **Parsing costs about a point.** At MATCHED beam 30, all sentences: closed 80.20 UAS / 74.57 LAS,
  open 79.56 / 73.65. ⚠ An earlier comparison appeared to show the open arm winning by 3 UAS; that
  was the open arm at beam 30 against the closed at beam 10, and it is entirely beam artefact.
  **Never compare two arms of this parser at different beam widths.** Part of the remaining 0.9 LAS
  is likely training length: the open arm's silver stage early-stopped at epoch 2, the closed at 5.

### Training the character model on in-vocabulary words does NOT help

An auxiliary task spelling a 25 % sample of in-vocabulary words, so the character model sees more
than the 1 % (silver) / 5 % (gold) out-of-vocabulary tail. Measured on the gold split at the real
vocabulary, best of 8 epochs:

| auxiliary | ppl_joint | char bits/word |
|---|---|---|
| none | **1325.4** | **1.557** |
| 25 % at weight 0.3 | 1323.9 | 1.560 |
| 25 % at weight 1.0 | 1340.7 | 1.598 |

A gentle weight is a wash and full weight is worse on both. The auxiliary objective competes for the
shared parser state faster than it improves orthography. **Two cheap proxies pointed the wrong way
before this settled it**: bits/char on curated dictionary words (the real tail is `graae`,
`succesfull`, `c'est`), and bits/char with the decoder evaluated at a FIXED ZERO STATE, which cannot
see capacity competition at all. Validate an auxiliary task on the quantity it is supposed to
improve, at the scale it will run at.

### Static vectors will not fix this (predicted, not yet measured)

⚠ **This prediction was made before the character fallback was built, and its premise — that a
better estimate means a smaller effect — was refuted by that result above.** It is kept because the
reasoning is still the right shape and the measurement was never made; treat it as an open question,
not a finding. The natural next move is to initialise the tied word table from `en_core_web_lg`: pretrained
embeddings buy perhaps 5–15 % perplexity on a small-data LSTM, which does not close a 4 700-to-150
gap, and a better estimate is currently associated with a *smaller* effect. `sud-md-static-vectors`
already measured fastText md on yue/id/ko at +0.2–0.9 LAS — inside seed noise, at 9–16× model size.
The levers that address the actual gap are more domain-matched text, and replacing the `<unk>` class
with a character or subword fallback so the ~5 % of words that should carry the highest surprisal
stop sharing one lumped probability.

Two further findings that a headline ΔAIC would hide:

- **The structural half of surprisal contributes essentially nothing.** `surprisal` and
  `surprisal_lex` differ only in whether the reduce action's probability is included, they correlate
  at **r = 0.998**, and they produce the same ΔAIC to within a point everywhere. So the quantity
  doing the (small) work is a language model's surprisal, not a parser's.
- **Self-training buys the SPILLOVER term, which is where the effect belongs.** The lagged
  coefficient `prev_surprisal` goes from unstable (t = 1.92, 0.40, −0.37, −0.26) to robustly
  positive on all four measures (t = 3.05, 2.58, 2.45, **5.04**). Reading time on a word is driven
  by the previous word's surprisal, which is the standard finding; the gold-only model was too weak
  a language model to show it.

## There is NO parsing-versus-surprisal trade-off — the two objectives are complementary

`--lex-weight` scales the LEXICAL half of the objective (shift type, word, spelling) against the
structural half (reduce action, label). It changes training pressure only: the model still
normalises over the whole vocabulary at every setting, so surprisal stays well defined and the
comparison is clean. All arms fine-tune from the SAME silver checkpoint; test at beam 30, Provo at
beam 30, identical rows.

| lex weight | LAS (all sents) | bits/word | ΔAIC gaze | ΔAIC total | ms/bit |
|---|---|---|---|---|---|
| 0 (parser only) | 66.55 | 14.33 | −70.0 | −40.1 | 0.89 |
| 0.1 | 72.40 | 13.56 | −100.0 | −62.5 | 0.90 |
| 0.3 | 73.16 | 13.46 | −110.0 | −62.8 | 0.98 |
| **1.0 (joint)** | **73.65** | 13.56 | −113.5 | −82.8 | 0.94 |
| 3.0 | 73.61 | 13.55 | −119.1 | **−92.8** | 0.95 |

**Optimising for parsing costs BOTH.** Dropping the lexical objective loses 7.10 LAS *and* takes the
surprisal fit from −113.5 to −70.0. Parsing peaks at the joint setting and is flat above it; there
is no setting at which trading surprisal away buys a better parser. Two mechanisms plausibly share
the credit and this experiment does not separate them: the beam prunes using word probabilities, so
a degraded language model degrades the SEARCH; and next-word prediction is itself a strong training
signal for the shared state, because you cannot predict the next word without encoding the syntax.
⚠ The control that would separate them — decode a lex-weight-0 checkpoint with
`--lex-decode-weight 0`, ranking on structural scores alone — is written (`run_control.sh`) and NOT
YET RUN. Do not attribute the effect to representation learning until it has been.

**The per-bit effect size is FLAT at 0.9-1.0 ms across every setting.** What degrades is the
PRECISION of the surprisal estimate, not the size of the reading-time effect. A ΔAIC column read
without the coefficient column would have suggested the effect itself was shrinking, and it is not.

## ⚠ The gold-only and self-trained models cannot be compared by perplexity

Self-training enlarges the vocabulary (dev OOV 7.03 % -> 4.96 %), and that makes the two models'
perplexities **measure different things**. The gold-only model is not a language model over English;
it is a model over `gold vocab + <unk>`, and it collects the whole probability of every rare word
into one class it is very good at predicting. The self-trained model has to name those words. So the
LARGER-VOCABULARY MODEL IS PENALISED FOR KNOWING MORE, and a perplexity comparison run naively will
report the better model as worse. Nothing in the training log says so; both curves look normal.

The two comparisons that stay valid are the ones used here:

- **UAS / LAS from the Viterbi derivation**, which does not involve the word distribution at all;
- **the reading-time fit**, where surprisal is only a predictor and the question is which model's
  predictor explains more variance -- a question that does not care how either was normalised.

A vocabulary-matched perplexity (map every out-of-gold-vocab word to `<unk>` for BOTH models and
renormalise) is computable, but it answers a question about the gold vocabulary rather than about
English, so it is not reported as the headline.

## Traps this arm has already paid for

- **`training_en_gum_sud` is the wrong arm for silver trees; `training_en_gum_ext` is the right
  one.** They differ by the whole extended-scope relabelling, so their **label inventories differ**,
  and silver trees built from the wrong one carry labels this model never saw — which trains
  silently. CLAUDE.md hazard 2, in a new place. Only `ROOT` → `root` needs renaming across the
  correct pair.
- **Gutenberg marks italics with underscores**, and they survive every cleaning step to become
  literal `_` **tokens** — which CoNLL-U reads as its own empty marker and spaCy keeps as a literal
  string. This is the Telugu lemma trap (`docs/dravidian.md`), in the FORM column.
- **A Vocab pickled from a `__main__` script cannot be unpickled from any other entry point.** The
  data files carry plain lists and the class is rebuilt on load.
- **A warm start must share the vocabulary object, not merely an equal set.** Both stages read one
  `lc_features.py` run, and `--init-from` **refuses** a mismatch rather than warning, because a
  scrambled tied embedding table does not fail — it merely converges worse and looks like a bad seed
  (CLAUDE.md hazard 7).
- **In the reading-time regression, every model must be fitted on identical rows.** Each predictor
  set drops a different set of rows to missing values, and AIC then falls simply because *n* fell.
  The frame is completed once, across the union of all predictors, before any fit.
- **The frequency table and its lookups must normalise identically.** Provo's word units keep their
  punctuation attached ("chased,"); a frequency table built over stripped tokens then gives every
  such word the unseen-word floor. That weakens the baseline for about a fifth of the data, and a
  weakened baseline **inflates whatever surprisal appears to add over it** — the exact direction
  that flatters the result. Nothing reports it: the fit converges and every coefficient still has a
  *t*. This one was found by reading the code, not by any diagnostic.
- **A skipped word is not a zero-millisecond word.** Skipping is itself predicted by surprisal, so
  scoring skips as 0 ms manufactures an enormous fake reading-time effect. Skips are dropped.
