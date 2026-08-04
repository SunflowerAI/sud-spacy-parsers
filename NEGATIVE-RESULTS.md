# NEGATIVE-RESULTS.md

Measured dead ends, extracted from `CLAUDE.md` so the main guide stays short. **Read the relevant
entry before retrying anything here** — each one cost real compute, and several look obviously
right until measured. Entries are grouped by area; each says what was tried, what it scored, why it
failed, and (where it applies) what would justify revisiting it.

---

## Tokenisation & segmentation

**en tokeniser tweaks.** `Tokenizer.v1` already matches EWT at the rule ceiling (token F1 0.991).
Hyphen and slash tweaks both regress — EWT is internally inconsistent about them, so there is no
rule to fit. Leave `en` alone.

**Decode-time lexicons never help, in either language.** Sanskrit beam rescoring −0.40 PM; greedy +
lexicon repair ±0.00 in domain and −0.80 on unseen text (it fires 2× per 250 sentences); Chinese
beam search loses at every setting — greedy 0.8950, pure beam 0.8935, worse with span bonuses. Pure
beam losing *slightly* is the expected sanity check: the model is non-autoregressive
(`Embed → residual(expand_window + Maxout) → Softmax`, no recurrence, no transition matrix), so
per-position argmax **is** the exact global MAP and a beam can only match it or lose to truncation.
**97 % of the zh segmenter's errors are confident errors** (margin ≥ 0.10), which no decoder
reaches; a transition model would touch 18 errors in 300 sentences, and the word-length distribution
it would encode is already matched to within half a point at every length.

**Sanskrit CSLiser lexicon reranking (the TransLIST idea).** A beam decoder scoring completed words
against a harvested lexicon looked good on the *weak* Vedic-only model (+0.67 split-location F,
+1.88 sentence PM) and collapsed once the model was trained on Vedic + DCS: +0.40 PM in domain,
+1.30 on under-represented Vedic, **−0.32 on genuinely unseen text** — i.e. it never helped with
novel vocabulary, the one thing a lexicon is for. It propped up an under-represented *domain*,
better fixed with training data. Coverage was not the problem (77.4 % of the unseen text's word
tokens were in the lexicon vs 81 % in domain). Costs avoided: a ~170 k-entry table in the wheel,
~100× slower decoding, and three hyperparameters whose correct values track model strength —
weights tuned for the weak model cost the strong one **−3.80 PM** until retuned.

**A morphological tier on that lexicon** (stem + ending decomposition, 42 440 stems / 1 863 endings)
added +0.10 / +0.04 / +0.12 PM — noise. With 193 k training sentences the character model has
already internalised the regularities the table encodes. NB an early sweep appeared to show the tier
was free because `morph_frac` was never plumbed through `decode`, so the rows were duplicates —
**wire a parameter through before believing an ablation of it.**

**Graded frequency loses to binary lexicon membership** (zh segmenter), across four encodings:

    binary membership (shipped)     0.8902
    count-band, no jackknife        0.8482
    count-band + jackknife          0.8252   banding flips on decade boundaries: jackknifing moves
                                             14.4 % of positions vs 1.8 % for membership
    rank-band + jackknife           0.7602   rank is subsample-stable but 5 bands over 17 k types is
                                             nearly constant — optimised the wrong property
    closed-class-only membership    not trained: enrichment 1.32× vs 1.45×, and it fires at only 33 %
                                    of real breaks (most zh boundaries are open-class)

Three explanations were each refuted by the next measurement — max-over-lengths collapse (the band
marginals are unimodal-centred, 8/22/36/21/12 %), rank stability, and band lopsidedness (all 25
codes used, top three carry 29.6 %). Binary membership IS nearly vacuous (1.46× enrichment, code 3
at 64 % of non-breaks) and something better must exist, but five attempts found only the simplest
one. **Stop here unless a new idea arrives; do not re-run the encoding search.** (The idea that did
arrive was jieba's *decision* rather than its dictionary — see CLAUDE.md.)

**Pre-correcting jieba buys nothing as a feature.** A force-split userdict harvested from train
(`del_word`) lifts jieba *as a standalone segmenter* from token F 0.7989 to **0.8570**, halving the
positions where jieba is wrong and the model right (1369 → 958) while keeping every rescue. As a
channel it scores 0.9172 against the raw channel's 0.9186 — a wash, so a 5.8-point better source
bought exactly nothing. Same lesson as the decode-time lexicons: hand a noisy source over **raw**
and let the model weight it. Correcting it first collapses the distinction between "jieba
confidently merged this" and "jieba split it", which is what the model was learning from.

**`add_word` is the wrong direction for jieba.** It under-splits relative to GSD, so adding the 636
GSD word types its dictionary lacks *costs* 0.8 F (0.7989 → 0.7911). The words needing forced
separation are pronoun+classifier and light-noun compounds — 这个 这次 有人 因此 为什么 一名 那个 企业家.

**Apte's stems do not transfer to spaced input (−3.50 F).** As a CSLiser input feature the 128 872-stem
Apte lexicon (`build_apte_lexicon.py`, extracted from CDSL `csl-orig` AP90 + AP; Apte cites
NOMINATIVE SINGULARS, so the SLP1 visarga/anusvara endings are stripped to recover the stem a
compound actually uses) with the inflection/sandhi-aware suffix strip (`sa_inflect_feature.py` —
direct lookup recovers only 35.7 % of running Vedic forms, stripping a 500-entry ending inventory
first reaches 84.8 %, and it never materialises stem × ending; the stem must be reconstructed with
the ending's OWN strip, since blanket-truncating gives a set matching almost anything, 1.2×
enrichment vs 2.92×) is worth +3.50 F on **continuous
saṃhitā** (89.08 → 92.58 dev, against a zeroed-channel capacity control). The released CSLiser is
`sa_presegment_ortho`, trained on **spaced** IAST/Devanagari, and the feature answers "where does a
word END" — which spaces already mark. +0.64 there, negative on the regime the wheel actually sees.
`models/apte_stems.txt` and `models/sa_endings.json` are kept for the continuous case.

**An inference-time prime for the zh batching gap** (prepending a throwaway `。` chunk) did recover
the whole 0.27 F, but was discarded once the real cause was found — `build_lex_model` had dropped
the `pad` argument. Padding reaches the same place without the hack and makes batched and per-text
scoring agree exactly, which priming does not. Kept only as evidence the diagnosis was right.

**Neither published Sanskrit segmenter can be called from this pipeline.** TransLIST is pinned to
Python 3.7.3 / PyTorch 1.5.0 / CUDA 9.2, requires patching fastNLP's source, and scrapes the
Sanskrit Heritage Reader at inference. ByT5-Sanskrit (`chronbmm/sanskrit5-multitask`, ~555 MB) does
run locally under `transformers`. But **both emit a segmented word list, not CSL** — no elision
markers, no coalescence marks, and TransLIST works in SLP not IAST. They solve "where are the
breaks", not "which two vowels fused", so neither can drive `desandhi_csl`.

---

## Encoders, affixes and model architecture

**Do NOT widen sa lexeme-level `PREFIX`/`SUFFIX` (costs 2.9 LAS).** `PREFIX`/`SUFFIX` are plain
entries in `lex_attr_getters` (`spacy/lang/lex_attrs.py`, `string[0]` and `string[-3:]`), so a
language *may* widen them by overriding `Sanskrit.Defaults`. Widening to PREFIX 3 / SUFFIX 6
regressed everything but the tagger: **LAS −2.9, morph_acc −3.8, lemma −3.7** (tag +0.16). The width
was sized from the **form→lemma edit** (SUFFIX 6 covers 96.9 % of it vs 80.8 % at 3) — what a
*lemmatiser* needs — but three of the four components want the short, widely-**shared** inflectional
ending, and at 6 characters in a language whose median word is 5–6 the suffix is near
word-identity (23 433 types vs NORM's 32 854; 64 % of tokens have the whole word as their "suffix"),
so it memorises instead of generalising. Sizing compounded it: 2000 rows for 23 433 types in the
small encoders = 11.7× collision, against 1.9× at length 3. **Affix width is a lexeme attribute, so
all components in a language share one value.** It is also NOT a runtime knob — a model trained at
one width and loaded at another degrades silently, with nothing in the config to catch it. The
override and the `SA_PREFIX_LEN`/`SA_SUFFIX_LEN` knobs were removed from `scripts/sa_tokenizer.py`;
only a warning comment remains. NB the two arms differed in BOTH widths, so prefix vs suffix blame
is not separated; a single-variable run (suffix 4, prefix 1) is the informative one if revisited.

**Do NOT put `sud.MultiHashEmbedAffix.v1` on the shared base `tok2vec`.** Single-variable run
(suffix 5 / 8 000 rows added to `config_sa_mwt.cfg`'s base embed, lexeme `SUFFIX` left at 3):
**tag_acc +0.36 but UAS −0.81 and LAS −0.49**. That is the lexeme-widening result reproducing in
miniature — milder because it is per-component and suffix-only, same direction. Reading: a longer
suffix window helps components predicting WORD-level properties (tagger, morphologiser, lemmatiser)
and hurts the parser, which wants the short shared ending as a generalisation cue. This was the
clean single-variable base experiment the entry above asked for; the answer is no. (The layer IS a
win on the *dedicated* encoders — see CLAUDE.md.)

**More rows and longer windows are both worse** for the affix layer: suffix 5 at 8 000 rows beats
the same window at 16 000, and beats suffix 6 at 24 000. The cheapest good configuration wins.

**The affix layer hurts `sud_unsandhi`** (0.9748 vs 0.9788, −0.40) even though it helps the real
lemmatiser by +1.43. Sandhi reversal is a *final-character* alternation (-ṃ/-m, -ḥ/-s, -o/-aḥ)
already covered by the default 3-character suffix, whereas lemmatisation edits the stem and wants
more lexical identity. Ship `sud_unsandhi` without it.

**A curated inventory of real Sanskrit endings loses to raw window length.** Simulated as a
longest-match lookup, 92–243 entries score 47–55 % exact-bundle against plain `form[-3:]`'s 60.0 %;
~630 entries are needed just to draw level, ~12 000 to match `form[-5:]`. The signal is window
LENGTH, not linguistic curation — real surface forms carry stem-class and sandhi cues in the
pre-desinential characters that a clean morpheme list discards.

**`annotating_components = ["morphologizer"]` on the sa lemma config** (so the lemmatizer conditions
on predicted FEATS) is not worth it: lemma_acc 0.8627 with vs 0.8645 without. Predicted `Case` at
F 0.856 adds about as much noise as signal to an edit-tree classifier that already has the whole
form in `NORM`. Left at `[]`, matching the other ten arms.

**Morphologiser co-training is dominated.** Verified on id: standalone-frozen 92.8 vs
listener-on-frozen-encoder 92.2 (the XPOS-orthogonality penalty) vs co-train 92.95 **but LAS −0.3 /
TAG −0.5**. No UPOS gain worth having, and it damages parsing — hence the freeze recipe with a
dedicated encoder.

**The tree-aware encoder for `Reported` is well-behaved but unproductive.** `sud.HeadDepsTagger.v1`
(`sud_tagger.py`, `make_sud_config.py --tree --pool <mode>`) concatenates `[own | head | pooled
dependents]` to give the model the tree neighbourhood the rule reads. Ar dev F, `--structural` =
0.51 for reference:

    pool=none    (no tree info, diagnostic)   0.4505
    pool=closed  + detach                     0.4106   P 0.478  R 0.360
    pool=closed2 + detach                     0.4041   P 0.416  R 0.393
    pool=deps    + detach                     0.2023
    pool=deps2   + detach                     0.0948   P 0.053  R 0.460
    pool=deps    (gradients propagated)       0.0940
    whole subtree, propagated                 0.0950   (peaked 9.5 at step 200, decayed to ~4)
    whole subtree, stop-gradient              0.0770

**The `pool = none` diagnostic is what makes this readable** — it sets head-to-self and pools
nothing, so it must reproduce the plain encoder, and at 0.4505 it does. Every deficit therefore
comes from the pooled information, not the plumbing. (An earlier note blamed the wrapper; that was
wrong. **Always run the null-pool diagnostic before reading such an ablation.**) Two findings:
restricting the pool to CLOSED-CLASS dependents is decisive (quotation marks and discourse markers
are exactly what the rule reads; averaging in open-class clause content drowns them), and a second
level is free when closed-class-restricted and catastrophic without it. `detach` matters (a token
heading many dependents otherwise accumulates one gradient per dependent) but is not the whole
story: even as read-only context the pool costs 0.45 → 0.41, so the residual is capacity — Arabic
has **811 positive training instances** and tripling the Softmax input to 192 dims dilutes them
faster than the structural signal repays. Nothing ships with it. Worth revisiting only on a language
with far more positives, where `pool=closed` is the variant to try.

**The plain added-layer encoder is the wrong one for `Reported`** (F 0.12–0.40, recall-limited).
The standard dedicated `HashEmbedCNN` over NORM/PREFIX/SUFFIX/SHAPE, window 1 / depth 3, has a ±3
receptive field — right for `Subject` (the raising complement sits *next to* its control verb, F
0.72–0.92) and wrong here, where every cue is non-local. See `--structural` in CLAUDE.md.

---

## LLM relabelling

**The LLM pass over the remaining `udep` residue is not reliable enough to use.**
`relabel_residue.py` offered a CONSTRAINED multi-way choice (candidates = relations the treebank
attests for that signature, answered as a DIGIT since SUD labels contain `:` and `@`). It produced
21 353 decisions and **none were applied**:

    self-consistency, option ORDER shuffled, qwen3:8b, definitions only     75.3 %
    ... + one contrastive example per label, harvested from the treebank    76.7 %
    ... same prompt, gemma4                                                 68.0 %
    pass 1 vs pass 2 on the SAME 2467 en tokens (different option SETS)     36.4 %

Prompt engineering moved the label DISTRIBUTION enormously (en `comp:obl` 26 % → 55 % when
`comp:obl` was added as a guaranteed option; `compound` 3 % → 17 % when examples were added) and
consistency not at all. gemma4 and qwen3 produce nearly INVERTED majority labels on identical
prompts (`mod` 60 % vs 19 %; `comp:obl` 11 % vs 49 %). The comp/mod pipeline's few-shot success does
not transfer because that was a BINARY choice with a rule-built gold to select prompts against; here
there is no gold, so prompts can only be judged by consistency, and consistency does not move.
Caches archived under `archive_residue_pass{1,2}/`; the script is kept so this stays reproducible.

**Few-shot composition only slides the precision/recall frontier**; it cannot beat a good prompt.
English plateaus at ~0.91–0.93 on qwen3:8b, and the gains there came from **auditing the gold**, not
from more examples.

**Sanskrit comp/mod is at chance and stays un-relabelled.** `scripts/ufal_compmod_probe.py`
confirmed on classical UFAL that the LLM scores 0.43 against a 0.82 majority baseline on the
case-marked Ins/Acc/Gen residue — same as Vedic. Structural: Sanskrit is case-based, not
prepositional.

**The sa `udep@<subtype>` extension is a real but insufficient fix.** Using the annotators' ten
semantic-role subtypes (`@instr/@goal/@lmod/@tmod/@source/@manner/@soc/@benef/@grad/@path`, ~8850
tokens the bare-`udep` bucket never touched) moved test-gp `comp:obl` F **0.352 → 0.396** over the
case-only ext arm with LAS flat — genuine signal the Case-only view missed, but still below the
un-relabelled base's 0.404, so **the released sa model stays un-relabelled**. Method, kept because
it generalises: `sa_subtype_audit.py` found only `@manner` has in-treebank commit evidence (626
`mod@manner` / 0 `comp:obl@manner`); seven subtypes are dominated by cases already established as
circumstantial → mod; `goal`/`path` are >85 % headed by motion/placement/ritual-offering verbs →
comp:obl; `@soc` is a genuine ~54/46 mix → left to the LLM.

**`unk` is not a disambiguation target (audit-first negative).** `unk` is a second noncommittal
relation, distinct from `udep`, largest in ja (7519 train tokens) and ar (4111). Whole-train audit:
**99.2 %** of Japanese `unk` tokens are the bound continuation of an `Idiom=Yes`/`InIdiom=Yes`
periphrastic copula/auxiliary chain (である, てくれる, てくる…) — always adjacent to its head (99.4 %,
`id == head_id+1`), with the real relation carried entirely by the FIRST idiom token (1600 chains
are 3+ tokens, where the head is itself `unk`). Arabic splits 53 % the same idiom-chain pattern
(e.g. باسم, a complex preposition) / 47 % newswire dateline artifacts. In both languages `unk`
correctly marks "this token carries no independent grammatical relation" — there is no
comp:obl/mod-style choice being deferred, so relabelling would gain nothing.

**Typo correction, subject raising and reported speech were each ruled out as LLM targets**
(SUD-vs-UD survey). Typo (`Typo=Yes`/`CorrectForm=`) is a plain **UD** convention SUD leaves
untouched, and is fully gold-annotated wherever it appears (en EWT/GUM, some id/fa/la; zero in
zh/ko/ar/ja/la-ITTB/PROIEL/lzh/sa/yue). Subject raising (`Subject=SubjRaising`/`ObjRaising`) is
likewise already gold wherever the source treebank carries it (2000–7500+ in en/zh/id/fa/la; zero in
ar/ko/ja) — not ambiguous, hence not an LLM target, though it IS a good target for a trained output
layer (which is what `sud_tagger` became). Reported speech (`Reported=Yes`, `@reported`/`@rep`) is
real but vanishingly sparse — only 33 Latin instances (ITTB patristic scriptural citations); GUM's
richer `Discourse=attribution-*` layer (1650) is not in the training pipeline (`en` trains on EWT
only). Two claims in the original survey note were wrong and are corrected in CLAUDE.md:
`spacy convert` does **not** preserve MISC into the `Doc`, and every `_morph`/`_lemma` config does
have a `morphologizer`.

---

## Data balancing

**UFAL upsampling failed in three variants and is not worth retrying.** UFAL is 170 sentences /
1323 tokens against Vedic's 161 985. Across all-×5 (UFAL LAS 0.4032), ×612 duplication (killed after
3 h) and sampling to parity (0.4203), **UFAL LAS never moved more than ~0.4 from baseline while
Vedic swung 10+**. 170 sentences is too little to learn classical syntax from however often it is
shown. Duplicating docs also inflates the PARSER's workload — the expensive transition-based
component — 9.9×; `sud.SamplingCorpus.v1` (`sampling_corpus.py`) holds the syntactic token budget
FIXED instead and turned 200 steps in 20–168 min into 2000 steps in 5 min.

**md static vectors do not pay for themselves.** fastText `md` vectors tested on yue/id/ko: LAS
+0.2–0.9 (within seed noise), `comp:obl` F *hurt*, model 9–16× larger. Keep `sm`. Multi-seed runs
were essential to see this.

---

## Classical Chinese cross-unit relations

**An LLM binary on clause linking is below the majority baseline.** `cross_unit_bench.py` benchmarks
the one decision the derived rules leave open — is the following clause an ARGUMENT of the preceding
verb (`comp:obj`/`comp:obl`/`comp:pred`) or an INDEPENDENT following clause (`parataxis`/
`conj:coord`)? Gold is real: the in-unit clause links no rule covers, taken from dev+test, 2 714 of
them. qwen3:8b with definitions plus contrastive few-shot:

    majority baseline (complement)        55.9 %  (58.5 % on the sampled 200)
    qwen3:8b                              56.5 %

It answers `complement` 128 times in 200 against a true rate of 58.5 %, missing 49 of 83
`independent` items — a prior, not a discrimination. This is the Sanskrit comp/mod result again
(0.43 against a 0.82 majority), and it is why the baseline gets measured BEFORE the model: at 56.5 %
raw the number looks like a result until the constant is put beside it. The binary framing was not
the problem — the annotators' own in-unit split is only 56/44, so the task is genuinely
underdetermined where no particle marks it.

**spaCy cannot express an unknown head, so a "supervise only our chosen edges" corpus is not
buildable.** `ArcEagerGold` calls `example.get_aligned_parse()`, which reads `token.head.i` — and for
a token with no head that IS its own index, which arc-eager reads as ROOT. `has_head()` returns False
but nothing downstream consults it, and `Example.from_dict` with `heads=[0,0,None,2]` returns
`[0,0,2,2]` just the same. `HEAD_UNKNOWN` is reachable only through the ALIGNMENT path (a gold head
that fails to align 1:1 with a predicted token), i.e. via tokenisation mismatch, not annotation.
Built that way, 6.2 % of tokens were taught to be sentence roots and the parser collapsed to
**DEP_UAS 18.1 / DEP_LAS 16.9** (against ~79/74). Blanking only the LABEL does not help either:
`_replace_unseen_labels` rewrites an unseen label to the backoff `dep`, so the parser is simply
taught to emit `dep`. The supported equivalent is to make those boundaries SENTENCE BREAKS — with no
arc there is no loss to mask, and the unit root is then a genuine root.

**A parser cannot invent a relation, and asked to generalise it reproduces a known bias.** The
rules-only arm, which never saw an invented edge, chooses at unmarked cross-unit boundaries
(counts taken BEFORE the boundary-ownership fix below, so read the ranking, not the figures):
`comp:obj` 1 169 / ROOT 334 / `parataxis` 303 / `mod` 275 / `conj:coord` 159. Its output vocabulary
is fixed to the labels seen in training, so "better labels than we can write" was never available;
what it does is amplify the in-unit prior (`comp:obj` 37.0 %, `parataxis` 28.5 %, `conj:coord`
24.7 %). That prior is BIASED for this configuration — an editor does not put a comma inside a tight
complement — so a model generalising from in-unit evidence over-predicts complements exactly where a
mark argues against one. There is no gold, so this is not proof the parser is wrong; it is a reason
to distrust the transfer, and it is why the residue ships as sentence breaks rather than as either
our `parataxis` or its `comp:obj`.

## Meta-lessons worth more than the individual results

- **Always run a capacity control.** Adding a feature usually also adds parameters. The zeroed-channel
  control (zh lexicon, zh jieba, sa affix `w96`) is what separates "the feature works" from "the
  extra width works" — and in the sa affix case it showed the gain was the feature (Voice +17 vs
  +2.9) rather than the parameters.
- **A jackknifed feature is a different feature.** A corpus-harvested lexicon covers 100 % of train
  and 87.6 % of test, so naive it is *worse than useless* (below the zeroed control); jackknifed it
  is the single biggest zh win.
- **Compare within the same `n_sources`** — going 1 → 2 feature channels costs ~0.5–1.4 F on its own
  (char embed 56 → 48).
- **Hand noisy sources over raw.** Pre-correcting them (jieba userdict, lexicon repair) destroys the
  distinction the model was learning from.
- **Multi-seed or don't claim it.** zh raw LAS spreads ~6 points off a 0.22-point token-F spread;
  single-run comparisons produced at least one wrong claim in this repo's history.
- **Audit before building.** The `unk` and typo/raising/reported findings each avoided a whole
  gold/bench/relabel build.
- **Never compare numbers from two different harnesses.** The lzh merge was reported as costing
  ~7 LAS on within-unit edges; that came from scoring one arm through `clause_parser` and the other
  on whole merged sentences. Re-run through ONE harness, merging *helps* by ~3.9. Same species of
  error as the "+2.51 zh raw LAS" claim, and it survived longer because both numbers were individually
  correct — only the comparison was invalid.
- **Measure the majority baseline before the model, not after.** 56.5 % accuracy reads as a result
  until the 58.5 % constant sits next to it.
- **A round-trip test proves the CONTENT survived, not that it was filed correctly.**
  `align_kanripo_punct.py` placed every mark at the right character offset — text byte-identical,
  round-trip clean on all three splits — while assigning each boundary mark to the unit AFTER the
  one it closed. 2 780 sentence-final marks opened a unit and none closed one, so `sent_group` ran
  one unit late and every merged sentence was mis-segmented. Nothing raised, and the strongest check
  in the script was structurally blind to it. The measurements taken on that data were internally
  consistent and wrong: the punctuated gain held (+11.9 → +11.93) but the unpunctuated cost grew
  by 60 % (−1.34 → −2.16) and the idiom "improvement" (67.83) evaporated to parity (66.18). When a
  derived file has a NOTION OF OWNERSHIP, assert on it directly — no unit may open with a
  sentence-final mark — because the content-level check cannot see it.
- **"Missing" in an API is not missing in the code that consumes it.** `has_head()` says False while
  `get_aligned_parse` reads `token.head.i` and gets a root. Third instance in this repo after unset-
  vs-empty MORPH and the CoNLL-U `_` kept as a literal — when annotation is meant to be absent,
  verify what the CONSUMER sees, not what the setter reports.
