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
arrive was jieba's *decision* rather than its dictionary — see `docs/layers-and-tokenisers.md`.)

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
win on the *dedicated* encoders — see `docs/sanskrit.md`.)

**More rows and longer windows are both worse** for the affix layer: suffix 5 at 8 000 rows beats
the same window at 16 000, and beats suffix 6 at 24 000. The cheapest good configuration wins.

**The affix layer hurts `sud_unsandhi`** even though it helps the real lemmatiser by +1.43. Sandhi
reversal is a *final-character* alternation (-ṃ/-m, -ḥ/-s, -o/-aḥ) already covered by the default
3-character suffix, whereas lemmatisation edits the stem and wants more lexical identity. Ship
`sud_unsandhi` without it — `package_sud.sh` does, from `training_sa_mwt_unsandhi`.

⚠ **The size of that loss was understated 30-fold, because the test set was the wrong one**
(corrected 2026-08-16). The recorded figure, 0.9748 against 0.9788 (−0.40), came from the
DCS-dominated test, where most tokens sit INSIDE an MWT and are already written unsandhied — an
identity mapping, and the majority class. On the Vedic test, where the work actually is, `spacy
evaluate --gold-preproc` gives the plain arm **LEMMA 96.41** and the sfx5 arm **83.30**: −13.1, not
−0.4. `metrics_sa_mwt_unsandhi_sfx5_Vedic.json` was never taken, so the split that would have shown
this was the one split never measured — the entry above has `_DCS` and `_Vedic` files for the plain
arm and only the mixed file for sfx5. The decision was right; the number behind it was not. **A
mixed-domain average over a corpus that is 90 % one domain is not a measurement of the other 10 %**
— the same shape as reading a headline `morph_acc` for a rare label (standing hazard 6).

**A curated inventory of real Sanskrit endings loses to raw window length.** Simulated as a
longest-match lookup, 92–243 entries score 47–55 % exact-bundle against plain `form[-3:]`'s 60.0 %;
~630 entries are needed just to draw level, ~12 000 to match `form[-5:]`. The signal is window
LENGTH, not linguistic curation — real surface forms carry stem-class and sandhi cues in the
pre-desinential characters that a clean morpheme list discards.

**`annotating_components = ["morphologizer"]` on the sa lemma config** (so the lemmatizer conditions
on predicted FEATS) is not worth it: lemma_acc 0.8627 with vs 0.8645 without. Predicted `Case` at
F 0.856 adds about as much noise as signal to an edit-tree classifier that already has the whole
form in `NORM`. Left at `[]`, matching the other ten arms.

**Conditioning XPOS on UPOS+FEATS at the BOTTOM of the encoder costs 0.2-0.6 TAG.** ⚠ Read this
entry as being about the INJECTION POINT, not about the idea: injecting the same information at the
TOP, under the softmax, is a WIN of +0.05 to +0.48 in all nine languages tried and is described in
`docs/xpos.md`. What follows is why the bottom is the wrong place, and it cost three arms to learn. Every arm grew the same way -- base pipeline `[tok2vec, tagger,
parser]`, morphologiser added later as a frozen layer -- so the one component whose target is
largely a restatement of UPOS+FEATS is the only one that cannot see them, purely because of the
order the layers were built in. Fixing that looks obviously right and is not.
`make_xpos_config.py` moves the tagger to the END of the pipeline, behind the morphologiser, and
gives its own encoder `POS` and `MORPH` channels alongside the token embedding it already had
(explicit `MultiHashEmbed`, rows `[E, E/2, E/2, E/2]` reproducing `HashEmbedCNN` exactly, plus
POS 100 / MORPH 4000); `--no-cond` is the capacity control, identical minus the two channels.
Ordinary freeze recipe otherwise, so every other component comes out byte-identical.

    dev tag_acc     released   control   conditioned      test TAG   released  control  conditioned
    ar (346 tags)    0.8880    0.8873      0.8844         ar          89.44     89.28      88.67
    zh  (41 tags)    0.9072    0.9053      0.9016         zh          90.81     90.77      90.28
    en  (49 tags)    0.9287    0.9278      0.9243         en          93.09     92.90      92.73

The control is what makes this readable: the dedicated encoder is nearly free (-0.07 to -0.19 dev),
so the loss is the CONDITIONING, -0.29 to -0.37 dev and -0.17 to -0.61 test, same sign and
magnitude on three unrelated tagsets.

**Why, and it is the general lesson: an oracle measured on GOLD features says nothing about a
feature the model must PREDICT.** Majority-class maps fitted on train and scored on test say that
knowing gold UPOS+FEATS on top of the form is worth +19.6 XPOS points on ar, +14.2 zh, +13.8 la,
+13.2 en, +11.7 yue, +8.2 id, +4.3 ko, +4.1 fa -- and that ar and yue are all but deterministic
from UPOS+FEATS alone (99.9 / 100.0). Re-run the same maps against what the released arm actually
PREDICTS and the signal is below the tagger everywhere (`scripts/xpos_headroom.py --model`;
released tagger / map on gold / map on predicted, test):

    ar     89.45  94.00  86.81      ko     72.94  66.34  66.01
    zh     90.82  91.72  88.30      id     92.21  89.44  88.87
    en     93.13  96.41  92.13      lzh    92.27  97.31  91.89
    fa     96.20  97.47  95.86      yue    93.81  93.18  89.06
    en_gum 94.04  97.09  93.33      ja     95.16  92.21  91.64

Ten arms, no exceptions: gold -> predicted costs 3-7 points and lands the map BELOW the tagger it
was supposed to improve. Morphology is predicted at `morph_acc` 0.75-0.99 (exact-bundle), and its
errors fall on precisely the tokens the tagger also finds hard, so the channel is noise correlated
with the target. This is the same finding as the sa lemmatiser entry above, one component over.

**The per-FEATURE decomposition was then built and it does not rescue it either.** The obvious
objection to the above is that a single hashed embedding of the WHOLE bundle is the crudest way to
offer the information -- `Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` become unrelated symbols,
and an unseen bundle has no decomposition to fall back on. So `sud.MultiHashEmbedFeats.v1`
(`scripts/sud_feats_embed.py`, `make_xpos_config.py --feats`) gives each morphological category its
own hash-embedded table, `hash_string("Case=Nom")` per column and `hash_string("Case=")` where the
token has no value. `scripts/check_feats_embed.py` verifies it byte-for-byte against stock
`MultiHashEmbed` when no feature is configured, and confirms the decomposition holds and that an
UNSET morph and an EMPTY one land on the same row.

**`scripts/build_feats_inventory.py` picks the categories, and its most useful output is that three
languages have none.** It ranks each FEATS key by the information it carries about XPOS *once the
form is already known* -- the only question that matters, since the tagger reads the form anyway.
H(XPOS|form) is already 0.251 bits for zh, 0.089 for ko and 0.018 for id, and no category clears
0.02 bits in any of them: **their XPOS is a function of the spelling, so there is nothing to
condition on** and `train_xpos.sh` skips them rather than training dead channels. Where features do
clear the bar the list is small and sensible (ar Case/Number/Definite/Gender/AdpType/Mood, Case
alone worth 0.444 bits; en Number/PronType/VerbForm/Person/Tense/Mood/Degree -- exactly the PTB
VBD/VBN/VBP/VBZ and JJ/JJR/JJS distinctions; la Number/Case/Gender/InflClass/Aspect/PronType).

    dev tag_acc  released  control  bundle  per-feat  |  test TAG  released  control  bundle  per-feat
    ar             .8880    .8873   .8844    .8836    |  ar          89.44    89.28   88.67    88.76
    en             .9287    .9278   .9243    .9246    |  en          93.09    92.90   92.73    92.83
    zh             .9072    .9053   .9016      n/a    |  zh          90.81    90.77   90.28      n/a
    la             .8945    .8886     --      .8897   |  la          86.16    86.06     --      85.71
    lzh            .9206    .9185     --      .9194   |  lzh         92.59    92.14     --      92.44

Per-feature against the hashed bundle is a WASH (ar -0.08 / +0.09, en +0.03 / +0.10 dev/test), and
against its own matched control it is noise with no consistent sign (test: ar -0.52, en -0.07,
la -0.35, lzh +0.30). **Every arm is still below its released tagger.** So the bottleneck was never
how FEATS is represented -- it is that predicted morphology carries almost no information about
XPOS that the spelling does not already carry, once its own error rate is paid for.

⚠ **Single seed per arm, and init is unseeded.** la is the measured warning: its control (explicit
`MultiHashEmbed` + `MaxoutWindowEncoder`) scores .8886 against the shipping arm's .8945 while being
architecturally IDENTICAL to it -- so the spread on that arm is ~0.5, larger than every conditioning
delta in the table. Read the individual deltas as noise; what carries the result is that all four
languages and both variants fail to beat the released tagger, in the same direction, and that the
three languages with no informative feature were predicted in advance by the inventory.

**RESOLVED: it was the injection point.** Both arms above put the channels in the EMBED, so a
`MaxoutWindowEncoder` of depth 4 then convolves them over a +-4 token window -- each token's tag
comes to depend on its NEIGHBOURS' predicted morphology, and the token representation is rebuilt
from scratch instead of reusing the co-trained shared encoder. Move the identical information above
the encoder (`sud.Tok2VecPlusFeats.v1`: keep the released tagger's `Tok2VecListener` on the frozen
shared encoder, concatenate the morphology under the softmax) and it helps everywhere. See
`docs/xpos.md`, "XPOS conditioned on UPOS+FEATS". The lesson worth carrying: **where a noisy predicted feature enters
the network matters more than how it is represented** -- bundle vs per-feature was a wash (<= 0.10),
bottom vs top was ~0.7. A feature that is right 75-99 % of the time should reach the decision it
informs and nothing else; convolving it spreads its errors over every neighbour.

Still unreached: the either-one-right ceiling sits 2-6 points above the tagger (ar 91.88 v 89.45,
zh 93.98 v 90.82, en 95.76 v 93.13) and top injection recovers well under one point of it. Arms and
configs are kept:
`training_{ar,zh,en}_xposdown{,_ctl}`, `training_{ar,en,la,lzh}_xposfeat`,
`training_{la,lzh}_xposdown_ctl`, `scripts/{make_xpos_config.py,train_xpos.sh,eval_xpos.sh,`
`check_xpos_inputs.py,xpos_headroom.py,sud_feats_embed.py,check_feats_embed.py,`
`build_feats_inventory.py}`.

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
0.72–0.92) and wrong here, where every cue is non-local. See `--structural` in `docs/sud-misc-layer.md`.

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

- **A smoke test that pipes to `head` kills the run.** SIGPIPE truncated an arm at its best
  checkpoint, and `model-best == model-last` is the tell -- a patience-terminated run always differs,
  because it trains 1600 steps past its best. The truncated number was reported as converged and was
  6 points low.
- **`pgrep -f` matches the wait-loop's own command line.** Three wait-loops in one session spun
  forever against themselves, one of them wasting 37 minutes of an idle machine. Match on `comm`,
  chain the commands directly, or wait on a log marker.

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

## Trained `Idiom`/`InIdiom` for ja, lzh and sa (2026-08-14)

Training the idiom layer instead of deriving it by rule works in Arabic (+3.70 F) and **fails in
the other three languages that annotate idioms**. Measured on test, end-to-end, against the rule on
the same base:

    ja   Idiom 96.88 -> 96.71   InIdiom 96.32 -> 95.40
    lzh  Idiom 75.68 -> 76.05   InIdiom 85.54 -> 73.91
    sa   Idiom 76.99 -> 76.60   InIdiom 80.48 -> 81.30

Two factors predict it, and neither is sufficient alone. **ja has no headroom**: its parser
supplies both rule inputs (`ExtPos` + an `unk` dependent) on 95.7 % of gold idiom heads, so the
rule is already at 96.88. **lzh and sa have headroom (49.5 % / 40.8 %) but half ar's training
data** -- 1 050 and 838 idiom heads against ar's 1 993.

The lesson worth keeping is lzh's InIdiom, which loses **11.63 F**: the rule there is a TRANSITIVE
CLOSURE (walk consecutive `unk` links up to a head bearing `ExtPos`), not a local classification,
and it is exact given `unk`. A classifier must rediscover transitivity from the data. Where a rule
expresses a closure rather than a local decision, prefer the rule unless the data is abundant.

⚠ **THE lzh RULE FIGURES ABOVE ARE UNRECONCILED WITH THIS ENTRY'S OWN AVAILABILITY NUMBER, AND
75.68 IS THE SUSPECT ONE** (found 2026-08-17, chasing a supposed regression that turned out not to
be one). The rule is EXACT given its inputs, so the 49.5 % availability stated three lines up is a
recall ceiling, and at P = 100 % it implies

    F = 2(1.000)(0.495) / (1.000 + 0.495) = 66.2

Re-measured with `eval_sud_idiom.py`, the rule on the both-scripts arm gives **P 100.00 / R 49.45 /
F 66.18** -- this entry's availability figure reproduced to two decimals, and the F that follows
from it. For the headline 75.68 to hold at P 100 the recall would have to be ~60.9 %, contradicting
the 49.5 %. The likely cause is a different DENOMINATOR (the eval reports `gold=182`; a rule scored
only over the heads where it fires inflates exactly this way). **Do not set a target from 75.68 /
85.54 until it is reconciled** -- and note the trained-vs-rule DELTAS in the table may still be
sound even if the absolute figures are not, since both arms of each comparison share whatever
denominator was used.

Do not re-run this without more data; the arms are kept as `training_{ja,lzh,sa}_idiom/`.

---

## XPOS as a parser input, and kanripo vectors, for lzh (2026-08-16)

Four channels intended to give the lzh parser information it lacks. **None beat its capacity
control**, and three of them could have been ruled out before any training run. Baseline is
`training_lzh_trad` (traditional-only, punctuation-restored, rule-merged) on test with
`--gold-preproc`: TAG 92.59 / UAS 82.92 / LAS 77.20.

    arm                                  TAG     UAS     LAS   vs its control
    constant-channel control           92.46   82.58   77.10        --
    XPOS fields 1+2, per-form lexicon   92.59   82.36   76.91      -0.19
    whole 118-way tag, same route       92.50   82.96   77.33      +0.24
    predicted XPOS fields (tagger)      92.56   82.43   76.84      -0.25
    shuffled-vector control            92.63   82.42   77.04        --
    kanripo static vectors, 300d       92.82   82.98   77.50      +0.46   <- seed 0 ONLY

The whole spread is 0.49 LAS on one seed, and this arm family's seed spread is ~0.5. The vectors
row is the trap: **+0.46 on seed 0 became +0.04 mean over three seeds** (+0.46 / -0.13 / -0.20,
sd 0.29). Read no single-seed row in that table as a result, in either direction.

**1. A per-form lexicon carries ZERO information beyond `NORM`, and that is an identity.** A
majority-XPOS-per-form table is a deterministic function of the form, so conditioning on
`(form, f(form))` is conditioning on `form` -- which the parser already holds. Measured on test
with one plug-in estimator throughout, 34 233 tokens:

    H(deprel | form)                  1.1941 bits
    H(deprel | form, GOLD xpos)       0.9719   -> gold adds     0.2222 bits
    H(deprel | form, PREDICTED xpos)  1.0466   -> a tagger adds 0.1475 bits
    H(deprel | form, LEXICON fields)  1.1941   -> the lexicon adds 0.0000 bits

The useful part of XPOS is the WITHIN-FORM variation -- 58.8 % of lzh tokens have a form whose XPOS
field 2 varies (`H(XPOS field2 | form)` = 0.2808 bits) -- and a majority table destroys exactly
that. It was not obvious in advance: the lexicon looks strong on the metric that flatters it, a form
seen once or twice having its coarse field right 88.6 % of the time against 76.0 % for the whole
118-way tag. **A channel can be accurate and still carry nothing.**

**2. A feature predicted by a head that SHARES the encoder is already linearly present in every
other head's input.** `spacy.Tagger.v2` is a LINEAR softmax on the listener output; the parser's
first operation on the same listener output is also linear; `listener_map` shows both heads on the
one `tok2vec`. So XPOS is linearly decodable from the parser's own input at the tagger's own
accuracy, 92.59, by construction. Handing back a hashed embedding of the predicted tag gives the
first linear layer nothing it could not already extract -- which is why 0.1475 bits of genuinely
available information bought -0.25 LAS. The wiring was verified, not assumed: with
`annotating_components = ["xpos_prepass"]` the parser sees a tag on 100 % of tokens during
training, without it 0 %.

**3. Kanripo static vectors do not help parsing: +0.04 LAS mean over three seeds.** floret CBOW over
42 M tokens of Kanseki Repository text (~90x the treebank), trained on IDS-expanded strings so
subword n-grams share strength between graphically related characters, written out keyed by the
bare character. The channel has neither defect above: it is genuinely new information and it is
populated at 100 % of test tokens in every frequency slice. It still does not pay.

**The frequency-split probe says why, and it is the diagnostic worth reusing.** Predicting UPOS from
the vector for a HELD-OUT character, split by how often that character occurs in kanripo:

    kanripo freq   types   majority   vector
             1-5     289     47.40%   57.79%
            6-50     803     40.72%   60.27%
          51-500    1692     43.20%   62.77%
            >500    2715     45.86%   65.12%

At frequency 1-5 the vector has decayed to 57.79 %, which is the RADICAL probe's 57.00 to within
noise -- the distributional content is gone and only the graphic backoff remains. And that is
exactly the population the parser needed help with: treebank-unseen forms have a **median kanripo
frequency of 4**, with 59.2 % at five or fewer. So the vectors are informative where the parser
already copes and near-empty where it does not. Aggregate probe accuracy hides this completely
(63.30 % UPOS, 71.12 % XPOS field 2, both far above majority).

**96 dimensions do not preserve the aggregate** (PCA, 64.7 % variance retained, 6.8 MB against
20 MB): +0.06 LAS vs its own control, against 300d's seed-0 +0.46. One seed each, so the comparison
is weak, but there is nothing here to justify the wheel growing by 20 MB on a 12 MB base.

**4. The sub-character probe that preceded all of this, and which nobody should re-run.** Predicting
a held-out character's class from symbolic features, 5 472 types, logistic regression with the
regularisation swept and a bias-only null:

                     UPOS            XPOS field3 (44 classes)
    NULL (bias)      44.59%          33.33%
    radical          57.00%  +12.4   44.17%  +10.8
    IDS (depth 2)    55.30%  +10.7   39.42%   +6.1
    Qieyun           48.06%   +3.5   33.17%   -0.2
    all three        57.36%          42.69%

**The Kangxi radical (Unihan `kRSUnicode`, Unicode licence) beats full IDS (cjkvi-ids, GPLv2) and
adding IDS to it buys nothing**, so the licence question that would have blocked shipping never
arises. Qieyun (nk2028, CC0, 98.8 % token coverage) is far behind on lexical class and does not
stack. It was not built as a vector arm: it would be a third BACKOFF for the same starved
population, and the backoffs do not add. ⚠ An earlier version of this probe scored BELOW the null
because `cross_val_predict` used contiguous folds over codepoint-sorted characters, which groups
characters by Unicode block and therefore by radical. The null control caught it. **Run the null
before reading any ablation.**

**The corpus contained the evaluation text, and the treebank is why.** Kyoto was built FROM kanripo
(the sent_ids ARE kanripo ids, which is how `align_kanripo_punct.py` restored the punctuation), so
200 of 200 sampled test sentences appear verbatim as kanripo lines, and **127 of 279 treebank-unseen
types have their ONLY kanripo occurrence inside a test sentence**. Not label leakage, but the
vectors would have been fitted to the very text they were scored on, for exactly the population
under test. `make_leakfree_lzh_corpus.py` removes dev/test at a cost of 0.15 % of tokens; short
formulaic sentences (子曰, 何也) are exempt UNLESS they carry a form train never saw, which takes the
contaminated count to 160 of 279. All figures above are on the leak-free set.

⚠ **PRUNE VECTORS BY DIMENSION, NEVER BY VOCABULARY.** `vectors_lzh_apt96` in the aptness repo is
what the other mistake looks like: pruned to a training vocabulary, it covers **0 %** of
treebank-unseen forms and 83.0 % of test tokens overall, 93 % of the gap being punctuation it
predates. The rows are the value; the dimensions are the cost. Building the leak-free set hits the
same trap from the other side -- removing dev/test drops 577 treebank types out of the corpus
vocabulary, so `build_lzh_vectors.py --extra-types` emits their rows anyway (composed from subwords,
using no held-out text).

**Two pre-flight checks that cost minutes and would have killed three of these four.**

1. **Is the channel a function of something the component already reads?** If so it carries zero
   bits, whatever its accuracy. One conditional-entropy calculation.
2. **Does a head sharing the encoder already predict it?** If so it is linearly present in the
   input. One look at the model tree and `listener_map`.

And one that explains the fourth: **split any probe by the frequency of the thing being backed off
to.** An aggregate says the channel is informative; the split says whether it is informative where
it is needed.

A validity check worth copying: the constant-channel control was trained twice from different
configs (renamed component, `annotating_components` set) and came out with **bit-identical parser
weights**, max |Δ| 0.000e+00 over 431 723 parameters -- confirming that a `constant = true` channel
really is inert and that the two experiments shared one correctly-matched control.

Kept: `sud.LexFieldEmbed.v1` (`sud_lex_embed.py`, both `lexicon` and per-token `tag` sources),
`build_xpos_lexicon.py`, `check_lex_embed.py`, `make_xposlex_config.py`, `make_vec_config.py`,
`make_leakfree_lzh_corpus.py`, `build_lzh_vectors.py` (with an unbuilt `--qieyun` arm),
`shrink_vectors.py`, `init_lzh_vectors.py`, `eval_lex_slices.py` (the frequency-slice harness), and
`train_xposlex.sh`. Arms: `training_lzh_{xposlex,xposlex_ctl,xposlex_whole,xpostagpred,`
`xpostagpred_ctl,vec,vec_ctl,vec_s1,vec_ctl_s1,vec_s2,vec_ctl_s2,vec96,vec96_ctl}`.

---

## Grafting the lzh encoder into zh (2026-08-17)

The question came from a real asymmetry: lzh outscores zh (released gold-preproc LAS **77.20** vs
**69.01**), and unlike every earlier transfer idea here the donor/recipient ratio finally pointed
the right way. **It still does not pay: −0.61 LAS over three paired seeds.** The entry is worth
reading less for the number than for the two pre-flight bounds that priced it, and the wiring trap
that nearly voided the whole table.

    seed   arm       TAG     UAS     LAS
       0   control  90.81   73.82   69.01
       0   graft    90.93   72.67   68.05
       1   control  90.67   72.45   67.70
       1   graft    90.97   72.97   68.26
       2   control  90.62   73.02   68.23
       2   graft    90.29   71.93   66.80

    paired mean      TAG +0.04 (sd 0.33)   UAS −0.57 (sd 0.95)   LAS −0.61 (sd 1.04)
    per-seed LAS delta                     −0.96 / +0.55 / −1.43

The sd exceeds the mean, so the defensible claim is **no gain**, not "it hurts". Note the CONTROL's
own spread — 69.01 / 67.70 / 68.23 — which is why three seeds is the floor here: zh's test set is
12 010 tokens, a third of lzh's, so the noise floor is WIDER than the ~0.5 LAS of the lzh arm
family.

**The direction that looked obvious is the one that was already dead, and vice versa.** Two
conditional-coverage counts, minutes each, settled both before any training:

  * **zh → lzh** (the intuitive direction, since zh is the bigger "language"): **92.6 %** of lzh's
    unseen-form test tokens are absent from zh GSD entirely, and those present have a median zh
    frequency of 2. The donor is also SMALLER than the recipient (98 614 tokens against 460 390).
    Ruled out unmeasured.
  * **lzh → zh**: every quantity flips. Donor 4.7× the recipient, zh's OOV rate is **12.46 %**
    against lzh's **1.15 %**, and **84.0 %** of zh's OOV tokens have a first character the donor
    knows at median frequency **44**. This is the arm that was built.

**The transfer is CHARACTER-level, and that is a ceiling, not a detail.** Only 4.4 % of zh's OOV
tokens have their full form as an lzh key — OOV zh words are multi-character and lzh is one Han
character per token (97.2 %). So `NORM` carries almost nothing for the population that needs it and
`PREFIX` carries 84.0 %; `SUFFIX` is `orth_[-3:]`, which for a two-character word is the whole word,
so it transfers at 4.5 %. One of four tables does the work.

**Re-tokenising the donor into zh's word regime does not lift that ceiling, and the bound is
segmentation-INDEPENDENT.** A token must be a contiguous substring of the text, so ask directly
whether lzh text can yield zh's OOV word types at all. Of 1 319 such types, **13.3 %** occur
anywhere in 460 k tokens of lzh train text, and the distribution is the whole story:

    zh OOV type length    1 char   2 chars   3 chars   4 chars   5-6 chars
    present in lzh text    78.3%     15.6%      0.0%      0.8%       0.0%

The only healthy column is the single characters, which **already transfer without any
re-tokenisation**. Classical Chinese does not contain modern Mandarin vocabulary under any
segmentation. Worse, the surgery would destroy the channel that does work: jieba over Classical text
puts **31.9 %** of tokens into multi-character groups against lzh's true rule-merged rate of
**2.84 %**, and of the 4 991 distinct multi-character tokens it invents, only **5.5 %** are real zh
training types — pseudo-words in place of well-attested characters. ⚠ And note that "retrain lzh
with the zh tokeniser" is a **no-op** in this repo regardless: both arms train through gold tokens
(`gold_preproc = true` / `sud.GoldTokCorpus.v1`), which is the same property that makes the parser
segmenter-agnostic. Changing the donor's key inventory means treebank surgery, not a config edit.

**Why it loses, from the slice split — the donor is most informative where it is most wrong.** LAS
delta by training frequency of the token's form:

    slice     tokens   share    mean d     sd    per-seed
    unseen      1497   12.5%     +0.33   1.47    +0.66  +1.61  −1.27
    1-2          955    8.0%     +0.32   0.63    +0.32  +0.94  −0.31
    3-10        1518   12.6%     −2.00   0.93    −2.04  −1.05  −2.90   <- all three same sign
    11-50       2278   19.0%     −1.01   1.49    −1.85  +0.71  −1.90
    >50         5762   48.0%     −0.43   0.92    −0.98  +0.63  −0.95

The coverage argument predicted help on the rare tail and the sign there is positive in both tail
slices — but at +0.33 on 12.5 % of tokens it is inside noise, and it is swamped by the **3-10 band,
the only slice with a consistent sign across all three seeds**. That band is forms zh has its own
evidence for but little of it, and it is exactly where a Classical prior must be UN-learned rather
than built on: the characters that transfer most confidently (之, 而, 以) are the ones whose modern
distribution diverges most from their Classical one. **A donor can be well-attested, cover the right
population, and still be worth less than nothing** — which is a different failure from the kanripo
vectors (informative, but empty where needed) and worth keeping distinct from it.

**Two wiring traps, the second of which silently voided the first sweep.**

1. Cross-language `source=` is blocked by **E150** (nlp lang vs vocab lang), so the encoder moves as
   a BLOB through `[initialize] init_tok2vec` — bytes carry no language. This is the route yue's
   Mandarin init already takes. The recipient config must also carry a filled **`[pretraining]`**
   block: spaCy resolves the target through `get_tok2vec_ref(nlp, config["pretraining"])`, and every
   base config here ships that block empty.
2. ⚠ **`Loaded pretrained weights` is NOT evidence of anything, and its absence is not evidence
   either.** In `spacy/training/initialize.py` the `layer.from_bytes(weights_data)` call is
   unconditional; only the `logger.info` AFTER it is level-gated, so the line never appears without
   `--verbose` even though the load happened. A driver guard grepping for it **rejected all three
   graft arms** on the first sweep — the models were correct and were scored separately. The guard
   now checks the SAVED config (`init_tok2vec = "<blob>"` in `model-best/config.cfg`), verified to
   pass all three graft arms and reject all three controls.

Two positive checks worth copying. The graft was confirmed against the donor BEFORE the sweep, by
training both arms two steps and summing |Δ| over all 24 tok2vec tensors — graft−donor **797**,
control−donor **117 065**. Verify the wiring, never assume it. And the CONTROL was reconciled
against the shipped arm afterwards: seed 0 reproduces `metrics_release_zh.json` to every decimal on
UAS **73.8219**, LAS **69.0076** and SENTS_F **99.1027**, which proves the added `[pretraining]`
block is inert and that the thing the graft was compared against really is the released recipe. Only
TAG differs (90.81 against 91.12) — expected, because the wheel ships the later warm-started tagger
conditioned on UPOS+FEATS, not the base arm's. **Reconcile a control against a known arm whenever
one exists**; it costs one command and converts "plausibly matched" into "matched".

**One confound not separated.** The graft arms early-stopped systematically sooner — 5 400 / 7 000 /
3 800 steps against the controls' 6 600 / 8 000 / 7 200, same patience 1600 — and the worst graft
seed trained least. So "lands in a worse basin" and "would recover with more training" are not
distinguished. A longer-patience or LR-warmup rerun, or a freeze-then-unfreeze schedule, would
settle it; given the 13.3 % ceiling above, it was not judged worth the compute.

**What would justify revisiting.** A donor whose OOV coverage of the recipient survives the
LENGTH split above, not just the aggregate — i.e. a corpus that actually contains the recipient's
multi-character vocabulary. Classical Chinese is not that corpus for modern Mandarin, and no
segmenter can make it one. Contrast yue, where the same graft pays **+1.15 LAS**: there the donor is
**18×** the recipient (197 k against 11 k) rather than 4.7×, and the varieties share a vocabulary.

Kept: `extract_tok2vec.py` (blob dump, with the E150 rationale), `make_graft_config.py` (fills
`[pretraining]` from a config proven to work, leaving `init_tok2vec` CLI-controlled so ONE config
serves both arms), `train_zh_graft.sh`, `configs/config_zh_graft.cfg`, `lzh_trad_tok2vec.bin`.
`eval_lex_slices.py` is language-agnostic and produced the slice table unchanged. Arms:
`training_zh_{graft,ctl}_s{0,1,2}`, `metrics_zh_{graft,ctl}_s{0,1,2}.json`.

---

## Sub-character channels for the lzh segmenter (2026-08-17)

Two attempts to give the character segmenter a phonological/graphic backoff so it could merge a
multi-character token it had never seen. **Both fail, and the second fails even after the missing
ingredient was supplied.** See `docs/lzh-tokenisation.md` for the harness and for the one cue that
DID work.

**1. Radical + Qieyun as an extra embedding channel: −0.49 held-out recall over three seeds.**
Measured on the jackknife (158 multi-char types split apart in train+dev, 611 held-out multi-char
tokens in an untouched test split), matched control, three seeds each:

    metric              phon     ctl    delta   per-seed
    token F            97.21   97.27    −0.06   −0.21 / −0.04 / +0.07
    retained recall    65.93   62.61    +3.32   +1.92 / +5.85 / +2.20
    HELD-OUT recall     5.13    5.62    −0.49   +0.17 / +0.33 / −1.96

**The channel is backwards from its own rationale**: it consistently helps the MEMORISATION slice
(positive on all three seeds) and does nothing for generalisation, when its entire purpose was to
fire where identity cannot.

⚠ **The denominator was 611 but the effective sample is the ~30 tokens actually recovered.** Control
seeds recovered 28 / 30 / 45 — a 2.78-point swing that is Poisson noise on ~30 events, larger than
any effect under test. Three seeds could not resolve a sub-point difference here; a future run needs
a larger hold-out fraction, not just more seeds.

**2. A dedicated transliteration classifier from IDS / radical / Qieyun: below the null.** The
obvious objection to (1) is that the segmentation objective is weak supervision for "is this
character phonetic". Wiktionary's `Chinese terms borrowed from Sanskrit` supplies explicit labels
(228 multi-char terms, 248 characters), so the question can be asked directly — 5-fold CV, shuffled
folds, negatives sampled from frequent kanripo characters in no Sanskrit-derived term:

    arm                accuracy   pos-F   precision   (base rate 0.250)
    NULL (majority)      0.7500   0.000       --
    radical              0.5948   0.400     0.318
    qieyun               0.7137   0.216     0.342
    IDS                  0.6472   0.373     0.336
    radical+qieyun       0.6532   0.353     0.331
    all three            0.6673   0.337     0.336

**Every arm is below the null on accuracy and barely above the base rate on precision.** Supervision
was not the missing piece. The likely reason is that this is not a natural class: ANY character can
be pressed into phonetic service and which ones were is conventional, not systematic. The radical's
better recall is probably the mouth-radical convention (嚩, 囉, 誐) and nothing more.

⚠ **The two probes ORDER THE CHANNELS OPPOSITELY** — Qieyun beats radical on character-pair merging
(0.092 vs 0.055) and radical beats Qieyun here (0.400 vs 0.216). When two measurements of "which
channel is better" disagree, neither is measuring a mechanism. `NEGATIVE-RESULTS.md` already
recorded radical > Qieyun for lexical class; that ordering is now known to be task-dependent and
should not be carried anywhere.

**What did work, and why it is a different shape:** an INDUCED inventory of transliteration
characters (Buddhist/classical log-odds) used as a RUN cue, gating a Wiktionary term list — 21/21
correct on two gold sutras. The lesson is that the signal is sequence-level and lexical, not
sub-character: 帝 ranks 976 in the inventory because 帝 "emperor" is ordinary classical vocabulary,
so 揭帝 is visible only as a run. Details in `docs/lzh-tokenisation.md`.

Kept: `make_seg_jackknife.py`, `eval_seg_jackknife.py`, `lzh_char_channels.py`,
`probe_translit_char.py`, `train_seg_phon.sh`, and `sa_presegment.py`'s optional `aux` channels
(**verified byte-identical to pre-patch when unused**, since sa/zh/id share that file). Arms:
`models/lzh_seg_jk_{ctl,phon}_s{0,1,2}`.

---

## A rule for lzh's `unk`, to rescue the Idiom layer (2026-08-17)

The lzh wheel's `Idiom`/`InIdiom` score 53.01 / 54.18. ⚠ An earlier version of this entry called
that "~22-31 F below the documented figures" of 75.68 / 85.54; that comparison is wrong at both
ends. The attributable gap is **13 F against the same rule on the both-scripts arm (66.18)**, and
75.68 itself does not reconcile with its own entry's availability number -- see the ⚠ added there.
The cause is NOT the Idiom pipe -- `add_sud_idiom.py` installs the non-trainable
RULE, which is what the ship decision called for, and a trained pipe would cost a further 11.63 F on
InIdiom. The cause is upstream and is standing hazards 5 and 6 compounding:

    per 1,500 test sentences      ExtPos    unk
    gold                              67     68
    both-scripts arm (documented)     38     56
    traditional arm (SHIPS)           36     42

⚠ **THAT TABLE COUNTS PREDICTIONS, AND READING IT AS RECALL IS WRONG -- as it was here first time.**
Scored properly against gold, `unk` recall barely moved; what changed is PRECISION:

    arm                   unk P      R        F      tp / fp / fn
    traditional (ships)  0.9836   0.6061   0.7500    60 /  1 / 39
    both-scripts         0.7326   0.6364   0.6811    63 / 23 / 36

The traditional arm is BETTER on its own metric. It predicts `unk` less often because it makes ONE
false positive against the other arm's 23 -- not because it misses more.

**The Idiom rule nevertheless prefers the loose arm, and that is the finding.** Same rule, same
eval, only the arm differs:

    Idiom rule on the both-scripts arm    P 100.00   R 49.45   F 66.18
    Idiom rule on the traditional arm     P  98.51   R 36.26   F 53.01

Because the rule is a CONJUNCTION of `ExtPos` and `unk`, the `ExtPos` conjunct filters the loose
arm's false positives, and the ones that survive land on tokens that really are idiom heads:
precision stays at 100 % while recall rises 36 % -> 49 %. **A component that is worse on its own
metric makes the downstream rule better.** That is standing hazard 5's "upstream errors multiply
rather than add" running in the unexpected direction, and it means tuning `unk` for precision --
which looks like an improvement in isolation -- silently costs the MISC layer.

`unk` is 99 tokens in 34,233, so no headline metric can see any of this.

**A pair lexicon does not beat the parser.** `unk` looks lexically closed -- 670 train tokens over
just **28 distinct dependents**, 可以 alone accounting for 224 -- so a (head, dependent) list
harvested from train is the obvious rule. Measured on test against the parser it would supplement:

    method                          P        R        F
    parser (what ships)          0.9836   0.6061   0.7500     (tp 60, fp 1, fn 39)
    pair lexicon, min count 2    0.7303   0.6566   0.6915     (fires 89x)
    pair lexicon, min count 1    0.5500   0.7778   0.6444     (fires 140x)

The rule buys +0.05 recall for -0.25 precision, and the parser makes **one** false positive in the
entire test set. The lexical concentration is real but it is information the parser has already
absorbed -- the same shape as the XPOS-lexicon result above, where a table that is a deterministic
function of the form carries nothing beyond `NORM`. The lexicon's own ceiling is 77.8 % (the share
of test `unk` whose pair occurs in train at all), below what the parser already achieves in F.

**So the lead is not a rule and not `unk` recall: it is that the Idiom rule wants a LIBERAL `unk`
predictor and the traditional arm is a conservative one.** The tunable is the parser's willingness
to emit `unk`, not the Idiom pipe. The both-scripts arm reaches exactly the availability its own
entry records (49.45 % against 49.5 %), so the real quantity to explain is why the traditional arm
supplies both rule inputs on only 36.26 % of gold idiom heads. All of this predates the segmenter
work and is already in the released wheel.

Kept: `eval_misc_raw_delta.py` (raw-side MISC comparison -- all three
`eval_sud_{subject,shared,idiom}.py` build `Doc(vocab, words=gold)` and so cannot see a tokeniser
change at all), and `--model` on `eval_sud_subject.py`. ⚠ `eval_sud_shared.py` already had
`EVAL_SHARED_ARM`/`EVAL_LEMMA_ARM` env overrides -- a `--model` flag added there by regex silently
did not wire up, and an accepted-but-ignored flag is worse than none.

## Agreement as a parser input, and beam training, for Latin (2026-08-19)

Two changes tried on the Latin `lemvec` arm (`config_la_lemvec.cfg`: lemma vectors plus one hash
table per morphological category, test LAS 73.23), each mirroring something Sanskrit was trying.
Both failed, and **the pair indicts something neither indicts alone**.

Unusually, both premises were measured on Latin first, and both came out STRONGER than the Sanskrit
numbers the same ideas were built on:

    premise                                    Sanskrit          Latin
    agreement gap, gold arc vs nearby non-arc  89.5 / 65.4       93.5 / 13.6  (+79.9)
      ... under PREDICTED morphology              --             73.4 / 12.7  (+60.8)
    non-projective sentences                   23.97 %           37.37 %

So neither result is "the signal was not there". Both are "the signal was there and the intervention
could not convert it", which is a different and more useful finding.

### Agreement as an explicit relational input: null

`sud.LemmaVecFeatsAgreeEmbed.v1` hands the parser twelve dimensions computed where both tokens are
in hand -- compatible at each offset -4..+4, any-compatible left/right within 20 tokens, how many
are compatible, and a flag for a token that declares no Case/Number/Gender at all. The rationale is
sound and still is: agreement is a RELATION, a per-token embedding cannot hold one, and the parser
would otherwise have to reconstruct "do these two share a case?" from two independently-hashed
vectors read at different state positions.

    measurement                        lemvec    agree     delta
    dev LAS at the saved checkpoint     75.71    75.54     -0.17
    test LAS, all (gold-preproc)        73.23    73.63     +0.40
    test LAS, ITTB+PROIEL               77.67    77.98     +0.31
    test LAS, Perseus                   53.91    54.70     +0.79
    whole-doc UAS (own segmentation)    78.98    78.93     -0.05

**An effect that is 0.17 behind on dev, 0.40 ahead on test and 0.05 behind under the model's own
sentence segmentation is smaller than the disagreement between the ways of measuring it.** Seeds
(agree and lemvec, 1 and 2, both `--system.seed` and `--corpora.train.augmenter.seed`) were queued
to settle the Perseus slice specifically, and were STOPPED after one to free the machine for
Sanskrit. What that one seed did establish is the spread itself: two runs of the same config differ
by a mean absolute **0.272 LAS** at matched steps, max 0.82 (`docs/latin.md`). **The +0.40 sits
inside the noise band**, which is what the dev and whole-doc figures were already saying by
disagreeing with it.

The mechanism check is the only clean signal and it is tiny. Agreement-detectable errors --
`scripts/analyse_la_agreement_errors.py`, defined as the gold head agreeing where the predicted head
does not, so agreement alone rules the error out -- fell **383 -> 365**, concentrated on the subject
relation (88 -> 76). That is the block doing exactly its job, worth **0.03 UAS**. Total attachment
errors rose slightly (11 541 -> 11 569), so whatever the block fixed it paid back elsewhere.

**The explanation is that the per-feature tables had already extracted the signal.** Those tables
are what took agreement-detectable errors from 454 (released `aug` arm) to 383, and the residue is
worth only 0.70 UAS in total. Read against `config_la_morphfirst.cfg`, the pair is a clean
progression: the whole-bundle MORPH hash was worth nothing (0.7256 against a capacity control's
0.7255), DECOMPOSING it per category was worth +1.51 LAS, and making the RELATION explicit on top of
that was worth nothing again. The middle step is where all the value is.

⚠ Not ruled out: the block is silent on more than half of all tokens (`unknown` averages 0.554,
because it requires all three of Case, Number and Gender, which no verb has). A Number-only channel
for verbs, or a narrower reach, are different experiments. The measured 0.70 UAS ceiling does not
justify them.

### Beam training: refuted on its own mechanism, which is the interesting part

`beam_parser`, width 8, `beam_update_prob = 0.5` -- the settings from `config_sa_mp2_beam.cfg`,
untuned. The rationale was specific and quantified: Latin loses on discontinuity (37.4 % of test
sentences carry a crossing arc; those arcs are 5.0 % of tokens but 16.0 % of all attachment errors,
UAS 28.72 on them against 82.74 in a wholly projective sentence, and the gap survives a length
control at 8-11.5 points in EVERY length bucket including 2-9 tokens). Pseudo-projective encoding
makes such an arc a locally costly, globally correct decision -- `mod||subj` scores worse at that
step and only pays after de-projectivisation -- so a greedy decoder cannot take the bet and a beam
should be able to.

It hit patience at step 11 800 and lost by **2.76 test LAS** (70.47), below even the released `aug`
arm. But the headline is not the finding:

                                            lemvec (greedy)    beam
    crossing arcs EMITTED                        1 206        1 157
    gold crossing arcs recovered as crossing      31.1 %       27.8 %
    UAS on the crossing arcs                      30.45        28.58
    UAS on wholly projective sentences            83.87        82.03

**A beam that searches eight sequences instead of one and emits FEWER `||` actions is not
search-limited.** The model scores those actions low, and widening the search cannot help when the
scores are the problem. Beam is also uniformly worse, including on ordinary projective arcs, so
nothing was traded for anything.

### What the two results jointly indict

Three interventions have now been aimed at the same 2.68 UAS of non-projective headroom, from three
directions, and all three left it on the table:

    let the parser EXPRESS more   sa, min_action_freq 30 -> 5     45 -> 132 moves, -0.06 LAS
    push it to REACH more         sa, 3x upsampling               recall 0.119 -> 0.237, +0.43 net
    let it SEARCH more            la, beam_parser width 8         FEWER arcs emitted, -2.76 LAS

What that jointly indicts is **the pseudo-projective representation, not any decoder over it**. A
scheme that turns discontinuity into a rare composite LABEL makes it a data-sparsity problem, and
none of expressing, reaching or searching addresses data sparsity. Anything further here should
change the representation -- a genuinely non-projective transition system, or an arc-factored
decoder -- rather than tune a search over this one.

⚠ Beam stopped on patience at 11 800 of 20 000 while still climbing slowly, so "given more steps" is
not formally excluded. It was 2-3.7 LAS behind in every 2 000-step window from step 1, and its
emitted-arc count moved the WRONG way, which extra training does not explain.

Kept and reusable: `scripts/analyse_la_nonproj_errors.py` (per-arc / per-sentence non-projectivity
error attribution, `--lang` so it runs on any arm), `scripts/analyse_la_agreement_errors.py`,
`scripts/check_la_agreement_signal.py` (the go/no-go that costs two minutes rather than a night),
`scripts/check_la_agree_channel.py`, and `scripts/train_la_agree_beam.sh` with its `why` phase --
which is what caught the beam refuting its own premise while the LAS column only said "worse".

## Decode-time constraints on cross-clause arcs (Sanskrit, merged corpus) — both variants lose

The clause-merged arm (`docs/sanskrit.md`) joins the treebank's clause units into one tree, so some
arcs now span a unit boundary. Two ways of forbidding the wrong ones at decode time were built and
measured on the merged test; **both lose about 1.3 LAS, and — the part that settles it — both make
the crossing arcs they target WORSE.**

    pmerged arm                     LAS     crossing-arc LAS (1 837 arcs)
      unconstrained                 54.82   22.26
      label mask, {conj:coord, parataxis}
                                    53.51   14.64
      label mask, the 6 relations the merge actually introduced
                                    53.79   17.86
      unit-root mask (two-pass)     53.51   18.56
    pmerged + order augmentation
      unconstrained                 55.41   28.03
      unit-root mask (two-pass)     54.05   23.57

**Variant 1, by LABEL** (`parse_with_clause_bounds`): allow a crossing arc only if its relation is
one the merge introduces. The premise does not survive counting. The merge introduced SIX relations,
not two (`conj:coord` 1065, `parataxis` 852, `comp:obj` 221, `subj` 110, `mod` 103, `comp:obl` 87),
and more importantly *crossing a mark is not crossing a clause boundary*: a single daṇḍa is a
half-verse break sitting INSIDE one of the treebank's own units (`Punctuation=comma` is unit-medial
8 129 times). Restricting to the double daṇḍa still leaves 41 % of crossing gold arcs outside a
{coord, parataxis} set. Headroom said it would be close — 138 currently-correct arcs destroyed
(−0.61) against at most 353 wrong ones fixed (+1.56) — and it came out the wrong side.

**Variant 2, by STRUCTURE** (`parse_with_unit_roots`): allow a crossing arc only where the
dependent's subtree spans its whole unit, since a merge-introduced edge always REPLACES a root edge.
Much better founded: **81.5 % of gold unit-crossing content arcs satisfy it** (train 78.8), against
59 % for the label version. It still loses.

**Why, and the general lesson.** The rule is not decidable incrementally in ArcEager — a LEFT arc
pops its child so the subtree is final, but a RIGHT arc pushes it, and cross-unit arcs are
overwhelmingly RIGHT arcs (unit *n*'s root attaching back into unit *n−1*). Hence two passes: pass 1
bans every crossing arc to name each unit's root, pass 2 allows crossings only from those. Pass 1
finds a true unit root **83.4 %** of the time, and 8.6 % of units have more than one outward-pointing
token in gold anyway, so the admissible space is right about 0.815 × 0.834 ≈ **68 %** of the time.

A hard mask pays only when the banned action is nearly always wrong. The agreement constraint works
because a disagreeing adjectival `mod` is close to always wrong; these fail because the banned action
is correct roughly a third of the time, and the unconstrained decoder was already concentrating its
probability on those same correct arcs. **Consistency with gold at 80 % is not a licence to
constrain** — the bar is the error rate of what you are overriding, not the accuracy of your rule.
Same shape as the decode-time lexicon result above.
