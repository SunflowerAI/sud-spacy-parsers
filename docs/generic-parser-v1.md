> ## ⚠ SUPERSEDED — this describes v1, which read aligned fastText vectors
>
> Kept because its four data decisions and seven traps still apply, and because its measurements are
> the reference points v2 is read against: vector channel +8.60 macro LAS, the alignment specifically
> +2.26, FEATS +2.79, language identity −0.02.
>
> **v2 is `docs/generic-parser-v2.md`** — no lexical channel at all, four typological features in
> place of nine graded word-order parameters, ~50 treebanks, and a test set of held-out languages
> rather than leave-one-language-out. The Results section below was withdrawn mid-fix and never
> rewritten; read `metrics/generic/*.json` and the branch commit message instead, and note that every
> figure there is SINGLE-SEED.

# The generic parser: UPOS + FEATS + aligned vectors, thirteen languages, no wordforms

A single transition-based parser trained on all thirteen SUD treebanks at once, whose entire view of
a token is:

    UPOS            one of 17 universal categories
    FEATS           one hash-embedded table per morphological CATEGORY
    aligned vector  128 d, in the one shared space of `docs/aligned-vectors.md`, + an OOV flag

and nothing else. **No wordform, no prefix, no suffix, no shape, no script, no XPOS, no language
identifier.** Every other parser in this repo starts from `MultiHashEmbed`'s
`["NORM","PREFIX","SUFFIX","SHAPE"]` — the token string — and adds channels beside it. A string is
exactly what cannot be shared across thirteen writing systems, so this arm removes it, and the
aligned vectors are what stands in its place.

**The point is low-resource parsing.** Telugu-MTG has 5 097 training tokens and Tamil-TTB+MWTT
8 409; the question is whether twelve other languages' syntax can be made to reach them. That is
what the sample is balanced for, what the zero-shot arms test, and what the results table below is
organised around.

| | |
|---|---|
| build | `bash scripts/train_generic.sh` |
| corpus | `scripts/prep_generic.py` → `assets_generic/` → `corpus_generic/` |
| vectors | `scripts/build_generic_vectors.py` → `assets_vec/generic_vec.npz` |
| layer | `sud.GenericEmbed.v1` (`scripts/sud_generic_embed.py`) |
| reader | `sud.GenericCorpus.v1` (`scripts/generic_corpus.py`) |
| configs | `configs/config_generic{,_ctl,_shuf,_nofeats,_langid}.cfg` |
| eval | `scripts/eval_generic.py` |

---

## The four data decisions, each of which would have changed the experiment silently

### 1. `@`-subtypes are stripped; `:`-subtypes are kept — 120 labels become 27

The thirteen treebanks together use **120** deprels and roughly **60 of them occur in exactly one
language**: `mod@neg` is Persian's alone at 7 818 tokens, `flat@vv` Korean's at 4 868, `udep@instr`
Sanskrit's at 1 982, `mod@cmp` Latin's at 2 789. Asking a shared model to predict those is asking it
to reproduce one treebank's annotation convention, and at this corpus size `min_action_freq` would
delete most of them anyway with their recall pinned silently to zero (`docs/dravidian.md`).

Stripping `@` leaves **27** relations, every one attested in at least four languages and twenty in
ten or more:

    mod  comp:obj  punct  subj  root  compound  flat  conj:coord  comp:obl  comp:aux  cc  udep
    det  unk  comp:pred  discourse  conj:appos  parataxis  orphan  clf  vocative  conj:dicto
    goeswith  dislocated  list  comp  conj

The `:` subtypes stay. `comp:obj`/`comp:obl`/`comp:pred`/`comp:aux` and `conj:coord`/`conj:appos`
are core SUD, defined identically across treebanks, and the comp/mod distinction is the thing this
project exists to study.

### 2. `Shared` is removed from FEATS, and this one is leakage

`Shared=Yes/No` is **native SUD annotation** — it is in the pristine treebanks, not something this
repo hoisted — and it records whether a dependent is shared across the conjuncts of a coordination.
That is a fact about the **tree**. This project's own `sud_shared` pipe predicts it *from* a
finished parse (`docs/sud-misc-layer.md`), so handing it to a parser as an input inverts the
dependency and leaks coordination structure into the model that is supposed to recover it. It sits
on 10 178 English tokens alone and appears in all thirteen treebanks, so leaving it in would have
been quiet and would have flattered every coordination number in the results.

### 3. XPOS is blanked

One tagset per arm is this project's rule (`docs/xpos.md`) — Latin's composite codes, English's
`,`, Telugu's verbatim copy of UPOS — which makes XPOS the least commensurable column in the file.
UPOS is the universal one.

### 4. An empty LEMMA falls back to the FORM

Telugu's LEMMA column is `_` on every one of its tokens and **spaCy keeps `_` as a literal string,
not as missing** (CLAUDE.md; it once taught a Sanskrit transducer `FORM → "_"` on 5 043 tokens).
Sanskrit's aligned vectors are keyed by LEMMA, so a literal `_` reaching the lookup would be a
silently all-OOV language. Identity is the fallback, the same one `scripts/prep_te.py` uses.

---

## The balance is typological, not per-language

Equal shares per language would still be a genealogical accident: four of the thirteen treebanks are
Indo-European and three are Sinitic, so "one share each" hands **54 % of the corpus to two
families**. Instead each *group* gets an equal token budget, split equally within it, and capacity a
small treebank cannot fill is redistributed **inside its own group** — never across groups, which
would spend the balance being bought.

At the default budget of 60 000 train tokens per group:

| group | languages | tokens | % of mix |
|---|---|---|---|
| Indo-European | en 15 148 · fa 15 021 · la 15 116 · sa 15 003 | 60 288 | 16.3 % |
| Sinitic | zh 24 455 · yue 11 158 · lzh 24 449 | 60 062 | 16.2 % |
| Semitic | ar 60 162 | 60 162 | 16.2 % |
| Japonic | ja 60 052 | 60 052 | 16.2 % |
| Austronesian | id 60 103 | 60 103 | 16.2 % |
| Koreanic | ko 56 687 | 56 687 | 15.3 % |
| **Dravidian** | **ta 8 409 · te 5 097** | **13 506** | **3.6 %** |
| | | **370 860** | |

**Two groups cannot reach parity at any budget and the manifest says so rather than hiding it.**
Dravidian has 13 506 training tokens in existence across both its treebanks and Koreanic 56 687, so
they sit at 3.6 % and 15.3 % against an even seventh's 14.3 %. Raising the budget makes the
imbalance *worse*, not better — `--budget 30000` gives Dravidian 7.0 % of a 193 k-token corpus.
That trade (more data, less balance) is the one dial worth turning here.

`yue` (11 158) also caps out; its unused Sinitic share goes to zh and lzh, which is why those two
sit at 24 k rather than 20 k.

Sampling is by whole **10-sentence block**, matching `spacy convert -n 10`, so a training doc is ten
*consecutive* sentences of one text. Ten unrelated fragments would make sentence segmentation
easier than it is. Dev is capped at 3 000 tokens per language; **test is never sampled** — every
language is scored on its complete test set.

---

## What the model actually sees

### FEATS is decomposed, one table per category

`MORPH` as a spaCy column is a hash of the **whole normalised bundle**, so `Case=Nom|Number=Sing`
and `Case=Nom|Number=Plur` arrive as two unrelated symbols. Across thirteen treebanks that is fatal
rather than merely crude: no two of them share a bundle inventory, so a whole-bundle hash would make
every language's morphology a private vocabulary and there would be nothing cross-lingual left in
the channel. `sud.MultiHashEmbedFeats.v1` (already in the repo, for the conditioned tagger) is
reused verbatim — `Case=Acc` is the same symbol whether it came from Latin, Sanskrit or Tamil.

**42 channels**, derived from the corpus by `make_generic_config.py` rather than hardcoded. The
threshold is one language, not a majority, and that is deliberate: a per-language category costs one
small table and cannot confuse another language (a token that does not declare it hashes to
`InflClass=` and lands on the same row as every other token that does not), whereas dropping it
throws away real morphology — and **the low-resource languages have the most idiosyncratic FEATS
inventories**, so a majority threshold would tax exactly the languages this arm exists for.

The channel is very unevenly filled, which is itself part of the result to read:

| | en | ar | la | sa | ta | fa | id | lzh | zh | yue | ko | ja | te |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % of tokens with any FEATS | 70 | 78 | 69 | 82 | 88 | 61 | 42 | 33 | 15 | 10 | 4.7 | 4.1 | 2.3 |

For ko, ja, te and yue the morphology channel is nearly empty and the arm is effectively
**UPOS + vector**. `generic_nofeats` measures what that costs the other nine.

### The aligned vector is the only lexical channel

`assets_vec/generic_vec.npz` is 201 656 rows × 128 d (96.8 MB), pruned out of the 325 MB of
`release_vectors/` down to the rows the thirteen treebanks actually reach. Token coverage over each
full treebank:

| la | sa | lzh | ja | fa | ar | en | id | te | zh | yue | ta | ko |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 99.5 | 99.9 | 99.8 | 98.0 | 96.9 | 95.1 | 94.6 | 93.9 | 89.1 | 89.6 | 88.9 | 86.0 | **74.3** |

Korean is the weak one, for the reason `docs/korean.md` already records: 34.5 % of its test tokens
are unseen *strings*, because an eojeol is a phrase glued together.

Three things the table carries from each source asset's meta, because **none of them can be
guessed**: `key_attr` (**sa is keyed by LEMMA**, the other twelve by surface form), `lookup`
(eleven are lowercased, two are not — worth 31 points of English coverage), and `key_norm` (la alone
has an orthography fold, `v`→`u`, `j`→`i`, macrons off). `sud_generic_embed.py` reads them back and
imports the fold from `aligned_vectors.py` rather than reimplementing it.

**Rows are copied verbatim. Nothing is re-normalised, re-projected or re-fitted** —
`docs/aligned-vectors.md` measures a per-language transformation taking retrieval from 63.8 % @1 to
**0.0 %**.

### How the language is chosen, and why that is not a language feature

The thirteen tables live in one space but are separate row-sets, so a lookup has to know which
language a token is in. That is read off `Doc._.tb_lang`, used **only** to pick the row, and never
embedded: the model has no parameter that varies with it, and two identical (UPOS, FEATS, vector)
triples get identical treatment whatever language they came from. `generic_langid` is the control
that measures what actually knowing the language would be worth — the only honest way to say the
shortcut was not taken.

---

## The arms

| arm | differs by | isolates |
|---|---|---|
| `generic` | — | the arm |
| `generic_ctl` | `constant = true`: same Linear, same parameter count, every token handed the zero vector and the OOV flag | what the **parameters** buy |
| `generic_shuf` | `shuffle = true`: the same rows, key-to-row correspondence destroyed *within* each language | what the **alignment** buys |
| `generic_nofeats` | `feats = []` | what **morphology** buys |
| `generic_langid` | + a thirteen-row language embedding | what knowing the **language** buys |

`_ctl` and `_shuf` are not redundant, and `_shuf` is the harder test. A shuffled table still gives
each language a distinct, consistent, arbitrary code per wordform — which a *monolingual* parser
could exploit and a cross-lingual one could not. **If `_shuf` matches the arm, the alignment did
nothing and the model is thirteen parsers in a trench coat.**

⚠ **The static-vector negative results do not transfer to this arm, and it looks as though they
should.** `NEGATIVE-RESULTS.md` records kanripo vectors as a parser input for lzh at **+0.04 LAS
over three seeds**, and fastText `md` on yue/id/ko at +0.2–0.9 inside seed noise. Both measured a
vector channel added *beside the wordform*, for a parser that already read the string and had
already learned that string's syntax from the same treebank. Here the vector is the only lexical
channel there is, and its job is not to add information about a language the parser knows — it is to
put a Tamil noun and a Latin noun in the same place. Same asset class, different question; the
earlier results neither support nor refute this one. `generic_shuf` is what settles it.

---

## Results — WITHDRAWN AND RETRAINING (2026-08-23)

> ⚠ **Every number previously in this section was measured on a corpus whose Sanskrit came from a
> SUPERSEDED representation, and has been removed rather than left standing.** `prep_generic.py`
> took its source map from `train_sud.sh`'s `src_conllu()`, which still names
> `corpus_sa_csl_rev/` — and CLAUDE.md lists `rebuild_sa_csl_rev.sh` under *"Superseded but kept"*.
>
> | sa train corpus | sents | tokens | `udep` | `comp:obl` |
> |---|---|---|---|---|
> | `csl_rev` (used; superseded) | 21 707 | 163 802 | **7.89 %** | 1.90 % |
> | `relabeled_ext.csl_mwt` (current) | 21 647 | 163 308 | **0.00 %** | 4.32 % |
>
> Two independent defects, neither of which announced itself. Sanskrit entered the mix
> **unrelabelled** — 7.89 % of its tokens left as noncommittal `udep`, where the current generation
> commits every one — while all ten other relabelled languages were on their released generation.
> And its tokenisation differs from the released sa arm's, so **no monolingual comparison involving
> sa could have been valid on it**. A superseded corpus loads, converts and trains exactly like a
> current one.
>
> The corpus, the vector table (fingerprint `5701a545a8cb12a7`) and the configs have been rebuilt;
> all five arms, the four zero-shot arms and the multi-seed sweep are retraining. Results will be
> restored here from the new run.

### What is already settled and does not depend on the retrain

- **The method and its controls.** The arm, the four capacity-matched controls, the balanced sample,
  the 27-label inventory and the harness are unchanged; only sa's rows of the corpus moved.
- **`scripts/check_generic_inputs.py` passes all thirteen checks** — the three headline arms are
  capacity-matched to the parameter, the embed's attrs are exactly `["POS"]`, and an unset
  `Doc._.tb_lang` raises.
- **The evaluation regime, corrected three times.** `--monolingual` now refuses to run without
  `--gold-sents`, and tokeniser-supplied FEATS are stamped onto the monolingual arm's input
  (`TOKENISER_FEATS`; sa's `Compound`, which its parser reads and a gold-token harness would
  otherwise deny it — `docs/sanskrit.md`, +1.30 LAS).

### Reference points for the retrained run, established through the right harness

The released sa arm on **Vedic**, via `eval_sa_compound.py --reader norm` on
`corpus_sa_mwt_rl2/` — the harness and corpus `metrics_sa_*_Vedic.json` were measured with, and
reproducing them exactly:

| arm | UAS | LAS |
|---|---|---|
| `training_sa_mp2_s1` (shipped in v0.2.0) | 69.19 | **57.05** |
| `training_sa_mp2_sub_s1` (`SA_BASE` default) | 71.00 | 56.71 |

⚠ **`metrics_release_sa.json`'s 37.35 is NOT this number** — it is the 1843-token UFAL test set, not
Vedic (`docs/packaging-and-release.md`). Plain `spacy evaluate --gold-preproc` on the Vedic file
gives 49.73 for the same arm, because it denies the arm the tokeniser-set `Compound` input. Three
different, individually-correct Sanskrit numbers — 37.35, 49.73, 57.05 — none of which may be
substituted for another.

---

## Traps

**1. `Doc._.tb_lang` must be set, and the layer refuses rather than defaulting.** One row-set per
language; a default would look every token up in the wrong table, miss nearly all of them, and score
exactly like the layer's own dead-channel control. `sud.GenericCorpus.v1` sets it at training time;
`generic_corpus.annotate(doc, lang)` is the single definition of the inference-time regime.

**2. UPOS, FEATS and LEMMA are INPUTS, and the stock reader does not supply them.** They must be on
the *predicted* doc, not just the reference. `spacy.Corpus` builds the predicted doc from the
reference's words and nothing else, which would leave POS at 0 and MORPH unset on every token and
train the model to ignore three channels that then appear from nowhere at inference — the gap
`sud.CompoundCorpus.v1` exists to close for sa's `Compound`. This is also why `spacy evaluate` gives
the wrong answer for this arm and `scripts/eval_generic.py` exists.

**3. LEMMA is copied for exactly one reason.** Sanskrit's vectors are keyed by lemma
(`docs/aligned-vectors.md` trap 4). For the other twelve languages it is inert.

**4. The vector table is keyed off the FULL treebanks, not the sampled corpus.** Keying it off
`assets_generic/` would tie it to whatever `--budget` was last run: change the budget, and the new
sample's fresh types have no rows, so they reach the parser as OOV. Nothing raises, the model
trains, and it is simply worse. The table also carries a **fingerprint** that the layer refuses to
load against a mismatch, because a table from another asset release has the same shape and the wrong
rows — CLAUDE.md hazard 10 exactly.

**5. `min_action_freq = 1`, never the default 30.** With 27 labels and a long tail (`conj` at 17
tokens in the sample) the default would delete labels silently, which is the damage
`docs/dravidian.md` records.

**6. The languages must be interleaved.** The reader shuffles all thirteen together. Reading one
treebank at a time would give the optimiser thirteen sequential domain shifts per epoch, and
whatever it learned about Telugu would be overwritten by Latin.

**7. The `udep` rate is not uniform, and it is annotation policy rather than language.** Ten of the
thirteen source treebanks are extended-scope udep-relabelled; **ta, te and sa are not**. So `udep`
is 9.9 % of Tamil's tokens and 2.4 % of English's. The manifest records the per-language rate. Any
reading of `udep`, `comp:obl` or `mod` across languages has to account for it.
