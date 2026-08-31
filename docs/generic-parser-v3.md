# The generic parser v3: a lexical channel you can fill with an English gloss

> **Status: complete, three seeds throughout.** Deltas are computed per seed against that seed's OWN
> baseline, so baseline drift cannot leak into an arm. Ranges sit beside every mean, and the ones
> that swallow their effect are named as such.
>
> **Headline: 50 annotated sentences are worth +10.15 LAS** on genus-disjoint held-out languages —
> +9.45 from the trees through a fitted language embedding, +0.70 more from the gloss line through
> a lexical channel. The gloss channel alone is worth +5.12, recovering 94 % of what a real aligned
> fastText table would give.

v2 reads UPOS + FEATS + a typological profile and no wordform at all. v3 adds back **one lexical
channel**, and changes what fills it.

|  | v1 (`docs/generic-parser-v1.md`) | v3 |
|---|---|---|
| key | surface FORM | **LEMMA** |
| filled at deployment from | the language's own vector table | **an English gloss the user writes** |
| a language with no fastText | channel is empty — dead | channel fills from English |

That last row is the whole idea. Aligned fastText vectors are rotated into the **English hub**, so an
English gloss and a source lemma are points in one space rather than two kinds of thing. Training
coverage therefore limits how well the channel is **learned**, not where it can be **used**:
Chintang, K'iche' and Xavante have no fastText at all and can still fill it.

## The substitution was measured before anything was built

The arm rests on "a gloss vector stands in for a lemma vector", which is an assumption about an
input regime — and this repo has paid 4.83 F once for asking a model at inference for a regime it
never met (`CLAUDE.md` hazard 11). Measured first, on the two treebanks carrying both a `Gloss=`
column and an aligned table:

| | cos(source, en-gloss) | shuffled | random-en | beat shuffled |
|---|---|---|---|---|
| ar PADT, 236 k pairs | **+0.444** | +0.109 | +0.012 | 94.0 % |
| lzh Kyoto, 382 k pairs | **+0.340** | +0.116 | +0.022 | 84.5 % |

Re-measured end to end through the **built** table (lemma-keyed, with the Arabic fold): **+0.4597**
against +0.1115 shuffled, 95.4 % beating their partner, 239 748 pairs.

Well above a permuted control, and nowhere near 1.0. **Both halves of that sentence turned out to
matter** — see the results.

## Scope: the aligned-44

fastText publishes 44 languages already rotated into the English space, so they need no rotation
fitting at all (`route = "pre"`). Of v2's 80 training languages **32 are in that set** — 45 % of
training tokens; of the 20 held-out languages, **6** are. Extending to the other 48 means a
Procrustes fit per language against a bilingual dictionary, whose quality v1's report shows varying
from 0.62 to 0.07 hit@1. That is a separate decision and has not been taken.

Mean training token coverage of the lemma column: **92.5 %**, worst case 79.1 % (Norwegian, whose
misses are `$.` and `$,` punctuation lemmas and are correctly OOV).

## Four key folds, and three languages that would have trained on an empty channel

| | before | after | why |
|---|---|---|---|
| ar | 41.2 % | **96.2 %** | PADT lemmas are fully vocalised citation forms; fastText Arabic is not |
| ko | 36.4 % | **83.8 %** | `청+하+고` — morpheme-segmented lemmas against orthographic words |
| et | 82.6 % | 88.4 % | `maa_ilm` compound-boundary marks nobody writes |
| fi | 84.7 % | 88.6 % | the same, spelled `#` (`yli#opisto`) |

**The absent fold is the dangerous case, not a wrong one.** A 41 % channel trains, converges and
reports an ordinary loss curve; it is simply worse, and no metric in the sweep names the cause.

Two had an obvious fix that measurement rejected. **Korean: joining the morphemes is 16.7 points
worse than taking the first** (67.1 % against 83.8 %) — joining rebuilds an inflected surface string
rather than a lexeme, and Korean is stem-initial so the head morpheme is the lexical one.
**Arabic: the general-purpose NFD-then-drop-combining costs 2.0 points** against an explicit harakat
class (94.2 % against 96.2 %), because it also takes the hamza off `أ` and `إ` and folds both onto
`ا`, which fastText keeps distinct.

⚠ **Type coverage, which is what `align_vectors.py` printed, would have found none of this.** German
reads 28.5 % of TYPES and covers 90.2 % of TOKENS — the gap is the hapax-compound tail. The reports
carry token coverage now.

## The evaluation, and why one macro is the wrong number

**The group split is the whole analysis.** Six of the twenty held-out languages are in the
aligned-44, so for them the `lemma` fill finds REAL vectors — that run is the channel's UPPER BOUND.
For the other fourteen the identical run is all-OOV, its FLOOR. Averaged together the two cancel and
the channel looks inert. `compare_generic_v3.py` refuses to print one number for both.

**Two gloss keys, and only one is the deployment number.** `--gloss-key lemma` is an upper bound —
the v2 contract declares UPOS and `tb_lang` as user inputs and says nothing about lemmas.
`--gloss-key form` is what a deployer has. Report both; never quote one for the other.

**Three languages cannot be scored on the gloss fill by any route** — Bororo, Komi-Zyrian and
Xavante have neither a Wiktionary extract nor a `Gloss=` column. A macro "over the test languages"
that silently means seventeen is the kind of number this repo has had to retract.

**The Wiktionary figure understates a real glossing user**, which is the direction worth being wrong
in: mean fillable is 51.3 % by lemma and 37.0 % by form, while the three held-out languages carrying
a real gloss column run 67–81 %. Yoruba is the instructive case — 67.4 % from its own column against
15.3 % from Wiktionary, so the shortfall is the dictionary, not the language.

## Results

Every figure is three seeds, delta computed per seed against that seed's OWN baseline, on the six
held-out languages that have aligned rows. The baseline reproduces v2's published `g2_base`
(54.45 / 53.82 / 54.45 against 54.24), so this harness and v2's agree.

### The zero-shot channel

| fill | mean | range |
|---|---|---|
| real aligned vectors — the **ceiling** | **+5.44** | 1.80 |
| Wiktionary glosses | +0.53 | 2.89 |
| **LLM glosses** (proxy for a human annotator) | **+5.12** | 2.22 |

**Good glosses recover 94 % of the ceiling; a dictionary recovers nothing.** Every seed positive,
every language positive. Two languages (lt, vi) BEAT the aligned-vector ceiling, because a
contextual gloss disambiguates where a static type-level vector cannot.

Per language, and the spread is the point:

| | gloss fill | Wiktionary | LLM | ceiling |
|---|---|---|---|---|
| el | 99.1 % | +0.37 | +3.86 | +4.04 |
| hu | 98.9 % | −0.42 | +3.96 | +4.82 |
| lt | 98.8 % | −1.84 | **+4.95** | +4.65 |
| lv | 97.9 % | −2.84 | **+4.96** | +5.53 |
| th | 99.6 % | +5.07 | +7.39 | +8.54 |
| vi | 98.1 % | +2.86 | **+5.59** | +5.07 |

⚠ **Wiktionary was ACTIVELY HARMFUL on three of the six.** Its +0.53 average was not a small effect;
it was real gains cancelling real damage. A bag of every sense at once is a blurry centroid, and
feeding one to the channel is worse than leaving it empty.

**So gloss QUALITY was the binding constraint all along, not the gloss route.** Everything measured
before the glosser existed — that the substitution fails, that augmentation is needed to rescue it,
that the strength of that augmentation might be tunable — was measuring a bad dictionary.

### The four cells

The deployment story is a human annotating a small sample. That produces BOTH artefacts from one
pass: the gold trees fit the 128-d language row, the gloss line fills the lexical channel. At 50
annotated sentences:

| cell | mean | range |
|---|---|---|
| neither (baseline) | 0.00 | — |
| gloss channel only | +5.12 | 2.22 |
| language embedding only | **+9.45** | 0.90 |
| **both** | **+10.15** | 1.14 |

**The trees are worth about twice the gloss line, and the two overlap heavily** — independent
addition would give +14.57 against the +10.15 observed. Most of what a gloss tells the parser, a
fitted language row already tells it.

The average hides an interaction that matters more than it does:

| | embedding | both | marginal from glosses |
|---|---|---|---|
| el | +3.84 | +5.18 | **+1.35** |
| hu | +3.87 | +6.85 | **+2.99** |
| lt | +8.78 | +9.77 | +0.99 |
| lv | +9.53 | +10.52 | +0.99 |
| th | **+19.58** | +16.79 | **−2.78** |
| vi | +11.12 | +11.80 | +0.68 |

**Glosses help most where the embedding helps least** (hu +2.99, el +1.35 — the two languages the
embedding alone barely moved) and HURT where it helps most (th, whose embedding is worth +19.58).
They are substitutes, not complements, and +0.70 is an average over that, not a small uniform gain.

⚠ The row is fitted with the channel FILLED, from the same 50 glossed sentences — otherwise it is
fitted under one input regime and used under another. Measured both ways: **+10.15 either way**, so
the mismatch cost nothing here. Worth having checked rather than assumed, and worth knowing that
this particular guard was not load-bearing.

### Getting the glosses

A dictionary is not enough, and the gap is both coverage and sense:

| | coverage | cos to a HUMAN gloss |
|---|---|---|
| Wiktionary, yo | 15.8 % | +0.2998 |
| **LLM, yo** | **37.7 %** → 91.7 % windowed | **+0.4412** → +0.7024 |
| Wiktionary, xcl | 41.4 % | +0.3953 |
| **LLM, xcl** | **69.9 %** → 99.5 % windowed | **+0.5088** → +0.6547 |

Graded against the two held-out treebanks that carry a human `Gloss=` column, so the proxy is
validated rather than assumed. gemma3:27b, one call per 15-token window with the whole sentence as
context and the Wiktionary candidates in the prompt.

⚠ **PER-SENTENCE CALLS FAIL ON LENGTH, AND THE FAILURE IS SILENT.** Measured on gemma4: 0 % fallback
below 10 tokens, 67 % at 20–29, 100 % above 50. Yoruba's median sentence is 25 tokens, so a third to
a half fell back to the dictionary and dragged the measured quality toward the dictionary's own —
making the model look barely better than it. Windows fixed it, and a failing window now falls back
alone rather than taking its sentence with it.

## Traps

1. **`paths.vectors` is spaCy's own key.** `[initialize] vectors` interpolates it and tries to load
   a spaCy `Vectors` object (E884). A lookup table the layer reads is not vectors the vocab
   attaches; use a custom `paths.vec_table`, as typology does.
2. **`basis.npz` and `fit_report.json` are per-GENERATION.** A v3 run into v1's work directory would
   rebuild thirteen v1 assets against a 32-language basis, and they would load and retrieve normally
   — a basis is only wrong relative to the rows it was fitted with. `--work` is required with
   `--sources`.
3. **The fold must reach the ARTEFACT, not just the build.** `stage_emit` took `key_norm` from the
   fit report, so a fold added after the fit was applied at build and recorded nowhere; Arabic came
   out at 41.2 % with the correct fold present in the code. It now comes from the declaration.
4. **A guard written from the deployment story can make the arm untrainable.** Two here fired on
   legitimate input: the no-rows guard (48 of 80 training languages have none) and the empty-doc
   guard (with one sentence per doc, a short sentence has nothing glossable). Enforcement of "the
   caller supplied nothing" belongs with the caller, which is the only place with a corpus view.
5. **The `lemma` fill is not a floor for every language.** For the six in the aligned-44 it is the
   upper bound. Labelling a run by its flag rather than by what it does to each language produced a
   wrong conclusion once in this file's history.
6. **`set_vectors_fill` and `set_gloss_debias` return a count and refuse at zero.** A silent no-op
   would score the lemma fill on a held-out language — all-OOV — and report it under the gloss
   fill's name.
7. **The predicted doc must carry LEMMA for the lemma fill.** `generic_corpus` copies UPOS, FEATS
   and LEMMA during training; an eval that copies only the first two makes the channel silently
   all-OOV. Caught by the layer's own refusal, not by any score.

## Files

| file | what |
|---|---|
| `scripts/fetch_vec_aligned44.sh` | the 32 training + 6 test aligned spaces |
| `scripts/build_vec_manifest_v3.py` | derives the corpus map from the v2 manifest; the four folds |
| `scripts/align_vectors.py` | `--sources`, `--work`, and the basis/fold refusals |
| `scripts/aligned_vectors.py` | `KEY_NORM` — one definition of each fold, read by builder and layer |
| `scripts/build_generic_vectors_v3.py` | merges 38 assets into one table; three vocabularies |
| `scripts/sud_generic_embed_v3.py` | `sud.GenericEmbed.v3`, the fills, the controls, the refusals |
| `scripts/make_generic_config_v3.py` | the five arms; `g3_base` built by calling v2's `build()` |
| `scripts/eval_generic_v3.py` | scoring under a named fill regime |
| `scripts/compare_generic_v3.py` | the group split |
| `scripts/estimate_gloss_shift_v3.py` | the shift, its mean, and the sample used for augmentation |
| `scripts/report_gloss_coverage_v3.py` | what fraction of each held-out language is fillable |
| `scripts/fetch_gloss_dicts_v3.sh` | 16 Wiktionary lemma→English bags |
| `metrics/generic_v3/` | the results |
