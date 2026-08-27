# The generic parser v3: a lexical channel you can fill with an English gloss

> **Status: seed 0 only.** Every number below is single-seed and none of it is a claim yet. Seeds 1
> and 2 are running. The repo's standing rule applies — one seed once reported +0.46 LAS on a
> channel whose three-seed mean was +0.04.

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

## Results (seed 0)

The baseline reproduces v2's published `g2_base`: **54.45** against **54.24**, so this harness and
v2's agree and the delta is read against a published number.

Held-in dev LAS: base 72.21, capacity control 72.05, shuffled 72.55, **`g3_vec` 73.89**,
`g3_vec_aug` 73.74. **Three different null channels converge to 72.0–72.6 and only real vectors move
it** — so the gain is information, not capacity. (A mid-training reading of the shuffled arm at step
11 000 said 70.39 and looked like "a wrong vector is actively harmful"; it caught up by convergence.
Read controls at convergence.)

Zero-shot, held-out and genus-disjoint:

| | lemma fill | gloss fill |
|---|---|---|
| **6 with real aligned rows** | | |
| `g3_vec` | **+4.51** | −0.86 |
| `g3_vec_aug` | +3.90 | **+0.74** |
| **10 with no rows, gloss-scorable** | | |
| `g3_vec` | −1.88 | −2.23 |
| `g3_vec_aug` | **−0.21** | −0.24 |
| **all 20, lemma fill** | `g3_vec` +0.17 · `g3_vec_aug` **+0.84** | |

**1. The channel works: +4.51 macro LAS** on languages whose genus never appears in training. Large
for one input, and the uplift v1 looked for and did not find.

**2. An empty channel is not free: −1.88.** This contradicts what the layer's own comment argued
while the guard was being softened — that 48 of 80 training languages being all-OOV would teach the
model to condition on the OOV dimension. It does not. v2's "an unfitted channel is not a neutral
one", arriving where it had been argued away.

**3. The gloss substitution does not transfer on its own: −0.86.** Geometrically sound and still a
loss, with a NEGATIVE correlation between fill rate and delta. cos 0.46 is a different distribution,
not a noisy version of the same one.

## The shift is a structured displacement, and that is what fixes it

Characterised on 239 748 Arabic pairs, so the augmentation was built from a measurement rather than
guessed:

| | |
|---|---|
| cos(source, gloss) | +0.4597 (sd 0.1845) |
| ‖mean shift‖ / mean ‖shift‖ | **34.6 %** — one constant direction |
| residual, top-8 of 128 dims | **36.3 %** of variance (6.2 % if isotropic) |
| cos after removing the mean | +0.5017 |

So **an isotropic-Gaussian augmentation would model the angle and none of the structure.**

The free correction that follows — subtract the constant direction from gloss rows at inference, no
retraining — was implemented and **does not work**: −0.86 → −1.05 on the six, −2.23 → −2.30 on the
rest. Raising the cosine to 0.50 buys nothing, so the mean displacement is not what the model fails
on. The residual is, and it is 65 % of the shift.

`g3_vec_aug` therefore displaces each training lemma vector by a **real shift vector sampled from
the measured 20 000**. The training signal is still the language's own aligned lemma vector — the
displacement is applied TO it, not substituted for it — and **no gloss enters training**. It moves
the gloss fill from −0.86 to +0.74 and the dead-channel cost from −1.88 to −0.21, at a cost of 0.61
on the lemma fill.

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
