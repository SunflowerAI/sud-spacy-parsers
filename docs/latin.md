# Latin (`la_sud_ittb_proiel_perseus`)

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

Trains on a plain `cat` of **ITTB + PROIEL + Perseus** (each keeps its own sent_ids);
`add_perseus_la.sh` is the reproducible driver (`merge|macron|relabel|train`). Perseus ships only
train + test, so it is added train→train / test→test and dev stays ITTB+PROIEL.

- **XPOS normalisation (was: blanking).** The three treebanks use mutually-incompatible XPOS
  tagsets, and this used to be handled by DELETION -- `blank_perseus_xpos.py` blanked field 5 on
  the Perseus tail, leaving PROIEL's rival 23-value tagset beside ITTB's composite codes.
  `normalise_la_xpos.py` converts both onto ITTB's conventions instead; see `docs/xpos.md`.
  Orthogonal to the macron (FORM) and relabel (DEPREL) transforms.
- **Results (ext+macron union = release).** Apples-to-apples on the ITTB+PROIEL test: LAS
  77.7→**78.3**, UAS 83.1→83.8, `comp:obl` F ~69 — Perseus *improves* the original domain.
  Perseus-only test LAS ~54.6 (classical poetry, genuinely hard). The combined-test headline (LAS
  73.9) is lower only because the test now includes Perseus.
- **Macrons in.** One union parser handles plain + macronised input
  (`train_la_ext_macron.sh` trains on plain-ext ∪ macron-ext; `macronise_la.py` uses the Alatius
  Docker macroniser, `transfer_macrons.py` composes the FORM transform onto the ext deprels).
- **Licence: CC BY-NC-SA** — all three sources are NonCommercial, the only such released model.

## Orthographic augmentation: sampling replaces the two copies

The union above buys exactly the two spellings it contains, at twice the data, and says nothing
about the other axes on which printed Latin varies. `train_la_aug.sh` trains the same architecture
on the **macronised copy alone**, with `la_augment.py` rewriting each document into a freshly
sampled EDITION STYLE every epoch (`la_orth.py` holds the transforms, `la_orth.Style` a style).
Macron-stripping is exact — it reproduces the plain FORM on 586 604/586 604 tokens and leaves LEMMA
untouched — so the macronised treebank is a strict superset and the plain spelling is derived, never
stored. Four axes, each resting on something measured rather than assumed:

- **Macrons**, per word. **Breves**, per word on top: a breve marks a SHORT vowel, so the candidates
  are the vowels Alatius left unmarked, minus diphthong members (no independent length) and minus
  the glides (not vowels).
- **`u`/`v`.** Which `u` is the consonant is lexical (`silua` → *silva*, `minuere` stays vocalic), and
  the treebanks disagree in a useful way: **ITTB writes `u` throughout, PROIEL and Perseus write
  `v`**, so the v-writing half is labelled data for the u-writing half. `build_la_glide_lut.py`
  harvests it (29 734 types; answers 78.3 % of ITTB's u-tokens outright) and derives a
  (prev-prev, prev, next) context rule for the residue, **97.94 % accurate held out by type**. NB
  the usual 0.90-dominance bar is WRONG here and costs 3.1 points: word-initial `u`+`e` sits at 0.89
  (glide in every real word, non-glide only in the v-writing treebanks' own spelling slips). The bar
  is for rules that COMMIT an annotation; this one only picks which of two attested spellings to
  show the model.
- **`i`/`j`.** The one axis with **no in-corpus evidence** — not one `j` in 586 604 tokens — so it is
  a rule: `i` is the consonant when it opens the word or follows a syllabic vowel and a vowel
  follows. Three guards make it hold: no `ji` exists in Latin (saves `iis`, `ii`, `iit`); the `u` of
  `qu`/`ngu` and a glide `u` are not syllabic vowels (saves `quia`, and `uiam` = *viam*); and the
  `eo` "go" family is excluded **by LEMMA**, because `iens`/`ierunt` are shaped exactly like
  `iecit`/`iacere` and only the lexeme separates them (16 tokens, all correct). Output spot-checked:
  `ejus`, `hujusmodi`, `major`, `jussit`, `Isajae`, `Pompejo`, `trajecta`, archaic `quojusdam`.
- **`ae`/`oe` → `æ`/`œ`**, only where the pair is a real diphthong — **which the macrons already
  record**: Alatius writes hiatus `āēr`, `āeris`, `poētae` but diphthongal `aere`, `caelum`, so an
  unmarked literal `ae` is the diphthong and a marked one is not.
- **Sentence-initial capitals**, per document (an edition is consistent). Worth having: ITTB
  capitalises **0 %** of sentence openings, PROIEL 16 %, Perseus 28 %.

**Results.** `make_la_variant_conllu.py` renders the test set in each style — same trees, same gold,
only FORM moves — and `eval_la_variants.py` scores both arms across all of them (gold-preproc):

| | LAS union → aug | LEMMA union → aug | |
|---|---|---|---|
| plain | 72.26 → 71.72 | 90.96 → 90.90 | the two the union arm was built for |
| macron | 72.16 → 71.32 | 88.74 → 88.45 | |
| **breve** | **18.74 → 64.91** | **20.67 → 73.22** | one unseen character inside 78 % of words |
| vj | 71.18 → 71.77 | 86.72 → **90.78** | |
| lig | 70.55 → 71.67 | 88.08 → **90.64** | |
| caps | 71.46 → 71.86 | 88.89 → 90.56 | |
| **all axes at once** | **17.93 → 64.90** | 18.92 → 71.81 | |

So ~0.5–0.8 LAS on the two spellings the union covered, in exchange for the LAS spread across
orthographies collapsing from **54.4 to 7.0**. The lemmatiser gains most and loses nothing (−0.05 on
plain, +2.6 to +4.1 on the glide/ligature axes) — edit trees are literal string edits, so it was the
component least able to generalise across spellings on its own. **TAG is the real cost**: −2.7 on
plain, since ITTB's 1 952-label composite XPOS is the most form-sensitive target here. Dev, un­augmented
union: LAS 74.19 vs 75.29, pos 95.09 vs 95.49, `lemma_acc` 93.36 vs 93.60. Morph and lemma stack by
the usual freeze recipe (parser byte-identical at each layer).

⚠ **`max_epochs` must be `-1`, and that has two silent consequences.** At `0` spaCy's
`create_train_batches` does `examples = list(corpus(nlp))` ONCE and reshuffles that same list every
epoch, so a corpus-level augmenter samples a single style per document for the whole run — the run
looks normal and trains on one fixed perturbation. `-1` streams instead, but then (1) the training
loop stops shuffling, so the reader needs `shuffle = true` (harmless: under `sud.GoldTokCorpus.v1` a
document IS an example, so it is the same shuffle by another name), and (2) `init_nlp` initialises
from `islice(train_corpus(nlp), 100)` — which truncated the tagger to **639 of 1 952** labels and
killed training on the first batch carrying a missing one. Hence `init_aug_labels.py`, which
collects labels over several augmented passes. `spacy init labels` cannot substitute: it runs
through the same `init_nlp` and reproduces the truncation.

**The lemmatiser's labels are the ones that actually need the passes.** Tagger/parser/morphologiser
labels are properties of the TREES, which augmentation never touches. Edit trees are properties of
the FORMS, so `uītae`, `vitae` and `vītæ` → `uita` are three labels, and a missing one does NOT
raise — `get_loss` maps it to `tree2label.get(tree_id, 0)` and the token is quietly taught label 0.
Growth is sub-linear (union 18 512 trees → +1 pass 20 132 → +5 22 498 → +10 26 029), so 10 passes is
affordable and lands at 29 123 kept labels; measured, **0.50 %** of tokens in a fresh augmented pass
fall outside the kept set, against the union arm's own **1.19 %** — better than the baseline, not
worse.

**ADOPTED, 2026-08-09 (user decision).** The released `la_sud_ittb_proiel_perseus-0.2.0` is now the
augmented chain: `training_la_aug` → `_aug_morph` → `_aug_lemma` → `_aug_sud`, with `package_sud.sh`
naming it and `LA_BASE` to get back the union. Measured on the released arm (gold-preproc, plain
test, `metrics_release_la*.json`): combined LAS 72.26 → **71.72**, UAS 79.17 → 78.72, TAG 80.35 →
**77.61**, `comp:obl` F 64.80 → 64.75; ITTB+PROIEL 76.58 → **75.90**, Perseus 53.47 → **53.53**. That
is the bill; the benefit is the orthography table above, where the LAS spread falls from 54.4 to 7.0.
(Every TAG figure in this section is on the OLD mixed tagset. Since the XPOS normalisation the same
arm reads TAG **86.16** on the plain test — see `docs/xpos.md`; LAS/UAS/LEMMA are unchanged.)
Wheel 17.7 → 27.3 MB, almost all of it the lemmatiser's edit-tree inventory (18,512 → 29,123 labels).

**The SUD layer is trained through the SAME augmenter** (`configs/config_la_aug_sud.cfg`), not on the
union corpus: `sud_subject` reads NORM/PREFIX/SUFFIX/SHAPE, so a pipe trained on two fixed spellings
sitting on an orthography-robust parser would be the arm's own weak point. Grafting the augmenter
onto a `make_sud_config.py` config needs its two companions too — `max_epochs = -1` and
`shuffle = true` — for the reason recorded above: at `0` spaCy lists the corpus ONCE and a
corpus-level augmenter then samples one style per document for the whole run. The SUD pipes need no
`init_aug_labels`, because Yes/No/O are properties of the TREES, which augmentation never touches;
only the lemmatiser's edit-tree labels are properties of the FORMS.

⚠ **Promoting the arm REVERSED a ship decision, and re-measuring is why it was caught.** `Shared`
on the augmented base is trained **38.11** v rule 36.78 v morphologiser 10.23 — where on the union
base the table won, 35.85 v 35.10. The reversal is trustworthy in the direction it points: this is
the three-feature arm whose `model-best` is picked on the mean of Subject/Reported/Shared, the same
handicap that cost Shared ~5 points on the union base, so it runs AGAINST the winner here. The
candidate mask also reaches 48.84 % of gold against the union base's 45.0 %, i.e. the augmented
parser recovers the coordination the layer is defined over slightly better. `Subject` stays trained
(67.02 v the rule's 52.41) and `Reported` still ships nowhere (rule 17.65, trained 8.00, n=24).

**Macronised OUTPUT: the `la_macronise` component.** The inverse of the above. Alatius = RFTagger +
a Morpheus-derived lexicon; our pipeline already does the tagging half, so only the lookup remains.
`build_la_macron_lut.py` harvests it from the plain/macron treebank pair, keyed `L1 (form,upos,feats)`
/ `L2 (form,upos)` / `L3 (form)`, backoff-pruned 152 443 → 42 817 entries. **Two tables now,
cascaded**: the harvest falls through to **Morpheus itself**, fetched at runtime by
`fetch_morpheus()` (4 MB on the wire → ~2.2 MB in `~/.cache/sud-spacy/`).

| | harvest has the word (92.1 %) | it does not (7.9 %) | whole-token |
|---|---|---|---|
| harvested alone | 98.23 % | 52.46 % | 94.42 % |
| Morpheus alone | 93.98 % | 90.42 % | 93.71 % |
| **cascaded** | 98.23 % | 90.42 % | **97.63 %** |

On Perseus (OOV share 7.9 → 23.8 %): 87.02 / 95.75 / **97.33**. Agreement with Alatius on held-out
test is 94.09 % whole-token / 97.13 % per-vowel with predicted morphology — *agreement*, not gold
vowel length (Alatius is itself ~98–99 %). Morphology is matched through a nine-slot key
(`ud_key`/`ldt_key`) with a backoff LADDER rather than one exact key, because the tagger is imperfect
(`cano` → ADJ) and an exact key turns every mis-tag into a total miss. The S4/S3 suffix levels are
**gone from the builder** (still read for older tables): at 52.46 % they were barely better than a
coin toss, which is what the Morpheus fall-through replaced. **The bound was VOCABULARY, not
morphology** — OOV levels were 71 % of all errors from 8 % of tokens, and perfect morphology would
have bought +0.23.

**`malus` vs `mālus`: the `MP` level, and why the cascade is interrupted for it.** L3 answers 90 %
of tokens and is keyed on the STRING ALONE, so on a word whose vowel length depends on its part of
speech it returned the corpus majority and Morpheus — which knows the difference — was never
reached. `malus` ADJ and `mālus` NOUN, `liber` "book" and `līber` "free", were literally one
question. Two changes, together: `_RUNGS` gains two POS-bearing rungs **at the end** (so a token
with full FEATS, already answered by a more specific rung, is untouched), and the builder marks the
**4 094 POS-SPLIT forms** — those where part of speech alone settles the length and settles it
differently per part of speech. For those, and only those, the UPOS-aware answer jumps ahead of L3.
Morpheus table format 3; a format 1/2 cache lacks the list, so `MP` never fires and the cascade is
exactly what it was — `fetch_morpheus()` again to get it.

Three guards, each found by a token that went wrong:
- **Only a DECISIVE rung answers** (`rung_mask`), never `mask`'s form-wide majority fallback. That
  fallback was giving vocative `canis` the `cānīs` of `cānus` — displacing a correct answer with a
  *worse* kind of majority than the one being replaced.
- **Never inside an idiom.** SUD gives an idiom's head the IDIOM's part of speech and records it in
  `ExtPos`, so `satis` in `satis facit` is tagged VERB while the word is still the adverb "enough";
  reading that UPOS as the word's own produced `satīs`. The treebank flags it, so the guard is exact.
- Ambiguity that POS cannot settle (`os` "mouth"/"bone", `populus` "people"/"poplar") is not in the
  list and still gets the old answer, because a rung is stored only where it is decisive.

**Read the measurement carefully — agreement with Alatius is the wrong referee here.** It barely
moves (gold morphology 97.60 → 97.59 whole-token, predicted 97.39 → 97.34), and it *should* fall:
Alatius is RFTagger-predicted on exactly these hard words. Of the 24 held-out tokens where the new
answer differs from it, ~18 are ours right and its tagger wrong — `mēnse` not `mēnsē` (the
third-declension ablative is short), `ūtī` from `ūtor` not the conjunction `utī`, `capī`, `ācer`,
`audīte`, `īnsignis`; 2 are treebank POS errors we faithfully propagate (`regis` tagged VERB in
*Pharaonis regis Aegypti*); 3 are the iambically-shortened imperative `cave`. Taking Morpheus's
GOLD-POS answer as referee instead, on the 1 516 POS-split test tokens that have one:

    old (L3 corpus majority)      86.02 %          new (predicted UPOS + FEATS)   92.74 %

**+6.7 points on the words this is about**, and that is with the tagger at its weakest: UPOS
accuracy on POS-split tokens is **87.92 %** against 92.35 % overall, since these are precisely the
words that are hard to tag. Same shape as `_PARADIGM` below — measured agreement with Alatius falls
while real accuracy rises.

`_PARADIGM` is a small exceptionless override keyed on `(InflClass, Case, Number, final letter)`
(a-stem Nom/Voc sg `-a` short, Abl sg `-ā` long; o-stem Dat/Abl sg `-ō`; e-stem Abl sg `-ē`) plus
`_LEMMA_CLASS` (PROPN carries no `InflClass` in ITTB/PROIEL, so declension comes from the lemma's
ending). It exists because the table memorises pairs and cannot express a paradigm rule — nominative
`Gallia` came out `Galliā` because the treebank only attests the ablative. **The harvested data is
WRONG on these cells** (Alatius's RFTagger contradicting gold morphology), so the rule *lowers*
measured agreement (−0.03) while raising real accuracy. It faithfully transmits Case errors, so
`config={"paradigm": False}` disables it.

**The lookup key is now ORTHOGRAPHY-TOLERANT, which promoting the augmented arm made compulsory.**
The table is keyed on the treebank's own spelling, so `jussit`, `silva`, `cælum` and `mēnsĕ` all
missed and the component returned the form unchanged — on exactly the editions the parser had just
learnt to read. Two changes, both in `resolve`:

- **Both length marks come off the PRIMARY key**, not just macrons. A breve can only ever miss, and
  missing is not harmless: Morpheus's suffix levels answer almost anything, so `ŏstēnsum` was
  answered off its last four characters (mask 0) instead of falling through to the entry that knows
  it is `ostēnsum`. A wrong answer that pre-empts the right one is worse than no answer.
- **A ladder of fallback keys, least normalised first** (`key_ladder`): length-stripped, then
  ligatures expanded, then `j`→`i` / `v`→`u`. Order is load-bearing — folding the glides at the
  first step reroutes every `v` form to the `u` spelling, and the two are SEPARATE entries with
  different answers (`vitae` → `vītae`, `uitae` → nothing), because the treebanks disagree among
  themselves. Folding early cost 5.5 points on a breve-marked edition.

The mask is a bitmask over character positions, so each key carries a map back to the form: a
ligature is one character in the form and two in the key. The OUTPUT keeps the caller's orthography
and replaces only the macrons — `jussit` stays `jussit`, `cælum` stays `cælum`.

**A BREVE VETOES the inference, over every level and over the paradigm rule.** It is not noise to be
normalised away: it is the caller stating that this vowel is short, which is exactly the claim a
macron would contradict. So the breve positions are cleared from the mask last, after the lexicon,
Morpheus and `_PARADIGM` have all had their say, and they are carried through into the output —
`intĕllectam` comes back `intĕllēctam`, our answer where the caller said nothing and theirs where
they did, and `mĕnsĕ` comes back unchanged. The `-B` suffix on the level records that a breve
overruled something.

Whole-token agreement with Alatius, same gold throughout, FORM alone re-rendered:

    style   raw key   + fallback   + breve veto
    plain    93.78      93.78         93.78
    vj       95.91      95.92         95.92
    lig      92.50      92.93         92.93
    breve    70.79      93.78         94.47
    all      70.47      95.08         95.80

**Plain is unchanged to the decimal** — the raw key is still tried first, so nothing that answered
before answers differently. And breve now scores ABOVE plain, which is the point of honouring it:
a marked edition is telling the macroniser something it would otherwise have to guess, so it should
come out ahead of an unmarked one rather than merely level with it.

Purely additive: spaCy tokens are immutable, so it sets `token._.macron`/`doc._.macron` and never
touches `token.text`. `la_parse_macronised.py` joins parse + macrons into CoNLL-U or a table;
`--mode reparse` re-parses the macronised string (measured free by `eval_la_reparse.py` over 4300
sents — LAS 70.30 plain / 70.33 ours /
70.31 Alatius) but `attach` is the default for being one pass and parse-identical.

⚠ **The component ships WITHOUT its data** (licensing): Morpheus is CC BY-SA 3.0, which forbids
adding restrictions, and the la wheel is CC BY-**NC**-SA. `package_lemma.sh la` runs
`add_la_macronise.py … --no-lut`. Users get data by calling `fetch_morpheus()` once (no config — the
component finds the cache itself), or `bash scripts/build_la_macron.sh` for a harvested table to
cascade on top, or both; route (1) needs none of Docker, the Morpheus compile or RFTagger.
**Fetching is not redistributing** — GPL restricts distribution,
not use — which is why the easy route exists, and the cache sits outside the model directory so the
wheel stays clean. With no data it **degrades, it does not raise** (`require_data=False`): every
token passes through unchanged and ONE `RuntimeWarning` per component instance names both routes —
necessary now the pipe is in the DEFAULT pipeline, since a raising component would break every
ordinary `nlp(text)`. The warning's fetch line is built from `__name__` so it reads correctly from
inside the wheel. RFTagger (non-commercial) is used only to label the treebank offline.

