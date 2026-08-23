# What is behind each number in the README's results table

Every row of the README's **Results (test split)** table carries a caveat, and several of them are
large enough to change what the number means. They live here rather than in the README so that the
table stays readable; each section links on to the doc that covers the area properly.

All figures are **gold-preproc** (over the treebank's own tokens) unless marked raw. `TOK` is the
one column that is raw, because a tokeniser is the one thing `gold_preproc` cannot measure.

---

## Chinese — the segmenter and its jieba channel

⚑ *Full detail: [`chinese-family.md`](chinese-family.md), [`layers-and-tokenisers.md`](layers-and-tokenisers.md).*

Chinese segments with a **treebank-trained character segmenter**, retrained on traditional GSD for
this arm: strict whole-token F **0.9196** on the traditional test, against 0.9210 for the simplified
segmenter on its own.

Its second input channel is jieba's segmentation decision, and because jieba's own dictionary is
simplified while this arm is traditional throughout, that channel reads a **traditional jieba
dictionary** — jieba's own converted with OpenCC `s2tw`, the same conversion applied to incoming
simplified input (`scripts/build_jieba_trad_dict.py`). Asking jieba about traditional text with its
own simplified dictionary would cost that channel boundary F 0.9237 → 0.8931. Only jieba's OOV HMM
still consults the `t2s` rendering, since its emission probabilities are per character and were
estimated on simplified text; that alone is worth 0.9203 → 0.9237.

Until 2026-08-22 the wheel instead converted the whole chunk to simplified before asking, which
scores the same (0.9236) but asked jieba about a string `t2s` had already collapsed — 乾, 幹 and 干
all reach it as 干. The regime and the dictionary both travel inside the segmenter, so a loaded
model cannot ask the question differently from the way it was trained, and the wheel does not grow:
the vendored jieba drops the simplified `dict.txt` it no longer opens.

Raw end-to-end on the traditional test (`metrics/release/metrics_release_zh_raw.json`): TOK 96.8
lenient / 92.0 strict, LAS 62.3.

## Sanskrit and Classical Chinese — clause units, not sentences

† *Full detail: [`sanskrit.md`](sanskrit.md), [`chinese-family.md`](chinese-family.md).*

Both tokenise deterministically (TOK 100), but the Vedic/Kyoto treebanks segment into
punctuation-free **clause units** (句讀 / clause) with no in-text sentence boundaries. Both models
bundle a `clause_parser` component that splits punctuated input at its boundary marks (。，；for
Classical Chinese; daṇḍa ।॥ and `.` `?` `!` `|` `||` `/` `//` for Sanskrit), parses each clause in
isolation, and reattaches each mark as a `punct` dependent — recovering the per-clause accuracy on
punctuated running text; only **unpunctuated** running text collapses.

`sa_sud_vedic_ufal_dcs` takes **raw sandhied Sanskrit in IAST or Devanagari** — it segments and
de-sandhis internally. Persian, by contrast, runs fine on raw text (raw LAS 85.3).

## Classical Chinese — restored punctuation, and one script

‖ *Full detail: [`chinese-family.md`](chinese-family.md), [`lzh-tokenisation.md`](lzh-tokenisation.md).*

**The Classical Chinese row is not comparable with pre-0.2.0 ones.** `lzh_sud_kyoto` now trains on
a punctuation-restored Kyoto — the treebank deliberately carries no punctuation, so the marks are
aligned in from the Kanseki Repository editions it was built from (CC BY-SA 4.0; see `NOTICE.md`) —
and its 句讀 units are merged into punctuation-delimited sentences wherever a derived rule licenses
it. The test set therefore contains punctuation tokens and longer sentences, so 82.9 / 77.2 / 66.5
is a *different measurement* from the pre-punctuation 84.1 / 79.0 / 73.1, not a regression against
it.

At 0.2.0 the arm is also **traditional-only** — the simplified half was an OpenCC conversion of the
same text, and dropping it costs the parser ~2.4 LAS but stops 遠 and 远 splitting one character's
counts; simplified input is converted at the pipeline boundary and converted back on output, so
either script still goes in and out.

Measured like for like on the same input, the punctuated arm is **+11.9 LAS** on punctuated editions
(the old one attaches content words to marks it has never seen) and **−2.2 LAS** on bare
unpunctuated 白文, both single-seed. It expects punctuated input; `clause_parser` runs with
`keep_marks=True`, which is coupled to this base. Lemmas come from a 163-entry variant-character
(異體字) table rather than a trained lemmatiser — 99.73 % vs 99.65 %, and 1.4 MB smaller.

⚠ The released lzh tokeniser splits 孔子, and **no standard metric can see it**, because
`gold_preproc` bypasses the tokeniser entirely. The trained character segmenter recovers 孔子/匈奴
(token F 0.9624 → 0.9825). See [`lzh-tokenisation.md`](lzh-tokenisation.md).

## Arabic — clitics, and a tokeniser with a dependency

‡ *Full detail: [`vocalisation.md`](vocalisation.md).*

Arabic is heavily cliticised (PADT splits proclitic و/ف/ل/ب/ك and enclitics). `ar_sud_padt` bundles a
**CAMeL-Tools ATB tokeniser** that reproduces PADT segmentation on raw text (token-F1 0.91, raw
end-to-end LAS ~72 vs 77 on gold tokens). It requires the CAMeL data (GPL v2, not bundled):

```bash
pip install camel-tools
camel_data -i morphology-db-msa-r13 disambig-mle-calima-msa-r13
```

The arm is vocalisation-augmented and ships the `Vform` table plus a trained `Idiom` pipe.
Augmentation costs are **not uniform across labels** — the rare ones pay first.

## Sanskrit — reported on classical prose, not Vedic

◊◊ *Full detail: [`sanskrit.md`](sanskrit.md).*

**Sanskrit is reported on held-out UFAL (classical prose), not Vedic.** The shipped arm is a joint
multi-task model — one shared encoder for tagger/parser/morphologizer/lemmatizer instead of the
freeze recipe the other arms use — and it was chosen for classical Sanskrit, which is the actual use
case, at a deliberate cost to Vedic. Reporting the Vedic figure would overstate what the model does
on the text people bring to it.

These numbers are on the 1843-token UFAL test (`corpus_sa_ufal_eval`). A smaller 494-token holdout
used during arm selection reads several points *lower*, and three different UFAL test sets span
about ten points, so treat any single classical Sanskrit figure as approximate — it rests on far
less data than any other row here.

⚠ **The release set is not cross-language comparable on this row.** Thirteen of the fourteen wheels
report their own treebank's test split; sa reports `corpus_sa_ufal_eval`. Differencing sa's LAS
against any other row is meaningless.

Sanskrit remains much the hardest language in the set: free word order, heavy compounding, and
sandhi that must be undone before parsing can start.

## Korean — eojeol tokens, and the morphemes they hide

§ *Full detail: [`korean.md`](korean.md).*

**Korean changed tokenisation at 0.2.0, so this row is not comparable with 0.1.0.** `ko_sud_gsd`
trains on the original `SUD_Korean-GSD` with spaCy's rule tokeniser (eojeol words), where the
earlier arm split each eojeol into mecab morphemes. The point is tokenisation fidelity: against that
treebank the shipped tokeniser scores TOK 99.8 (strict span match 0.95), where the morpheme arm
scored 0.31 — the old LAS 75.6 was measured against a *retokenised* treebank and never applied to
raw Korean.

**At 0.3.0 the parser reads the morphemes an eojeol hides**, supplied by mecab-ko at runtime through
`sud.KoAnalyserEmbed.v1`. That is where the jump from the 0.2.0 row comes from. The wheel therefore
**requires `python-mecab-ko`**. Sentence boundaries are learned and the arm ships a `senter`.

The cost, accepted deliberately, is the Korean case-particle relabel result: eojeol tokens fuse
noun+particle, so the signal that lifted `comp:obl` F on morphemes has nowhere to live at that
scope.

⚠ Two things the headline does not say. This treebank populates FEATS on only 4.7 % of tokens, so
`morph_acc` 95.6 is ~the base rate for predicting empty and says nothing; POS and lemma are real.
And **34.5 % of test tokens are unseen STRINGS, and they parse 29.6 LAS below the rest.**

## Cantonese — a test-only treebank

◊ *Full detail: [`chinese-family.md`](chinese-family.md).*

SUD_Cantonese-HK ships a **test split only** (1004 sentences), so `yue_sud_hk` is trained on a
deterministic 80/10/10 round-robin carve of it — the 100-sentence test makes its figures noisier
than the others. The Cantonese XPOS column is empty, so UPOS is copied into it and the tagger
predicts UPOS in `tag_`.

TOK 94.7 is the bundled pkuseg word segmenter, trained from scratch on the treebank's gold tokens
(vs 63 for the character fallback). The parser's `tok2vec` is initialised from the dual-script
Mandarin model and fine-tuned — a free TAG/UAS/baseline-LAS lift on so little data. The segmenter is
*not* helped by a Mandarin warm start: a local-feature CRF learns Cantonese boundaries from scratch
just as well.

## Latin — three treebanks, one tagset, and a spelling-robust parser

¶ *Full detail: [`latin.md`](latin.md), [`xpos.md`](xpos.md).*

The headline Latin figures are on the **combined ITTB+PROIEL+Perseus** test, which spans two very
different registers. Broken out by sub-domain on the released arm (gold-preproc, in
`metrics/release/metrics_release_la_ittbproiel.json` / `metrics_release_la_perseus.json`):

| slice | UAS | LAS | `comp:obl` F |
|---|---:|---:|---:|
| ITTB + PROIEL | 83.19 | 77.67 | 68.92 |
| Perseus (classical poetry — Virgil, Ovid, Phaedrus) | 66.09 | 53.91 | 45.30 |

Perseus is much the harder half and pulls the combined headline down. Adding it *improved* the
original domain and *added* poetry coverage the model previously lacked.

⚠ **Ignore `tag_acc` in the two slice files.** The slice corpora predate the XPOS normalisation
below, so the tagger is scored against gold it no longer targets (Perseus's is blanked outright,
which scores 0.00). Every other field in them is valid: only the parser moved between generations,
and POS/LEMMA/MORPH reproduce the previous release set to the decimal.

**All three treebanks share ONE tagset.** They arrive with mutually-incompatible XPOS — ITTB's Index
Thomisticus composite codes, PROIEL's 23 part-of-speech codes, Perseus's 9-position morphology
strings — and Perseus's used to be blanked outright. PROIEL and Perseus are now re-rendered as Index
Thomisticus codes derived from their own (form, lemma, UPOS, FEATS), with ITTB's own rows left
untouched; held out on ITTB that rendering reproduces the treebank's own gold tag 93.7 % exactly. On
the **ITTB test slice — the one span whose gold did not move, so the only like-for-like comparison —
TAG goes 90.68 → 92.92**, and the combined figure 77.61 → **86.16**. Blanking never removed Perseus
from the metric: spaCy reads CoNLL-U `_` as a literal tag, so those 10,964 test tokens were scored
against a gold value of `"_"`.

### Orthographic augmentation: the bill, and what it buys

Latin trains on **one copy of the macronised treebank, resampled into a fresh edition style every
epoch** (`scripts/la_augment.py`). Each pass rewrites the FORM column along five axes printed Latin
genuinely varies on: macrons, breves, `u`/`v`, `i`/`j`, `æ`/`œ`, and sentence-initial capitals. The
trees never move.

Measured between the two arms that differ only in this — the plain∪macron union arm and the
augmented one, both a generation behind the shipped parser — plain-spelling LAS goes 72.26 →
**71.72**. That is the bill. What it buys shows up only when the spelling moves — the same test
re-rendered in other edition styles, same trees, same gold, FORM alone changing:

| | union arm | augmented arm |
|---|---:|---:|
| plain | 72.26 | 71.72 |
| macron | 72.16 | 71.32 |
| **breve** | **18.74** | **64.91** |
| `u`/`v` + `i`/`j` | 71.18 | 71.77 |
| `æ`/`œ` ligature | 70.55 | 71.67 |
| sentence-initial capitals | 71.46 | 71.86 |
| **all five at once** | **17.93** | **64.90** |

The LAS spread across orthographies collapses from **54.4 to 7.0**. One unseen character inside 78 %
of words is what the breve row measures, and it is not exotic: school and teaching editions mark
quantity throughout. The lemmatiser gains most and loses nothing, edit trees being literal string
edits and so the component least able to generalise across spellings on its own. Cost in bytes: the
lemmatiser's edit-tree inventory grows 18,512 → 29,123 labels, because `vitae`, `vītae`, `uitae` and
`vītæ` are four trees for one lemma.

**The shipped 0.3.0 arm adds a lemma-vector channel on top of that**, reading predicted lemma vectors
plus per-feature morphology rather than surface forms alone: plain-spelling LAS **71.72 → 73.23**.
The table is sealed into the model's own bytes.

⚠ 63 % of Latin attachment errors sit in a non-projective sentence, and no headline metric says so.

## Tamil and Telugu — small treebanks, and two traps

*Full detail: [`dravidian.md`](dravidian.md).*

Both are **CC BY-NC-SA 3.0** and both are new at 0.1.0. Neither ships a SUD MISC layer, and that is
measured rather than skipped: ta `Subject` reaches P 75.0 % over EIGHT predictions, a 95 % interval
of [41 %, 93 %] that spans the floor, and `Shared` is capped at zero by a candidate mask that reaches
no gold at all.

**Tamil** trains on SUD_Tamil-TTB plus the test-only SUD_Tamil-MWTT, carved 80/10/10. The two
treebanks disagree about annotation, not merely tagset, so the TTB slice is reported separately in
`train_ta.sh`'s own eval. Its parser reads LEMMA + per-feature morphology (+1.34 LAS over a tight
capacity control). Its `sud.TamilSandhiTokenizer.v1` decomposes akṣaras, which makes sandhi splitting
ordinary segmentation: strict token F **0.8389 → 0.9420**.

⚠ **Tamil has no raw end-to-end row**, and cannot have one from `spacy evaluate`: the tokeniser
rewrites its input, so the predicted and reference texts differ and alignment raises `E949` — the
same reason `eval_sa_compound.py` exists one language over. Its token F above comes from
`scripts/eval_ta_tokenizer.py`, measured on raw `# text` over 173 test sentences.

⚠ **Two Telugu columns are traps, not achievements.** MTG carries **no lemmas at all** — the LEMMA
column is `_` on every token, and spaCy keeps `_` as a literal string — so `scripts/prep_te.py` falls
back to IDENTITY and `lemma_acc` 100.00 is measuring a copy. It carries almost no FEATS either (115
values in 6,465 tokens), so `morph_acc` 98.20 is ~the base rate for predicting empty. Neither says
anything about Telugu. There is therefore no lemma or morphology channel in the te arm.

⚠ MTG also shipped **no multiword tokens whatsoever** ("Word count: 6465, Token count: 6465" in its
own README), against Tamil's 9.67 % of orthographic words. That is an annotation policy, not a fact
about Telugu. `scripts/split_te_mwt.py` re-annotates 20 of them from the treebank's own evidence, and
`training_te_nomwt_*` keeps the unsplit control.

⚠ `min_action_freq = 30` — spaCy's default — **deletes most of the label inventory at this corpus
size**, silently, with the deleted labels' recall pinned to zero: 7 of ta TTB's 19 deprels, 19 of the
combined arm's 33, and 14 of te's 29. `scripts/make_dravidian_config.py` sets it to 1.

## English — two wheels, one licence

‽ *Full detail: [`languages.md`](languages.md).*

`en_sud_ewt_gum` is the second English wheel: it adds the ten GUM genres whose sources are not
NonCommercial, which is +66 % training tokens and ~+0.6 LAS on EWT's own test.

It shipped **CC BY-NC-SA 4.0** at v0.2.0 because GUM's `LICENSE.txt` opens "The treebank is licensed
under CC BY-NC-SA 4.0", which read strictly puts the *annotations* under NC whatever the document —
and annotations are what a trained model absorbs. GUM's maintainer has since settled it: the
annotations are Georgetown's, under **CC BY**, and the NC belongs only to the individual underlying
documents. Dropping the five NonCommercial genres is therefore enough, and this wheel is **CC BY-SA
4.0** — the same terms as `en_sud_ewt`, which it now outscores.

Its extra obligation is **attribution**: cite GUM, link <https://gucorpling.org/gum/>, credit the
annotators, and cite the text sources. See [`NOTICE.md`](../NOTICE.md). `en_sud_ewt` remains
available for anyone who wants an EWT-only provenance and the narrower attribution that comes with
it.

Merging the two exposed one tagset conflict, now fixed. EWT and GUM share the PTB tagset and agree on
every word class, but PTB reserves `,` for the comma and gives dashes, semicolons and ellipses `:` —
GUM follows that without exception, while EWT tags `;` as `,` 101 times out of 101. EWT's half now
follows the standard: accuracy on the affected punctuation goes **72.5 % → 83.0 %**, with the
headline TAG flat (94.19 → 94.20 — this is 0.3 % of the corpus). `en_sud_ewt` is unaffected: on its
own, EWT's convention is internally consistent.

## Both Han scripts, one script inside

*Full detail: [`chinese-family.md`](chinese-family.md).*

Both Chinese models read **simplified and traditional** text, but since 0.2.0 they do it by
normalising at the boundary rather than by training on both scripts. Each trains on traditional
alone; simplified input is converted to traditional before tokenisation and FORM/LEMMA are converted
back on the way out, so either script goes in and out while the model holds one vocabulary.

Training on both real treebanks — `SUD_Chinese-GSD` and its simplified auto-conversion
`SUD_Chinese-GSDSimp` — worked, but **split** that vocabulary: 22.7 % of the type inventory is a
cross-script twin (15,848 types collapse to 12,248 under `t2s`), so 個 and 个 never pooled their
counts. The conversion is safe in the direction that matters, because simplification is many-to-one:
`t2s` is a function, so the output round trip is exact for 99.98 % of simplified types, and the
ambiguous `s2t` direction — which can cost the parse but never the surface string returned — agrees
with the traditional gold on 98.6 % of tokens (~99.3 % once the `” / 」` quotation convention is set
aside). Classical Chinese went the same way and pays the same price.
`scripts/both_scripts_release.sh` regenerates the superseded both-scripts arms.

## Cantonese `udep` scope

*Full detail: [`udep-relabel.md`](udep-relabel.md).*

`yue_sud_hk` is a coverb/prepositional system like Chinese: the in-scope `udep` adpositions are
coverbs (喺 *at*, 畀 *dative*, 到 *goal*, 由 *from*, 根據 *according-to*), disambiguated by the same
verb-frame + temporal rules and qwen3:8b relabel. Its extended scope adds two clean deterministic
signals — the associative/genitive **嘅** (`你嘅牙` "your tooth" → mod, like Mandarin 的) and the
treebank's own temporal subtype **`udep@tmod`** (而家/今日/嗰陣時 → mod).
