# SUD spaCy parsers (small/CPU, eleven languages, twelve wheels)

Small, CPU-only spaCy pipelines (`tok2vec` → `tagger` → `parser` → `morphologizer` → `lemmatizer`)
for **eleven languages** (English, Chinese, Korean, Indonesian, Persian, Sanskrit, Classical Chinese,
Japanese, Arabic, Latin, and Cantonese), trained on **Surface-Syntactic Universal Dependencies (SUD)** treebanks. They
predict SUD relations (`subj`, `comp:obj`, `mod`, …) rather than UD relations, and — the research
focus of this repo — disambiguate the noncommittal `udep` on adpositional/case dependents into
`comp:obl` (complement) vs `mod` (modifier). Models ship as installable wheels — **twelve**, since
English ships twice at two licences (see
[Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases)).

## Released models

Twelve wheels over eleven languages (English ships twice — see the licence column). The version
is the one on the [Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases) page: seven
models were rebuilt at **0.2.0**, the other five are unchanged since **0.1.0** and install from that
release. Japanese is at **0.3.0**: its tagger was a no-op at inference (the tokeniser pre-sets every
tag and spaCy's tagger does not overwrite by default), and its encoder now reads the conjugation and
XPOS channels SudachiPy already supplies.

| Model | Language | Version | Treebank | Licence |
|-------|----------|:-------:|----------|---------|
| `en_sud_ewt` | English | 0.2.0 | SUD_English-EWT | CC BY-SA 4.0 |
| `en_sud_ewt_gum` | English | 0.2.0 | SUD_English-EWT + GUM (ten non-NC genres) | CC BY-SA 4.0 |
| `zh_sud_gsd` | Chinese | 0.2.0 | SUD_Chinese-GSD | CC BY-SA 4.0 |
| `ja_sud_gsd` | Japanese | 0.3.0 | SUD_Japanese-GSD | CC BY-SA 4.0 |
| `ko_sud_gsd` | Korean | 0.2.0 | SUD_Korean-GSD | CC BY-SA 4.0 |
| `la_sud_ittb_proiel_perseus` | Latin | 0.2.0 | SUD_Latin-ITTB + PROIEL + Perseus | CC BY-NC-SA 4.0 |
| `lzh_sud_kyoto` | Classical Chinese | 0.2.0 | SUD_Classical_Chinese-Kyoto (+ kanripo punctuation) | CC BY-SA 4.0 |
| `ar_sud_padt` | Arabic | 0.1.0 | SUD_Arabic-PADT | CC BY-SA 4.0 |
| `fa_sud_perdt` | Persian | 0.1.0 | SUD_Persian-PerDT | CC BY-SA 4.0 |
| `id_sud_gsd` | Indonesian | 0.1.0 | SUD_Indonesian-GSD | CC BY-SA 4.0 |
| `sa_sud_vedic_ufal_dcs` | Sanskrit | 0.1.0 | SUD_Sanskrit-Vedic + UFAL + DCS | CC BY-SA 4.0 |
| `yue_sud_hk` | Cantonese | 0.1.0 | SUD_Cantonese-HK | CC BY-SA 4.0 |

## Training data

Sentences and tokens in the CoNLL-U each released model is trained and evaluated on — the SUD
treebank after the `udep` relabel, which changes DEPREL only and so leaves every count untouched.
Range lines (multiword tokens) and empty nodes are not counted as tokens.

| Model | Train (sent / tok) | Dev (sent / tok) | Test (sent / tok) |
|-------|-------------------:|-----------------:|------------------:|
| `en_sud_ewt` | 12,544 / 204,578 | 2,001 / 25,148 | 2,077 / 25,094 |
| `en_sud_ewt_gum` | 20,273 / 340,324 | 3,023 / 43,580 | 3,014 / 43,403 |
| `zh_sud_gsd` | 3,997 / 98,614 | 500 / 12,665 | 500 / 12,010 |
| `ja_sud_gsd` | 7,050 / 168,333 | 507 / 12,287 | 543 / 13,034 |
| `ko_sud_gsd` | 4,400 / 56,687 | 950 / 11,958 | 989 / 11,677 |
| `la_sud_ittb_proiel_perseus` | 40,305 / 586,604 ¤ | 3,334 / 43,805 | 4,300 / 54,897 |
| `lzh_sud_kyoto` | 59,215 / 460,390 ‖ | 5,111 / 38,739 | 4,567 / 34,233 |
| `ar_sud_padt` | 6,075 / 223,881 | 909 / 30,239 | 680 / 28,264 |
| `fa_sud_perdt` | 26,196 / 452,496 | 1,456 / 25,147 | 1,455 / 24,133 |
| `id_sud_gsd` | 4,482 / 97,602 | 559 / 12,661 | 557 / 11,756 |
| `sa_sud_vedic_ufal_dcs` | 21,647 / 163,308 ∴ | 2,996 / 23,862 | 230 / 1,843 ∴ ◊◊ |
| `yue_sud_hk` | 804 / 11,158 ◊ | 100 / 1,499 | 100 / 1,261 |

¤ Latin trains on **one copy of the macronised treebank, resampled into a fresh edition style every
epoch** (`scripts/la_augment.py`) — not on two fixed spellings. Each pass rewrites the FORM column
along five axes printed Latin genuinely varies on: macrons, breves, `u`/`v`, `i`/`j`, `æ`/`œ`, and
sentence-initial capitals. The trees never move, so the token count is the treebank's own; what
changes is which spelling the model meets on any given epoch. Macron-stripping is exact
(586,604/586,604 tokens reproduce the plain FORM), so the plain spelling is *derived* rather than
stored and the macronised treebank is a strict superset. Dev and test are the plain half, so the
Results table understates what the arm is for — see ¶.

∴ **The Sanskrit row is the parser's data**, which is the whole of it for the UAS/LAS above.
DCS is much larger — 244,481 sentences / 1,732,852 tokens — but it carries **no dependency
annotation**, so it trains the **morphologiser and lemmatiser** only (and the tagger, whose XPOS is
a copy of UPOS on 100 % of tokens here, so it is predicting the same labels). Its docs are built
with no heads or deps at all, which is the only representation spaCy reads as genuinely missing —
blanking the columns to `_` would teach the parser a literal `_` label — so the parser takes no
gradient from them whatever. Read the DCS figure against `pos_acc`, `morph_acc` and `lemma_acc`,
never against LAS. The test row is the held-out **UFAL** set the model is reported on, not the
Vedic test the dev split comes from.

## Results (test split)

Measured on the arm each wheel ships (verified by hashing `parser/model` out of the released wheel
against the training directory). UAS/LAS and `comp:obl` F are **gold-preproc** — over the treebank's
own tokens, so they are comparable across languages and independent of the tokeniser; TOK is raw
end-to-end token accuracy, which is where the tokeniser is measured instead.

| Model | Language | Version | UAS | LAS | `comp:obl` F | TOK (raw) |
|-------|----------|:-------:|----:|----:|-------------:|----------:|
| `en_sud_ewt` | English | 0.2.0 | 86.3 | 81.3 | 70.9 | 99.6 |
| `en_sud_ewt_gum` | English (EWT+GUM) | 0.2.0 | 86.8 | 81.9 | 70.8 | 99.7 |
| `zh_sud_gsd` | Chinese | 0.2.0 | 73.3 | 68.9 | 28.7 | 96.9 ⚑ |
| `ja_sud_gsd` | Japanese | 0.3.0 | 92.0 | 90.0 | 72.9 | 99.4 |
| `ko_sud_gsd` | Korean | 0.2.0 | 65.6 | 56.8 | 35.5 | 99.8 § |
| `la_sud_ittb_proiel_perseus` | Latin | 0.2.0 | 78.7 | 71.7 | 64.7 | 100.0 ¶ |
| `lzh_sud_kyoto` | Classical Chinese | 0.2.0 | 82.9 | 77.2 | 66.5 | 100.0 † ‖ |
| `ar_sud_padt` | Arabic | 0.1.0 | 83.7 | 77.3 | 62.9 | 91.4 ‡ |
| `fa_sud_perdt` | Persian | 0.1.0 | 90.6 | 87.2 | 79.2 | 99.1 |
| `id_sud_gsd` | Indonesian | 0.1.0 | 83.6 | 74.2 | 68.2 | 99.9 |
| `sa_sud_vedic_ufal_dcs` | Sanskrit (classical prose) | 0.1.0 | 52.2 | 37.3 | 27.6 | 100.0 † ◊◊ |
| `yue_sud_hk` | Cantonese | 0.1.0 | 72.4 | 64.5 | 46.2 | 94.7 ◊ |

Full per-relation breakdowns for exactly these runs are in `metrics_release_*.json`; the other
`metrics_*.json` files hold the development arms these were selected from, and several of them
predate the shipped generation, so read the `metrics_release_*` set when you want the released
model's own scores.

⚑ Chinese segments with a **treebank-trained character segmenter**, retrained on traditional GSD
for this arm: strict whole-token F **0.9196** on the traditional test, against 0.9210 for the
simplified segmenter on its own. Its second input channel is jieba's segmentation decision, and
because jieba's own dictionary is simplified while this arm is traditional throughout, that channel
reads a **traditional jieba dictionary** — jieba's own converted with OpenCC `s2tw`, the same
conversion applied to incoming simplified input (`scripts/build_jieba_trad_dict.py`). Asking jieba
about traditional text with its own simplified dictionary would cost that channel boundary F 0.9237
→ 0.8931. Only jieba's OOV HMM still consults the `t2s` rendering, since its emission probabilities
are per character and were estimated on simplified text; that alone is worth 0.9203 → 0.9237. Until
2026-08-22 the wheel instead converted the whole chunk to simplified before asking, which scores the
same (0.9236) but asked jieba about a string `t2s` had already collapsed — 乾, 幹 and 干 all reach it
as 干. The regime and the dictionary both travel inside the segmenter, so a loaded model cannot ask
the question differently from the way it was trained, and the wheel does not grow: the vendored
jieba drops the simplified `dict.txt` it no longer opens. Raw end-to-end on the traditional test
(`metrics_release_zh_raw.json`): TOK 96.8 lenient / 92.0 strict, LAS 62.3.

⚠ **zh was re-uploaded on 2026-08-22 at the same version 0.2.0**, with the traditional jieba
dictionary above replacing the `t2s`-the-text channel. Only the tokeniser changed — every model
weight is byte-identical to the previous asset, verified by hashing them out of the downloaded
wheel — so the gold-preproc table above is untouched and only the raw figures moved. `pip install
-U` will NOT replace an older copy, since the version is unchanged; `--force-reinstall` will.

⚠ **The wheel published on 2026-08-08 could not segment at all** — it shipped with no
`tokenizer/segmenter/` directory and returned each input string as a single token. It has been
rebuilt and re-uploaded at the same version. If you installed `zh_sud_gsd` before 2026-08-09,
reinstall with `--force-reinstall`; `pip install -U` will not replace it, since the version is
unchanged. Every model weight is byte-identical between the two builds, so no score moved — the
gold-preproc row above never ran the tokeniser and was correct throughout.

† Sanskrit and Classical Chinese tokenise deterministically (TOK 100), but the Vedic/Kyoto
treebanks segment into punctuation-free **clause units** (句讀 / clause) with no in-text sentence
boundaries. Both models bundle a `clause_parser` component that splits punctuated input at its
boundary marks (。，；for Classical Chinese; daṇḍa ।॥ and . ? ! `|` `||` / // for Sanskrit), parses
each clause in isolation, and reattaches each mark as a `punct` dependent — recovering the
per-clause accuracy (79.0 / 54.8) on punctuated running text; only **unpunctuated** running text
collapses (LAS ~48 / ~41). `sa_sud_vedic_ufal_dcs` takes **raw sandhied Sanskrit in IAST or Devanagari** — it
segments and de-sandhis internally (see below). Persian runs fine on raw text (raw LAS 85.3).

‖ **The Classical Chinese row is not comparable with earlier ones.** `lzh_sud_kyoto` now trains on
a punctuation-restored Kyoto — the treebank deliberately carries no punctuation, so the marks are
aligned in from the Kanseki Repository editions it was built from (CC BY-SA 4.0; see `NOTICE.md`) —
and its 句讀 units are merged into punctuation-delimited sentences wherever a derived rule licenses
it. The test set therefore contains punctuation tokens and longer sentences, so 82.9 / 77.2 / 66.5
is a *different measurement* from the pre-punctuation 84.1 / 79.0 / 73.1, not a regression against
it. At 0.2.0 the arm is also **traditional-only** — the simplified half was an OpenCC conversion of
the same text, and dropping it costs the parser ~2.4 LAS but stops 遠 and 远 splitting one
character's counts; simplified input is converted at the pipeline boundary and converted back on
output, so either script still goes in and out.
Measured like for like on the same input, the new arm is **+11.9 LAS** on punctuated editions (the
old one attaches content words to marks it has never seen) and **−2.2 LAS** on bare unpunctuated
白文, both single-seed. It expects punctuated input; `clause_parser` runs with `keep_marks=True`,
which is coupled to this base. Lemmas come from a 163-entry variant-character (異體字) table rather
than a trained lemmatiser — 99.73 % vs 99.65 %, and 1.4 MB smaller.

‡ Arabic is heavily cliticised (PADT splits proclitic و/ف/ل/ب/ك and enclitics). `ar_sud_padt` bundles a **CAMeL-Tools ATB tokeniser** that reproduces PADT segmentation on raw text (token-F1 0.91, raw end-to-end LAS ~72 vs 77 on gold tokens). It requires the CAMeL data (GPL v2, not bundled): `pip install camel-tools` then `camel_data -i morphology-db-msa-r13 disambig-mle-calima-msa-r13`.

◊◊ **Sanskrit is now reported on held-out UFAL (classical prose), not Vedic.** The shipped arm is
a joint multi-task model — one shared encoder for tagger/parser/morphologizer/lemmatizer instead of
the freeze recipe the other arms use — and it was chosen for classical Sanskrit, which is the actual
use case, at a deliberate cost to Vedic. Reporting the Vedic figure (UAS 65.1 / LAS 51.4) would
overstate what the model does on the text people bring to it. These numbers are on the 1843-token
UFAL test (`corpus_sa_ufal_eval`); a smaller 494-token holdout used during arm selection reads
several points higher, so treat classical Sanskrit accuracy as approximate — it rests on far less
data than any other row here. Sanskrit remains much the hardest language in the set: free word
order, heavy compounding, and sandhi that must be undone before parsing can start.

§ **Korean changed tokenisation, so this row is not comparable with earlier ones.** `ko_sud_gsd`
now trains on the original `SUD_Korean-GSD` with spaCy's rule tokeniser (eojeol words), where the
previous arm split each eojeol into mecab morphemes. The point is tokenisation fidelity: against
that treebank the shipped tokeniser scores TOK 99.8 (strict span match 0.95), where the morpheme arm
scored 0.31 — the old row's LAS 75.6 was measured against a *retokenised* treebank and never applied
to raw Korean. Sentence boundaries are learned (raw SENT F 83.8, raw LAS 55.0). The cost, accepted
deliberately, is the Korean case-particle relabel result: eojeol tokens fuse noun+particle, so the
signal that lifted `comp:obl` F to 0.386 on morphemes has nowhere to live. NB this treebank populates
FEATS on only 4.7 % of tokens, so the arm's `morph_acc` 95.4 is ~the base rate for predicting empty
and says nothing; POS 83.1 and lemma 78.3 are real.

◊ SUD_Cantonese-HK ships a **test split only** (1004 sentences), so `yue_sud_hk` is trained on a
deterministic 80/10/10 round-robin carve of it — the 100-sentence test makes its figures noisier
than the others. The Cantonese XPOS column is empty, so UPOS is copied into it and the tagger
predicts UPOS in `tag_` (as with every model here, a `morphologizer` fills `pos_`/`morph` and a
`lemmatizer` fills `lemma_` — see *Use* below). TOK 94.7 is the bundled pkuseg
word segmenter (trained on the treebank's gold tokens; vs 63 for the character fallback). The
parser's `tok2vec` is initialised from the dual-script Mandarin model `zh_sud_gsd_simp_trad`
(the 0.1.0 Chinese arm) and fine-tuned — a free TAG/UAS/baseline-LAS lift on so little data; the segmenter is *not* helped by a
Mandarin warm-start (a local-feature CRF learns Cantonese boundaries from scratch just as well).

¶ The Latin figures are on the **combined ITTB+PROIEL+Perseus** test, which spans two very
different registers. Broken out by sub-domain on the released arm (gold-preproc, and in
`metrics_release_la_ittbproiel.json` / `metrics_release_la_perseus.json`): the **ITTB+PROIEL** test
scores LAS **75.9** / UAS **81.8** / `comp:obl` F **68.4**, and the **Perseus** test (classical
poetry — Virgil, Ovid, Phaedrus) is much harder at LAS **53.5** / UAS **65.4**, which pulls the
combined headline down. Adding Perseus *improved* the original domain and *added* poetry coverage
the model previously lacked — measured at the time as ITTB+PROIEL LAS 77.7 → 78.3, on an earlier
generation of the arm, so read that gain as the reason Perseus is in and not as a comparison against
the numbers above.

**All three treebanks now share ONE tagset.** They arrive with mutually-incompatible XPOS — ITTB's
Index Thomisticus composite codes, PROIEL's 23 part-of-speech codes, Perseus's 9-position
morphology strings — and Perseus's used to be blanked outright. PROIEL and Perseus are now
re-rendered as Index Thomisticus codes derived from their own (form, lemma, UPOS, FEATS), with
ITTB's own rows left untouched; held out on ITTB that rendering reproduces the treebank's own gold
tag 93.7 % exactly. On the **ITTB test slice — the one span whose gold did not move, so the only
like-for-like comparison — TAG goes 90.68 → 92.92**, and the combined figure 77.61 → **86.16**.
Blanking never removed Perseus from the metric: spaCy reads CoNLL-U `_` as a literal tag, so those
10,964 test tokens were scored against a gold value of `"_"` and the arm made 24.29 on them.
Parsing is untouched — only the tagger was retrained, on the frozen arm, so LAS/UAS/POS/LEMMA are
identical to the decimal.

**These are all plain-spelling numbers, and they are the augmentation's bill, not its benefit.**
Against the previous plain∪macron union arm the released one is LAS 72.26 → **71.72** on ordinary
input, and TAG cost 80.35 → 77.61 on the mixed tagset both were measured on; TAG paid most, ITTB's
composite XPOS being the most form-sensitive target here. (Since the tagset normalisation above,
the released arm reads TAG **86.16** on that same test — a different, harder target, so it is not
comparable with either figure in this sentence.) What it buys shows up only when the spelling moves — the same test
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

So the LAS spread across orthographies collapses from **54.4 to 7.0**. One unseen character inside
78 % of words is what the breve row measures, and it is not exotic: school and teaching editions
mark quantity throughout. The lemmatiser gains most and loses nothing (−0.05 plain, +2.6 to +4.1 on
the glide and ligature axes), edit trees being literal string edits and so the component least able
to generalise across spellings on its own. Cost in bytes: the wheel goes 17.7 → 27.3 MB, almost all
of it the lemmatiser's edit-tree inventory growing 18,512 → 29,123 labels, because `vitae`, `vītae`,
`uitae` and `vītæ` are four trees for one lemma.

## Layout

```
requirements.txt        spacy 3.8 + click + thinc-apple-ops (Apple Silicon CPU ops)
assets/                 downloaded SUD .tgz, extracted treebanks, merged + relabelled *.conllu
configs/config*.cfg     training configs (init config --optimize efficiency)
metrics_release_*.json  spacy evaluate output for the arm each RELEASED wheel ships
metrics_*.json          spacy evaluate output for every arm (baseline / relabel / extended)
training_*/model-best/  shipped models (see "Available models"); other arms regenerate via scripts/
```

The binary corpora (`corpus_*/`) and the per-language model variants are build artifacts — the
`scripts/` drivers regenerate any of them from the kept `*.conllu` + configs, so only the
deployable models and the canonical metrics are kept in-tree.

## Available models

Small CPU pipelines (`tok2vec` → `tagger` → `parser` → `morphologizer` → `lemmatizer`). Each
model is matched to its treebank's tokenisation so it runs on **raw text** and predicts the
disambiguated `comp:obl`/`mod` labels. Versions, dataset sizes and scores are in the three tables
at the top; this one records the `udep` scope, the tokeniser and the extra dependency each wheel
needs.

| Package | Language | Treebank | `udep` | Tokenisation | Licence |
|---------|----------|----------|--------|--------------|---------|
| `en_sud_ewt`     | English    | SUD_English-EWT     | disambiguated (ext) | default rules | CC BY-SA 4.0 |
| `en_sud_ewt_gum` | English    | SUD_English-EWT + GUM | disambiguated (ext) | default rules | CC BY-SA 4.0 ‽ |
| `zh_sud_gsd`     | Chinese    | SUD_Chinese-GSD     | disambiguated (ext) | treebank-trained character segmenter (needs `jieba`, `opencc`) ⚑ | CC BY-SA 4.0 |
| `ko_sud_gsd`     | Korean     | SUD_Korean-GSD      | disambiguated (ext) | eojeol words, rule tokeniser | CC BY-SA 4.0 |
| `id_sud_gsd`     | Indonesian | SUD_Indonesian-GSD  | disambiguated (ext) | treebank-trained character segmenter, enclitics split | CC BY-SA 4.0 |
| `fa_sud_perdt`   | Persian    | SUD_Persian-PerDT   | disambiguated (ext) | rule tokeniser (eval gold-preproc) | CC BY-SA 4.0 |
| `sa_sud_vedic_ufal_dcs` | Sanskrit | SUD_Sanskrit-Vedic + UFAL | kept (baseline) | **accepts raw sandhied text**, IAST or Devanagari (needs `indic-transliteration`); segments and de-sandhis internally; Devanagari in gives Devanagari FORM/LEMMA + `Translit`/`LTranslit`; padapāṭha form on `token._.unsandhied` | CC BY-SA 4.0 |
| `lzh_sud_kyoto`  | Classical Chinese | SUD_Classical_Chinese-Kyoto (+ kanripo punctuation) | disambiguated (ext) | character tokeniser, one Han character per token (bundled) | CC BY-SA 4.0 |
| `ja_sud_gsd`     | Japanese   | SUD_Japanese-GSD    | disambiguated (ext) | SudachiPy (needs `sudachipy`+`sudachidict-core`) | CC BY-SA 4.0 |
| `ar_sud_padt`    | Arabic     | SUD_Arabic-PADT     | disambiguated (ext) | CAMeL ATB tokeniser (needs `camel-tools` + data) | CC BY-SA 4.0 |
| `la_sud_ittb_proiel_perseus` | Latin   | SUD_Latin-ITTB+PROIEL+Perseus | disambiguated (ext) | rule tokeniser, enclitic `-que` split (bundled) | CC BY-NC-SA § |
| `yue_sud_hk`     | Cantonese  | SUD_Cantonese-HK    | disambiguated (ext) | pkuseg (needs `spacy-pkuseg`); char fallback | CC BY-SA 4.0 |

‽ `en_sud_ewt_gum` is the second English wheel: it adds the ten GUM genres whose sources are not
NonCommercial, which is +66 % training tokens and ~+0.6 LAS on EWT's own test. It shipped
**CC BY-NC-SA 4.0** at v0.2.0 because GUM's `LICENSE.txt` opens "The treebank is licensed under
CC BY-NC-SA 4.0", which read strictly puts the *annotations* under NC whatever the document — and
annotations are what a trained model absorbs. GUM's maintainer has since settled it: the
annotations are Georgetown's, under **CC BY**, and the NC belongs only to the individual underlying
documents. Dropping the five NonCommercial genres is therefore enough, and this wheel is
**CC BY-SA 4.0** — the same terms as `en_sud_ewt`, which it now outscores. Its extra obligation is
**attribution**: cite GUM, link <https://gucorpling.org/gum/>, credit the annotators, and cite the
text sources. See [`NOTICE.md`](NOTICE.md). `en_sud_ewt` remains available for anyone who wants an
EWT-only provenance and the narrower attribution that comes with it.

Merging the two exposed one tagset conflict, now fixed in this wheel. EWT and GUM share the PTB
tagset and agree on every word class, but PTB reserves `,` for the comma and gives dashes,
semicolons and ellipses `:` — GUM follows that without exception, while EWT tags `;` as `,` 101
times out of 101. So the same character in the same context carried different gold depending on
which treebank the sentence came from. EWT's half now follows the standard: accuracy on the
affected punctuation goes **72.5 % → 83.0 %**, with the headline TAG flat (94.19 → 94.20 — this is
0.3 % of the corpus) and everything else identical to the decimal. `en_sud_ewt` is unaffected: on
its own, EWT's convention is internally consistent.

§ The Latin model is trained on the union of three SUD Latin treebanks, **all NonCommercial**:
ITTB (CC BY-NC-SA 3.0), PROIEL (CC BY-NC-SA), and Perseus (CC BY-NC-SA 2.5). The model and its
derived data are therefore **NonCommercial (CC BY-NC-SA)**. Perseus ships only train + test (no
dev), so it is added train→train and test→test; the dev split stays ITTB+PROIEL. The three
treebanks use mutually-incompatible XPOS tagsets, so Perseus's sparse 9-position XPOS is blanked
(`scripts/blank_perseus_xpos.py`) to keep the tagger coherent — Perseus's UPOS and full dependency
annotation are kept. See [`NOTICE.md`](NOTICE.md) and `scripts/add_perseus_la.sh`.

Most models ship the **extended-scope disambiguated** parser; Sanskrit ships the **baseline**
(un-relabelled, predicts `udep`), because its `comp:obl`/`mod` signal is case-based and near-chance
for the LLM, so relabelling did not improve `comp:obl` F — confirmed on the classical UFAL test
set too (LLM 0.43 vs a 0.82 majority baseline on the ambiguous Ins/Acc/Gen residue). The Sanskrit
model takes **raw sandhied Sanskrit** — IAST or Unicode Devanagari — and does the whole front end
itself: a trained character tagger inserts the word / compound / coalescence boundaries, a
mechanical splitter turns those into tokens, and a trained edit-tree transducer recovers the
unsandhied (padapāṭha) form. Clay-Sanskrit-Library notation is used only as an **internal**
representation; no caller has to produce it. Devanagari input yields Devanagari `FORM`/`LEMMA` with
the romanisation on `token._.translit` / `token._.ltranslit` (UD's `Translit`/`LTranslit`), and
every token carries its padapāṭha form on `token._.unsandhied` and its character span in the raw
input on `token._.src_span`.

**Both Han scripts, one script inside.** Both Chinese models read **simplified and traditional**
text, but since 0.2.0 they do it by normalising at the boundary rather than by training on both
scripts. Each trains on traditional alone; simplified input is converted to traditional before
tokenisation and FORM/LEMMA are converted back on the way out, so either script goes in and out
while the model holds one vocabulary. Training on both real treebanks — `SUD_Chinese-GSD` and its
simplified auto-conversion `SUD_Chinese-GSDSimp` — worked, but **split** that vocabulary: 22.7 % of
the type inventory is a cross-script twin (15,848 types collapse to 12,248 under `t2s`), so 個 and
个 never pooled their counts. The conversion is safe in the direction that matters, because
simplification is many-to-one: `t2s` is a function, so the output round trip is exact for 99.98 % of
simplified types, and the ambiguous `s2t` direction — which can cost the parse but never the surface
string returned — agrees with the traditional gold on 98.6 % of tokens (~99.3 % once the `” / 」`
quotation convention is set aside). Classical Chinese went the same way and pays the same price:
dropping its OpenCC-converted simplified half cost the parser ~2.4 LAS, accepted for the pooled
vocabulary. `scripts/both_scripts_release.sh` regenerates the superseded both-scripts arms.

**Cantonese.** `yue_sud_hk` is a coverb/prepositional system like Chinese: the in-scope `udep`
adpositions are coverbs (喺 *at*, 畀 *dative*, 到 *goal*, 由 *from*, 根據 *according-to*), disambiguated by
the same verb-frame + temporal rules and qwen3:8b relabel. Its extended scope adds two clean
deterministic signals — the associative/genitive **嘅** (`你嘅牙` "your tooth" → mod, like Mandarin 的)
and the treebank's own temporal subtype **`udep@tmod`** (而家/今日/嗰陣時 → mod). Because SUD_Cantonese-HK
ships only a test split, it is carved 80/10/10 (`scripts/split_yue.py`, which also copies the empty
XPOS from UPOS); spaCy has no `yue` module, so `scripts/yue_tokenizer.py` registers one. With only
804 training sentences the parser's `tok2vec` is initialised from `zh_sud_gsd_simp_trad` (the 0.1.0
Chinese arm) and fine-tuned
(`config_yue.cfg`; `scripts/train_yue.sh`), which lifts TAG/UAS. The bundled **pkuseg** raw-text
segmenter (`scripts/train_pkuseg_yue.py`, swapped in by `bundle_yue_pkuseg.py`) is trained from
scratch on the gold tokens — fine-tuning it from the Mandarin segmenter was tried and ties
from-scratch, so the self-contained model ships.

```bash
# install a model from the latest release (example: Chinese)
pip install https://github.com/SunflowerAI/sud-spacy-parsers/releases/download/v0.2.0/zh_sud_gsd-0.2.0-py3-none-any.whl
pip install jieba opencc-python-reimplemented   # Chinese tokeniser + script conversion
```

English ships **twice**, and both wheels are CC BY-SA 4.0: `en_sud_ewt` is EWT-only, while
`en_sud_ewt_gum` adds the ten non-NonCommercial GUM genres for +66 % training tokens and ~+0.6 LAS
on EWT's own test. Prefer `en_sud_ewt_gum` unless you want an EWT-only provenance — it carries
GUM's CC BY attribution obligation. See [`NOTICE.md`](NOTICE.md) for licensing.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # spaCy lacks 3.14 wheels; use 3.12
pip install -r requirements.txt

# Data: SUD 2.18 EWT + GUM (CoNLL-U)
cd assets
curl -sSLO https://grew.fr/download/SUD_2.18/SUD_English-EWT.tgz
curl -sSLO https://grew.fr/download/SUD_2.18/SUD_English-GUM.tgz
tar xzf SUD_English-EWT.tgz && tar xzf SUD_English-GUM.tgz && cd ..

# Merge the two treebanks per split, then convert to spaCy binary
for s in train dev test; do
  cat assets/SUD_English-EWT/en_ewt-sud-$s.conllu \
      assets/SUD_English-GUM/en_gum-sud-$s.conllu > assets/en_sud-$s.conllu
  python -m spacy convert assets/en_sud-$s.conllu corpus/ --converter conllu -n 10
done

# Config, train (CPU), evaluate
python -m spacy init config configs/config.cfg --lang en \
  --pipeline tagger,parser --optimize efficiency --force
python -m spacy train configs/config.cfg --output training/ \
  --paths.train corpus/en_sud-train.spacy --paths.dev corpus/en_sud-dev.spacy
python -m spacy evaluate training/model-best corpus/en_sud-test.spacy --output metrics.json
```

## Use

```python
import spacy
nlp = spacy.load("en_sud_ewt")              # after: pip install en_sud_ewt-0.2.0-...whl
doc = nlp("She put the book on the table.")
print([(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.head.text) for t in doc])
# "on" attaches to "put" and is labelled comp:obl vs mod — this model resolves the prep-dependent
# ambiguity that the baseline left as the noncommittal `udep`.
```

Every model carries a `morphologizer` and a `lemmatizer`, so `token.pos_` (UPOS), `token.morph`
and `token.lemma_` are populated alongside `token.tag_` (the treebank's XPOS) and the dependency
parse. Both were added on top of the frozen parser/tagger (each with its own small encoder), so the
dependency and XPOS output is unchanged from the parsing-only release — UPOS, morph and lemma are
purely added annotation layers. The lemmatiser is a trainable edit-tree lemmatiser trained on the
treebank's `LEMMA` column.

### SUD's own annotation layer

Ten of the twelve models (all but `zh` and `ko`) also predict SUD features that have no home in a
spaCy `Doc`. They go on the extension `token._.sud_misc`, a dict: `Idiom`/`InIdiom`, `Subject`,
`Reported`, `Shared`. Which of them a given model carries is decided per language by measurement, so
read `nlp.pipe_names` — a `sud_*` pipe is there only where it beat the alternatives.

```python
doc = nlp("He bought a book and read it in the garden.")
print([(t.text, t._.sud_misc) for t in doc if t._.sud_misc])
# [('He', {'Shared': 'Yes'}), ('it', {'Shared': 'No'}), ('in', {'Shared': 'No'})]
# "He" is the subject of both conjuncts; "it" and "in the garden" belong to the second alone.
```

⚠ **`Shared` is not in `token.morph`** on a model whose pipeline lists `sud_shared` or
`sud_shared` (`en`, `fa`, `ar`, `id`, `lzh`, `la`). The treebanks put that feature in FEATS, so
the morphologiser learned it there and still emits it on the models that carry neither pipe — but
where one ships it owns the feature and clears the morphologiser's value, so a token has exactly one
answer rather than two that disagree. `sud_misc.feats_string(token)` renders it back into a FEATS
cell for a CoNLL-U writer, and `sud_misc.misc_string(token)` does the same for the other four keys,
which belong in column 10.

A small local web tester is in `webapp/` (`python webapp/server.py`, then open the printed URL);
it loads whichever model wheels you have installed.

## Multilingual extension & `udep` disambiguation

The pipeline extends to Chinese, Korean, and Indonesian SUD treebanks, and to the core research
task: relabelling the noncommittal `udep` on adpositional dependents of verbs as `comp:obl`
(complement) vs `mod` (modifier) via qwen3:8b, then retraining and comparing. Relabelling lowers
headline LAS by ~1–2 (the binary distinction is harder than the noncommittal label) but lifts the
per-label `comp:obl` F where the adpositional system is genuinely ambiguous — Indonesian +17,
Chinese +10 — while Korean *at this verb-ADP scope* (postpositional, `udep` ~96% temporal/causal
modifiers) looks near-vacuous. See `CLAUDE.md` for the architecture and the `metrics_*.json` files
for per-language results.

### Extending the disambiguation beyond verbs

`scripts/relabel_ext.py` widens the scope to the rest of the `udep` that is cleanly
disambiguable: adpositional dependents of **noun, proper-noun, and adjective heads**, clausal
verb PPs, participial complex prepositions (*according to*, *based on*), and the Korean
**case-marked noun dependents of verbs** (the case particle, read off the head-final eojeol,
drives the decision). Partitives (*one of them*) are deliberately left `udep` — the documented
SUD default. Retrained with `scripts/relabel_retrain_ext.sh`, this lifts `comp:obl` F further than the
verb-only relabel, with headline LAS flat:

| Lang | comp:obl F — baseline → verb-rl → **extended** | LAS (ext) |
|------|----------------------------------------------:|----------:|
| id   | 0.463 → 0.565 → **0.703** | 0.750 |
| ko   | 0.169 → 0.247 → **0.386** | 0.565 |
| zh   | 0.190 → 0.307 → **0.356** | 0.684 |
| en   | 0.860 → 0.740 → **0.730** | 0.819 |

The headline: **Korean is *not* near-vacuous** — its `comp:obl` signal lives on bare case-marked
noun dependents, which the verb-ADP-only view missed. English is the lone regression: it already
disambiguated verb `comp:obl` well, so folding in noun/adjective heads dilutes the class. (Each
relabelling also rewrites the test-set gold, so `comp:obl` F has a moving denominator.)

> The **English** row is the development-time EWT+GUM setup, on the UNFILTERED treebank. The
> shipped wheels are retrained: `en_sud_ewt` on EWT only, `en_sud_ewt_gum` on EWT plus GUM's ten
> non-NonCommercial genres (see [`NOTICE.md`](NOTICE.md)) — their scores are in the Results table
> above.

### Tokeniser–treebank matching

For a parser to work on raw text, the spaCy tokeniser must agree with the treebank. The direction is
chosen per language by whether the treebank tokenisation is a deterministic function of the text —
where it is, a rule reaches the ceiling and a trained segmenter cannot beat it; where it is not, the
tokeniser is trained. Strict token F against the treebank's own test set:

| Lang | Approach | token F |
|------|----------|--------:|
| lzh  | one Han character = one token (bundled) | 1.000 |
| ko   | eojeol words, spaCy's rule tokeniser | 0.9977 |
| id   | treebank-trained character segmenter, enclitics **split** | 0.9954 |
| la   | rule tokeniser + the enclitic `-que` split (bundled) | 0.9944 † |
| en   | default rule tokeniser — already matches EWT | 0.991 |
| yue  | pkuseg trained from scratch on Cantonese | 0.95 ‡ |
| zh   | character segmenter: jackknifed corpus lexicon + jieba's BMES decision off a traditional jieba dictionary | 0.9196 |
| ar   | CAMeL ATB tokeniser, splitting PADT's clitics | 0.91 |
| sa   | CSLiser + de-sandhi front end, from raw sandhied text | 0.8699 |
| fa, ja | rule tokeniser; SudachiPy | — |

† Perseus (classical orthography); ITTB is 0.9924. ‡ word F1, not strict token F.

Chinese and Sanskrit are the two with no lossless option — word boundaries in unsegmented text and
sandhi junctions in continuous saṃhitā are both inherently ambiguous, so their segmenters are trained
and their figures are ceilings on everything downstream. Everywhere else the tokenisation is
recoverable by rule, and matching the tokeniser to the treebank has consistently beaten re-tokenising
the treebank: `id` once merged its enclitics and `ko` once split into mecab morphemes, both since
replaced. Drivers: `scripts/retrain_seg.sh` per language, plus `scripts/train_all_retok.sh` and
`scripts/relabel_retrain_retok.sh` for the superseded matched-tokenisation arms.

## Licence

Source code (`scripts/`, `webapp/`, `configs/`) is MIT (see `LICENSE`). The released models and the
committed treebank-derived data are **CC BY-SA 4.0**, inherited from the SUD/UD treebanks they
derive from — except the Latin and Arabic wheels, whose treebanks are NonCommercial at source
(**CC BY-NC-SA**). SUD_English-GUM's five NonCommercial genres are excluded from `en_sud_ewt_gum`
so that wheel stays commercially usable; its annotations are CC BY and require attribution.
Full attribution and the per-treebank breakdown are in [`NOTICE.md`](NOTICE.md).
