# SUD spaCy parsers

Small, CPU-only spaCy pipelines for **thirteen languages**, trained on **Surface-Syntactic Universal
Dependencies (SUD)** treebanks. They predict SUD relations (`subj`, `comp:obj`, `mod`, …) rather than
UD relations, and — the research focus of this repo — disambiguate the noncommittal `udep` on
adpositional and case-marked dependents into `comp:obl` (complement) vs `mod` (modifier).

Models ship as installable wheels: **fourteen**, since English ships twice at two provenances. Get
them from [Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases).

```bash
pip install https://github.com/SunflowerAI/sud-spacy-parsers/releases/download/v0.3.0/ta_sud_ttb_mwtt-0.1.0-py3-none-any.whl
```

```python
import spacy
nlp = spacy.load("en_sud_ewt")
doc = nlp("She put the book on the table.")
print([(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.head.text) for t in doc])
# "on" attaches to "put" and is labelled comp:obl vs mod — the ambiguity the
# baseline treebank leaves as the noncommittal `udep`.
```

## Released models

Every wheel carries the same pipeline shape — `tok2vec` → `parser` → `morphologizer` → `lemmatizer` →
`tagger`, plus per-language extras. **The tagger sits behind the morphologiser** because it reads
UPOS and FEATS ([`docs/xpos.md`](docs/xpos.md)); `package_sud.sh` refuses to package an arm where it
does not.

| Model | Language | Version | Treebank | Licence | Extra dependency |
|-------|----------|:-------:|----------|---------|------------------|
| `en_sud_ewt` | English | 0.2.0 | SUD_English-EWT | CC BY-SA 4.0 | — |
| `en_sud_ewt_gum` | English | 0.2.0 | + GUM's ten non-NC genres | CC BY-SA 4.0 | — |
| `zh_sud_gsd` | Chinese | 0.2.0 | SUD_Chinese-GSD | CC BY-SA 4.0 | `opencc` |
| `ja_sud_gsd` | Japanese | **0.3.0** | SUD_Japanese-GSD | CC BY-SA 4.0 | `sudachipy` + `sudachidict-core` |
| `ko_sud_gsd` | Korean | **0.3.0** | SUD_Korean-GSD | CC BY-SA 4.0 | `python-mecab-ko` § |
| `la_sud_ittb_proiel_perseus` | Latin | **0.3.0** | SUD_Latin-ITTB + PROIEL + Perseus | CC BY-NC-SA 4.0 | — |
| `sa_sud_vedic_ufal_dcs` | Sanskrit | **0.3.0** | SUD_Sanskrit-Vedic + UFAL + DCS | CC BY-SA 4.0 | `indic-transliteration` |
| `lzh_sud_kyoto` | Classical Chinese | 0.2.0 | SUD_Classical_Chinese-Kyoto (+ kanripo punctuation) | CC BY-SA 4.0 | — |
| `ar_sud_padt` | Arabic | 0.2.0 | SUD_Arabic-PADT | CC BY-NC-SA 4.0 | `camel-tools` + data ‡ |
| `fa_sud_perdt` | Persian | 0.2.0 | SUD_Persian-PerDT | CC BY-SA 4.0 | — |
| `id_sud_gsd` | Indonesian | 0.2.0 | SUD_Indonesian-GSD | CC BY-SA 4.0 | — |
| `yue_sud_hk` | Cantonese | 0.2.0 | SUD_Cantonese-HK | CC BY-SA 4.0 | `spacy-pkuseg` |
| `ta_sud_ttb_mwtt` | Tamil | 0.1.0 | SUD_Tamil-TTB + MWTT | CC BY-NC-SA 3.0 | — |
| `te_sud_mtg` | Telugu | 0.1.0 | SUD_Telugu-MTG | CC BY-NC-SA 3.0 | — |

Every model is matched to its treebank's tokenisation, so it runs on **raw text**. Which release each
version sits on, and which wheels have been re-uploaded in place, is in
[`docs/release-notes.md`](docs/release-notes.md) — read it before assuming `pip install -U` will
fetch a fix.

**English ships twice.** `en_sud_ewt_gum` adds the ten GUM genres whose sources are not
NonCommercial: +66 % training tokens, and it outscores the EWT-only wheel. Both are **CC BY-SA
4.0** — GUM's maintainer has confirmed the annotations are Georgetown's under CC BY, with the
NonCommercial term belonging only to the individual underlying documents. What the GUM wheel adds is
an **attribution obligation**: cite GUM, link <https://gucorpling.org/gum/>, and credit the
annotators and the text sources ([`NOTICE.md`](NOTICE.md)). Prefer it unless you want an EWT-only
provenance and the narrower attribution that comes with it.

## Results (test split)

Measured on the arm each wheel ships, identified by hashing `parser/model` out of the **downloaded**
wheel. UAS, LAS and `comp:obl` F are **gold-preproc** — over the treebank's own tokens, so they are
comparable across languages and independent of the tokeniser. `TOK` is raw end-to-end, which is where
the tokeniser is measured instead.

| Model | Language | UAS | LAS | `comp:obl` F | TOK (raw) |
|-------|----------|----:|----:|-------------:|----------:|
| `ja_sud_gsd` | Japanese | 92.0 | 90.0 | 72.9 | 99.4 |
| `fa_sud_perdt` | Persian | 91.0 | 86.3 | 79.8 | 99.1 |
| `en_sud_ewt_gum` | English (EWT+GUM) | 86.8 | 81.9 | 70.7 | 99.7 |
| `en_sud_ewt` | English | 86.3 | 81.3 | 70.9 | 99.6 |
| `te_sud_mtg` | Telugu | 85.3 | 69.1 | 14.3 | — ◊◊ |
| `id_sud_gsd` | Indonesian | 83.6 | 74.2 | 68.2 | 99.9 |
| `ar_sud_padt` | Arabic | 83.0 | 76.8 | 62.8 | 91.4 ‡ |
| `lzh_sud_kyoto` | Classical Chinese | 82.0 | 76.5 | 67.1 | 100.0 † ‖ |
| `ko_sud_gsd` | Korean | 80.1 | 74.6 | 48.6 | 99.8 § |
| `la_sud_ittb_proiel_perseus` | Latin | 80.0 | 73.2 | 64.9 | 100.0 ¶ |
| `yue_sud_hk` | Cantonese | 75.2 | 67.3 | 52.2 | 94.7 ◊ |
| `zh_sud_gsd` | Chinese | 73.8 | 69.0 | 31.1 | 96.8 ⚑ |
| `ta_sud_ttb_mwtt` | Tamil | 72.9 | 59.7 | 29.9 | 94.2 ※ |
| `sa_sud_vedic_ufal_dcs` | Sanskrit (classical prose) | 68.6 | 48.5 | 24.4 | 100.0 † ∴ |

**Read a row's note before comparing it with another.** Several of these numbers do not mean what
the column heading implies, and the differences are large:

| | |
|:--:|---|
| **∴** | **Sanskrit is measured on a different test set** from every other row — held-out UFAL classical prose, not its own treebank's test. Differencing it against anything here is meaningless. |
| **◊◊** | **Two Telugu columns are traps.** Its treebank carries no lemmas and almost no FEATS, so `lemma_acc` 100.0 is measuring an identity copy and `morph_acc` 98.2 is the base rate for predicting empty. |
| **§** | **Korean's `morph_acc` says nothing** — FEATS is populated on 4.7 % of tokens. And 34.5 % of test tokens are unseen strings, which parse 29.6 LAS below the rest. |
| **¶** | **The Latin headline spans two registers**: ITTB+PROIEL scores LAS 77.7, Perseus's classical poetry 53.9. |
| **※** | Tamil has no raw `TOK` row: its tokeniser rewrites its input, so `spacy evaluate` cannot align the two texts. 94.2 is strict token F from `scripts/eval_ta_tokenizer.py`. |
| **†** | Sanskrit and Classical Chinese segment into clause units, not sentences; both bundle a `clause_parser` for punctuated running text. |
| **‖** | Classical Chinese trains on a punctuation-restored Kyoto, traditional-only, keeping the annotators' own subtypes — so it is not comparable with earlier lzh figures. |
| **⚑** | Chinese segments with a trained character segmenter whose second channel is jieba's decision, read off a traditional jieba dictionary. |
| **‡** | Arabic ships a CAMeL-Tools ATB tokeniser for PADT's clitics, and needs the CAMeL data (GPL v2, not bundled). |
| **◊** | Cantonese has a test-only treebank, carved 80/10/10, so its figures are the noisiest here. |

Every one of these is written out in **[`docs/results-notes.md`](docs/results-notes.md)**, with the
measurements behind it. Full per-relation breakdowns are in `metrics/release/*.json`; every other
file under `metrics/` is a development arm, and several predate the generation they describe.

Corpus sizes are in [`docs/training-data.md`](docs/training-data.md) — worth a look before reading
too much into any row, since `te` trains on 5,097 tokens and `la` on 586,604.

## What each model gives you

`token.pos_` (UPOS), `token.morph`, `token.lemma_`, `token.tag_` (the treebank's XPOS) and the
dependency parse. Every layer above the parser was added on top of the frozen arm with its own small
encoder, so the dependency and XPOS output is unchanged from the parsing-only release — UPOS, morph
and lemma are purely added annotation layers
([`docs/layers-and-tokenisers.md`](docs/layers-and-tokenisers.md)).

Sanskrit is the one exception: it ships a **joint multi-task** arm, and it accepts **raw sandhied**
IAST or Devanagari, doing the whole front end itself. Devanagari in gives Devanagari `FORM`/`LEMMA`
with the romanisation on `token._.translit` / `token._.ltranslit`, and every token carries its
padapāṭha form on `token._.unsandhied` ([`docs/sanskrit.md`](docs/sanskrit.md)).

### SUD's own annotation layer

Ten of the fourteen models — all but `zh`, `ko`, `ta` and `te` — also predict SUD features that have
no home in a spaCy `Doc`. They go on the extension `token._.sud_misc`, a dict: `Idiom`/`InIdiom`,
`Subject`, `Reported`, `Shared`. Which of them a given model carries is decided per language **by
measurement**, so read `nlp.pipe_names` — a `sud_*` pipe is there only where it beat the
alternatives.

```python
doc = nlp("He bought a book and read it in the garden.")
print([(t.text, t._.sud_misc) for t in doc if t._.sud_misc])
# [('He', {'Shared': 'Yes'}), ('it', {'Shared': 'No'}), ('in', {'Shared': 'No'})]
# "He" is the subject of both conjuncts; "it" and "in the garden" belong to the second alone.
```

⚠ **`Shared` is not in `token.morph`** on a model whose pipeline lists `sud_shared` (`en`, `fa`,
`ar`, `id`, `lzh`, `la`). The treebanks put that feature in FEATS, so the morphologiser learned it
there and still emits it on models carrying no such pipe — but where one ships, it owns the feature
and clears the morphologiser's value, so a token has exactly one answer rather than two that
disagree. `sud_misc.feats_string(token)` renders it back into a FEATS cell for a CoNLL-U writer, and
`sud_misc.misc_string(token)` does the same for the other four keys, which belong in column 10. See
[`docs/sud-misc-layer.md`](docs/sud-misc-layer.md).

A small local web tester is in `webapp/` (`python webapp/server.py`, then open the printed URL); it
loads whichever model wheels you have installed.

## The research contribution: disambiguating `udep`

SUD labels adpositional and case-marked dependents of verbs with the noncommittal `udep`. We relabel
each as `comp:obl` (complement) or `mod` (modifier) using a local LLM via Ollama (no thinking,
temperature 0), then retrain and compare.

Relabelling lowers headline LAS by ~1–2 — the binary distinction is harder than the noncommittal
label — but lifts the per-label `comp:obl` F wherever the adpositional system is genuinely ambiguous.
`scripts/relabel_ext.py` widens the scope beyond verbs to adpositional dependents of **noun,
proper-noun and adjective heads**, clausal verb PPs, participial complex prepositions (*according
to*, *based on*), and Korean's **case-marked noun dependents of verbs**:

| Lang | `comp:obl` F — baseline → verb scope → **extended** | LAS (ext) |
|------|---------------------------------------------------:|----------:|
| id   | 0.463 → 0.565 → **0.703** | 0.750 |
| ko   | 0.169 → 0.247 → **0.386** | 0.565 |
| zh   | 0.190 → 0.307 → **0.356** | 0.684 |
| en   | 0.860 → 0.740 → **0.730** | 0.819 |

The headline: **Korean is *not* near-vacuous at this task** — its `comp:obl` signal lives on bare
case-marked noun dependents, which the verb-and-adposition view missed entirely. English is the lone
regression: it already disambiguated verb `comp:obl` well, so folding in noun and adjective heads
dilutes the class.

⚠ Each relabelling also rewrites the **test-set gold**, so `comp:obl` F has a moving denominator.
And the Korean row is the development-time morpheme tokenisation, which the shipped wheel no longer
uses. The full method, the per-language prompts, the derived rules and the negative results are in
**[`docs/udep-relabel.md`](docs/udep-relabel.md)**.

Sanskrit ships the **baseline** (un-relabelled, predicting `udep`), because its `comp:obl`/`mod`
signal is case-based and near-chance for the LLM.

## Reproduce

Requires **Python 3.12** — spaCy has no 3.14 wheels, and `pip install spacy` does not pull in
`click`, which spaCy imports directly (both pinned in `requirements.txt`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: re-download from https://grew.fr/download/ — see docs/training-data.md
python -m spacy convert assets/en_ewt-sud-train.conllu corpus/ --converter conllu -n 10

# Train and evaluate. Every language but English needs --gold-preproc: spacy evaluate
# re-tokenises raw text, and a mismatch with gold tokens collapses alignment.
python -m spacy train configs/config.cfg --output training/ \
  --paths.train corpus/en_sud-train.spacy --paths.dev corpus/en_sud-dev.spacy
python -m spacy evaluate training/model-best corpus/en_sud-test.spacy \
  --output metrics/en/metrics.json
```

Per-language drivers run in the order `retrain_seg.sh` → `train_morph.sh` → `train_lemma.sh` →
`train_sud.sh`, then `package_sud.sh`. See `CLAUDE.md` for the full driver map.

## Layout

```
configs/config*.cfg     training configs
scripts/                the pipeline: prep, training drivers, custom components, packaging
metrics/release/        spacy evaluate output for the arm each RELEASED wheel ships
metrics/<lang>/         every development arm (baseline / relabel / extended / layer)
caches/                 resumable LLM decision caches — expensive to regenerate, so committed
gold/                   hand-annotated benchmark sets for the comp/mod decision
assets*/                treebanks and derived data (mostly gitignored; see .gitignore)
```

The binary corpora (`corpus_*/`) and trained models (`training_*/`) are build artifacts that the
`scripts/` drivers regenerate; models are distributed as release wheels, not committed.

## Where the details live

`CLAUDE.md` is the map. `NEGATIVE-RESULTS.md` records ~20 measured dead ends — **check it before
retrying anything that looks obviously right.** Each doc below records at least one defect that the
ordinary metrics reported as healthy.

| doc | covers |
|---|---|
| [`results-notes.md`](docs/results-notes.md) | what is behind each number in the results table above |
| [`training-data.md`](docs/training-data.md) | corpus sizes, and what each treebank does and does not annotate |
| [`release-notes.md`](docs/release-notes.md) | what changed in each wheel, and which were re-uploaded in place |
| [`udep-relabel.md`](docs/udep-relabel.md) | the research contribution — comp/mod prompting, extended scope, derived rules |
| [`layers-and-tokenisers.md`](docs/layers-and-tokenisers.md) | the seg / morph / lemma layers and the per-language tokenisers |
| [`packaging-and-release.md`](docs/packaging-and-release.md) | `package_sud.sh`, wheel contents, the release audits |
| [`xpos.md`](docs/xpos.md) | one tagset per arm, and the tagger conditioned on UPOS+FEATS |
| [`sud-misc-layer.md`](docs/sud-misc-layer.md) | `Idiom` / `Subject` / `Reported` / `Shared`: which arm ships which, and why |
| [`vocalisation.md`](docs/vocalisation.md) | Arabic and Persian vocalisation, and the augmentation recipe |
| [`languages.md`](docs/languages.md) | English's two arms and two licences; Indonesian |
| [`latin.md`](docs/latin.md) | three treebanks, macrons in and out, and where the parser's errors actually are |
| [`sanskrit.md`](docs/sanskrit.md) | the raw-sandhied-text front end, the CSLiser, the joint multi-task arm |
| [`chinese-family.md`](docs/chinese-family.md) | Chinese, Classical Chinese and Cantonese |
| [`lzh-tokenisation.md`](docs/lzh-tokenisation.md) | Classical Chinese multi-character tokens and the trained segmenter |
| [`korean.md`](docs/korean.md) | eojeol tokenisation, the mecab-ko analyser channel, the sentenciser |
| [`dravidian.md`](docs/dravidian.md) | Tamil's two treebanks and akṣara tokeniser; Telugu's missing lemmas and MWTs |
| [`aligned-vectors.md`](docs/aligned-vectors.md) | the aligned cross-lingual vector release |

## Licence

Source code (`scripts/`, `webapp/`, `configs/`) is MIT (see `LICENSE`). The released models and the
committed treebank-derived data inherit their treebanks' terms — **CC BY-SA 4.0** for most, and
**CC BY-NC-SA** for Latin, Arabic, Tamil and Telugu, whose treebanks are NonCommercial at source.
SUD_English-GUM's five NonCommercial genres are excluded from `en_sud_ewt_gum` so that wheel stays
commercially usable; its annotations are CC BY and require attribution.

Full attribution and the per-treebank breakdown are in [`NOTICE.md`](NOTICE.md).
