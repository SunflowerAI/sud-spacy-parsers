# SUD spaCy parsers (small/CPU, eleven languages)

Small, CPU-only spaCy pipelines (`tok2vec` → `tagger` → `parser` → `morphologizer` → `lemmatizer`)
for **eleven languages** (English, Chinese, Korean, Indonesian, Persian, Sanskrit, Classical Chinese,
Japanese, Arabic, Latin, and Cantonese), trained on **Surface-Syntactic Universal Dependencies (SUD)** treebanks. They
predict SUD relations (`subj`, `comp:obj`, `mod`, …) rather than UD relations, and — the research
focus of this repo — disambiguate the noncommittal `udep` on adpositional/case dependents into
`comp:obl` (complement) vs `mod` (modifier). Models ship as installable wheels (see
[Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases)).

## Results (test split)

UAS/LAS and `comp:obl` F are on gold tokens (gold-preproc, comparable across languages); TOK is
raw end-to-end token accuracy (how well the tokeniser matches the treebank on raw text).

| Model | Language | UAS | LAS | `comp:obl` F | TOK (raw) |
|-------|----------|----:|----:|-------------:|----------:|
| `en_sud_ewt` | English | 84.4 | 79.6 | 70.8 | 99.6 |
| `zh_sud_gsd_simp_trad` | Chinese (simp+trad) | 74.3 | 69.3 | 32.6 | 94.9 |
| `ko_sud_gsd` | Korean | 65.5 | 56.5 | 38.6 | 99.8§ |
| `id_sud_gsd` | Indonesian | 83.6 | 74.2 | 61.6 | 99.9 |
| `fa_sud_perdt` | Persian | 90.6 | 87.2 | 79.2 | 99.1 |
| `sa_sud_vedic_ufal_dcs` | Sanskrit (classical prose) | 52.2 | 37.3 | 27.6 | 100.0†◊◊ |
| `lzh_sud_kyoto` | Classical Chinese (trad+simp) | 84.1 | 79.0 | 73.1 | 100.0† |
| `ja_sud_gsd` | Japanese | 91.1 | 88.2 | 69.6 | 99.4 |
| `ar_sud_padt` | Arabic | 84.2 | 78.4 | 63.4 | 91.4‡ |
| `la_sud_ittb_proiel_perseus` | Latin | 80.6 | 73.9 | 65.2 | 100.0¶ |
| `yue_sud_hk` | Cantonese | 74.2 | 65.6 | 26.7 | 94.7◊ |

Full per-relation breakdowns are in the `metrics_*.json` files.

† Sanskrit and Classical Chinese tokenise deterministically (TOK 100), but the Vedic/Kyoto
treebanks segment into punctuation-free **clause units** (句讀 / clause) with no in-text sentence
boundaries. Both models bundle a `clause_parser` component that splits punctuated input at its
boundary marks (。，；for Classical Chinese; daṇḍa ।॥ and . ? ! `|` `||` / // for Sanskrit), parses
each clause in isolation, and reattaches each mark as a `punct` dependent — recovering the
per-clause accuracy (79.0 / 54.8) on punctuated running text; only **unpunctuated** running text
collapses (LAS ~48 / ~41). `sa_sud_vedic_ufal_dcs` takes **raw sandhied Sanskrit in IAST or Devanagari** — it
segments and de-sandhis internally (see below). Persian runs fine on raw text (raw LAS 85.3).

‡ Arabic is heavily cliticised (PADT splits proclitic و/ف/ل/ب/ك and enclitics). `ar_sud_padt` bundles a **CAMeL-Tools ATB tokeniser** that reproduces PADT segmentation on raw text (token-F1 0.91, raw end-to-end LAS ~72 vs 78 on gold tokens). It requires the CAMeL data (GPL v2, not bundled): `pip install camel-tools` then `camel_data -i morphology-db-msa-r13 disambig-mle-calima-msa-r13`.

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
parser's `tok2vec` is initialised from the dual-script Mandarin model `zh_sud_gsd_simp_trad` and
fine-tuned — a free TAG/UAS/baseline-LAS lift on so little data; the segmenter is *not* helped by a
Mandarin warm-start (a local-feature CRF learns Cantonese boundaries from scratch just as well).

¶ The Latin figures are on the **combined ITTB+PROIEL+Perseus** test, which now spans two very
different registers. Broken out by sub-domain (gold-preproc): on the **ITTB+PROIEL** test the model
scores LAS **78.3** / UAS **83.8** / `comp:obl` F **69.1** — *better* than before Perseus was added
(LAS 77.7); the **Perseus** test (classical poetry — Virgil, Ovid, Phaedrus) is much harder at LAS
**54.6** / UAS **66.8**, which pulls the combined headline down. So adding Perseus *improves* the
original domain and *adds* poetry coverage the model previously lacked. Perseus's XPOS is blanked
(incompatible tagset), so XPOS/TAG is reported on ITTB+PROIEL only.

## Layout

```
requirements.txt        spacy 3.8 + click + thinc-apple-ops (Apple Silicon CPU ops)
assets/                 downloaded SUD .tgz, extracted treebanks, merged + relabelled *.conllu
configs/config*.cfg     training configs (init config --optimize efficiency)
metrics_*.json          spacy evaluate output for every arm (baseline / relabel / extended)
training_*/model-best/  shipped models (see "Available models"); other arms regenerate via scripts/
```

The binary corpora (`corpus_*/`) and the per-language model variants are build artifacts — the
`scripts/` drivers regenerate any of them from the kept `*.conllu` + configs, so only the
deployable models and the canonical metrics are kept in-tree.

## Available models

Small CPU pipelines (`tok2vec` → `tagger` → `parser` → `morphologizer` → `lemmatizer`). The English/Chinese/Korean/Indonesian
models are matched to their treebank's tokenisation so they run on **raw text** and predict the
disambiguated `comp:obl`/`mod` labels. They are distributed as installable wheels on the
[Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases) page.

| Package | Language | Treebank | `udep` | Tokenisation | Licence |
|---------|----------|----------|--------|--------------|---------|
| `en_sud_ewt`     | English    | SUD_English-EWT     | disambiguated (ext) | default rules | CC BY-SA 4.0 |
| `zh_sud_gsd_simp_trad` | Chinese    | SUD_Chinese-GSD + GSDSimp | disambiguated (ext) | pkuseg, both scripts (needs `spacy-pkuseg`) | CC BY-SA 4.0 |
| `ko_sud_gsd`     | Korean     | SUD_Korean-GSD      | disambiguated | mecab morphemes (needs `mecab-ko` + `MECAB_PATH`) | CC BY-SA 4.0 |
| `id_sud_gsd`     | Indonesian | SUD_Indonesian-GSD  | disambiguated | rule tokeniser (enclitics merged) | CC BY-SA 4.0 |
| `fa_sud_perdt`   | Persian    | SUD_Persian-PerDT   | disambiguated (ext) | rule tokeniser (eval gold-preproc) | CC BY-SA 4.0 |
| `sa_sud_vedic_ufal_dcs` | Sanskrit | SUD_Sanskrit-Vedic + UFAL | kept (baseline) | **accepts raw sandhied text**, IAST or Devanagari (needs `indic-transliteration`); segments and de-sandhis internally; Devanagari in gives Devanagari FORM/LEMMA + `Translit`/`LTranslit`; padapāṭha form on `token._.unsandhied` | CC BY-SA 4.0 |
| `lzh_sud_kyoto`  | Classical Chinese | SUD_Classical_Chinese-Kyoto (+ simplified, + kanripo punctuation) | disambiguated (ext) | character tokeniser (bundled) | CC BY-SA 4.0 |
| `ja_sud_gsd`     | Japanese   | SUD_Japanese-GSD    | disambiguated (ext) | SudachiPy (needs `sudachipy`+`sudachidict-core`) | CC BY-SA 4.0 |
| `ar_sud_padt`    | Arabic     | SUD_Arabic-PADT     | disambiguated (ext) | CAMeL ATB tokeniser (needs `camel-tools` + data) | CC BY-SA 4.0 |
| `la_sud_ittb_proiel_perseus` | Latin   | SUD_Latin-ITTB+PROIEL+Perseus | disambiguated (ext) | rule tokeniser, enclitic `-que` split (bundled) | CC BY-NC-SA § |
| `yue_sud_hk`     | Cantonese  | SUD_Cantonese-HK    | disambiguated (ext) | pkuseg (needs `spacy-pkuseg`); char fallback | CC BY-SA 4.0 |

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

**Both Han scripts.** Both models are trained on the union of a traditional and a simplified
treebank, so they parse **simplified and traditional** text alike (within ~0.2 LAS of each other on
either script). For Chinese this uses **two real treebanks** for the same sentences —
`SUD_Chinese-GSD` (the original traditional annotation) and `SUD_Chinese-GSDSimp` (its simplified
auto-conversion) — rather than re-traditionalising GSDSimp, since simplification is lossy
(many-to-one, e.g. 後/后→后). The ext relabel lives on GSDSimp; `scripts/transfer_relabel_gsd.py`
overlays it onto the aligned GSD tokens (the `comp:obl`/`mod` decision is script-independent), and
the bundled pkuseg segmenter is retrained on both (`models/zh_gsdboth_pkuseg`), lifting raw
**traditional** segmentation/LAS (TOK 93.2→95.7, LAS 51.0→56.3) with simplified unchanged. Classical
Chinese has no simplified counterpart treebank, so its simplified half is auto-converted from Kyoto
with `scripts/opencc_conllu.py` (OpenCC `t2s`, character-level and length-preserving, so token
alignment and every deprel/head are unchanged); it tokenises one Han character per token, so
simplified needs no segmenter change. `scripts/both_scripts_release.sh` regenerates both arms end to
end.

**Cantonese.** `yue_sud_hk` is a coverb/prepositional system like Chinese: the in-scope `udep`
adpositions are coverbs (喺 *at*, 畀 *dative*, 到 *goal*, 由 *from*, 根據 *according-to*), disambiguated by
the same verb-frame + temporal rules and qwen3:8b relabel. Its extended scope adds two clean
deterministic signals — the associative/genitive **嘅** (`你嘅牙` "your tooth" → mod, like Mandarin 的)
and the treebank's own temporal subtype **`udep@tmod`** (而家/今日/嗰陣時 → mod). Because SUD_Cantonese-HK
ships only a test split, it is carved 80/10/10 (`scripts/split_yue.py`, which also copies the empty
XPOS from UPOS); spaCy has no `yue` module, so `scripts/yue_tokenizer.py` registers one. With only
804 training sentences the parser's `tok2vec` is initialised from `zh_sud_gsd_simp_trad` and fine-tuned
(`config_yue.cfg`; `scripts/train_yue.sh`), which lifts TAG/UAS. The bundled **pkuseg** raw-text
segmenter (`scripts/train_pkuseg_yue.py`, swapped in by `bundle_yue_pkuseg.py`) is trained from
scratch on the gold tokens — fine-tuning it from the Mandarin segmenter was tried and ties
from-scratch, so the self-contained model ships.

```bash
# install a model from the latest release (example: Chinese)
pip install https://github.com/SunflowerAI/sud-spacy-parsers/releases/latest/download/zh_sud_gsd_simp_trad-0.1.0-py3-none-any.whl
pip install spacy-pkuseg          # Chinese / Cantonese tokeniser dependency
```

The English model ships **EWT-only**: SUD_English-GUM is CC BY-NC-SA (NonCommercial), so it is
excluded to keep these models commercially usable. See [`NOTICE.md`](NOTICE.md) for licensing.

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
nlp = spacy.load("en_sud_ewt")              # after: pip install en_sud_ewt-0.1.0-...whl
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

> The **English** row is the development-time EWT+GUM setup. The **shipped** `en_sud_ewt` model is
> retrained on EWT only (GUM is NonCommercial; see [`NOTICE.md`](NOTICE.md)) — its scores are in
> the Results table above.

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
| zh   | character segmenter: jackknifed corpus lexicon + jieba's BMES decision | 0.9210 |
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
derive from; SUD_English-GUM (NonCommercial) is excluded so the models stay commercially usable.
Full attribution and the per-treebank breakdown are in [`NOTICE.md`](NOTICE.md).
