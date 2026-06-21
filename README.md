# SUD spaCy parsers (small/CPU, four languages)

Small, CPU-only spaCy pipelines (`tok2vec` → `tagger` → `parser`) for **English, Chinese, Korean,
and Indonesian**, trained on **Surface-Syntactic Universal Dependencies (SUD)** treebanks. They
predict SUD relations (`subj`, `comp:obj`, `mod`, …) rather than UD relations, and — the research
focus of this repo — disambiguate the noncommittal `udep` on adpositional/case dependents into
`comp:obl` (complement) vs `mod` (modifier). Models ship as installable wheels (see
[Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases)).

## Results (test split)

UAS/LAS and `comp:obl` F are on gold tokens (gold-preproc, comparable across languages); TOK is
raw end-to-end token accuracy (how well the tokeniser matches the treebank on raw text).

| Model | Language | UAS | LAS | `comp:obl` F | TOK (raw) |
|-------|----------|----:|----:|-------------:|----------:|
| `en_sud_ewt` | English | 84.5 | 79.5 | 69.4 | 99.6 |
| `zh_sud_gsdsimp` | Chinese | 72.6 | 67.6 | 29.1 | 94.1 |
| `ko_sud_gsd` | Korean | 79.7 | 75.6 | 24.7 | 100.0 |
| `id_sud_gsd` | Indonesian | 83.6 | 74.2 | 61.6 | 99.9 |

Full per-relation breakdowns are in the `metrics_*.json` files.

## Layout

```
requirements.txt        spacy 3.8 + click + thinc-apple-ops (Apple Silicon CPU ops)
assets/                 downloaded SUD .tgz, extracted treebanks, merged + relabelled *.conllu
configs/config*.cfg     training configs (init config --optimize efficiency)
metrics_*.json          spacy evaluate output for every arm (baseline / relabel / extended)
training_*/model-best/  shipped models (see "Available models"); other arms regenerate via scripts/
```

The binary corpora (`corpus_*/`) and the per-language model variants are build artifacts — the
`scripts/` drivers regenerate any of them from the kept `*.conllu` + configs, so only the four
deployable models and the canonical metrics are kept in-tree.

## Available models

Four small CPU pipelines (`tok2vec` → `tagger` → `parser`), each matched to its treebank's
tokenisation so they run on **raw text** and predict the disambiguated `comp:obl`/`mod` labels
(not the noncommittal `udep`). They are distributed as installable wheels on the
[Releases](https://github.com/SunflowerAI/sud-spacy-parsers/releases) page.

| Package | Language | Treebank | Tokenisation | Licence |
|---------|----------|----------|--------------|---------|
| `en_sud_ewt`     | English    | SUD_English-EWT     | default rules | CC BY-SA 4.0 |
| `zh_sud_gsdsimp` | Chinese    | SUD_Chinese-GSDSimp | pkuseg (needs `spacy-pkuseg`) | CC BY-SA 4.0 |
| `ko_sud_gsd`     | Korean     | SUD_Korean-GSD      | mecab morphemes (needs `mecab-ko` + `MECAB_PATH`) | CC BY-SA 4.0 |
| `id_sud_gsd`     | Indonesian | SUD_Indonesian-GSD  | rule tokeniser (enclitics merged) | CC BY-SA 4.0 |

```bash
# install a model from the latest release (example: Chinese)
pip install https://github.com/SunflowerAI/sud-spacy-parsers/releases/latest/download/zh_sud_gsdsimp-0.1.0-py3-none-any.whl
pip install spacy-pkuseg          # Chinese tokeniser dependency
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
print([(t.text, t.tag_, t.dep_, t.head.text) for t in doc])
# "on" attaches to "put" and is labelled comp:obl vs mod — this model resolves the prep-dependent
# ambiguity that the baseline left as the noncommittal `udep`.
```

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

For a parser to work on raw text, the spaCy tokeniser must agree with the treebank. Each language is
handled by whichever direction is lossless, measured by the raw end-to-end token accuracy:

| Lang | Approach | raw token match |
|------|----------|----------------:|
| en   | default rule tokeniser — already matches EWT | ~0.995 |
| ko   | retokenise the treebank to mecab morphemes (functional-head structure) | 1.000 |
| id   | coarsen the treebank (merge enclitics into whitespace tokens) | 0.999 |
| zh   | pkuseg segmenter trained on GSDSimp + a GSD user dictionary | 0.941 |

Chinese is the only language with no lossless option, because word boundaries in unsegmented text
are inherently ambiguous; the other three reach an exact or near-exact match. Drivers:
`scripts/train_all_retok.sh` and `scripts/relabel_retrain_retok.sh`.

## Licence

Source code (`scripts/`, `webapp/`, `configs/`) is MIT (see `LICENSE`). The released models and the
committed treebank-derived data are **CC BY-SA 4.0**, inherited from the SUD/UD treebanks they
derive from; SUD_English-GUM (NonCommercial) is excluded so the models stay commercially usable.
Full attribution and the per-treebank breakdown are in [`NOTICE.md`](NOTICE.md).
