A language-agnostic SUD pipeline trained on **80 SUD 2.18 treebanks**: a morphologiser that predicts
FEATS from UPOS, feeding a dependency parser that reads UPOS, decomposed FEATS and a **trainable
per-language embedding**. You supply UPOS; the wheel supplies everything else.

```bash
pip install --force-reinstall https://github.com/SunflowerAI/sud-spacy-parsers/releases/download/generic-v0.1.0/xx_sud_generic-0.1.0-py3-none-any.whl
```

```python
import xx_sud_generic
from spacy.tokens import Doc

nlp = xx_sud_generic.load()            # ['morphologizer', 'tok2vec', 'parser']
doc = Doc(nlp.vocab, words=["the", "cat", "sat", "on", "the", "mat"])
for t, p in zip(doc, ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]):
    t.pos_ = p                         # UPOS is YOUR input
doc._.tb_lang = "en"                   # one of the 80 fitted languages
doc = nlp(doc)                         # FEATS and the parse come out
```

### ⚠ This asset was re-clobbered in place

The 0.1.0 asset originally shipped the parser alone; it now carries the morphologiser as well. **The
version number is unchanged, so `pip install -U` will NOT pull this.** Reinstall from the URL above,
or use `--force-reinstall`. That is the standing trade-off with clobbering in this repo, and it is
why the fourteen monolingual wheels took a version bump instead.

### What you supply, and what you do not

| column | who provides it | why |
|---|---|---|
| tokens | you | there is no tokeniser |
| **UPOS** | **you** | it does not transfer — see below |
| FEATS | the wheel | +13.9 points over predict-nothing on held-out languages |
| heads / deprels | the wheel | 30 coarsened SUD relations |
| lemmas | nobody | measured inert on unseen languages; no lemmatiser ships |

### A language it has not seen needs ten annotated sentences

The embedding tables have **32 spare rows**. `adapt_lang_embed` ships inside the wheel: assign a
spare row and fit it on a small sample while every other parameter stays frozen (enforced by
wrapping the optimizer, and verified — 0.000e+00 drift, and no row but the target moves).

Ten sentences — 129–188 tokens — moved Thai **+12.18** LAS, Georgian **+6.62**, Basque **+1.05**,
saturating by about fifty; 400 was no better than 100 anywhere.

Two cautions. An **unfitted** spare row is not neutral — on Georgian it cost 4 LAS against having no
channel at all — so fit it before use. And in these measurements the sample and the test set came
from the same treebank, so part of the gain is domain adaptation rather than language adaptation.
The model **refuses** an unseen language rather than silently substituting another's vector.

### What did not work, and is documented rather than hidden

**Typological conditioning fails its own control.** Four features — OV/VO, SV/VS,
head-marking/dependent-marking, sex-based noun classification — encoded two-hot. Against a
deliberately *deranged* profile the correct one is worth **−0.12 macro LAS**, and both score below a
channel carrying nothing. Profile accuracy is not the constraint: 200 annotated sentences reproduce
a full-treebank oracle profile to two decimals (56.39 vs 56.38) and an empty channel still matches
at 56.33. Grambank/WALS profiles, at 62 % field accuracy, are wrong enough to cost ~1.8 LAS.

**The embedding cannot be predicted from Grambank** — at 128 dimensions (held-out cosine 0.226; the
prediction parses exactly as well as doing nothing) or at 8 (cosine 0.415, and it parses 2.18 *below*
doing nothing). It can, however, be *read*: probing recovers 19 of 65 Grambank features above a
permutation control, including prepositions (0.94) and verb-final order (0.88). The embedding
contains the typology plus a treebank-specific residual, and the residual is the part that matters.

**UPOS tagging does not transfer.** A multilingual tagger over all 80 treebanks reaches 32–39 % on
held-out languages — no better than a single English tagger — because tagging is lexical.
Romanisation is worth ~2 points (uroman > wiktra) and cannot close a 55-point gap.

**Lemmatisation is inert.** Across six held-out languages and two architectures, an edit-tree
lemmatiser never deviated from copying the wordform by more than +0.31 points. It is not shipped.

Full write-up and controls: `docs/generic-parser-v2.md`.

### Licence

**CC BY-NC-SA 4.0.** 24 of the 80 training treebanks are NonCommercial — 276 891 of 880 919 training
tokens — so the union of the corpus licences is NonCommercial and ShareAlike.
