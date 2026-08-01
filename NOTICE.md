# Licences & attribution

This repository combines components under different licences.

## Source code — MIT
`scripts/`, `webapp/`, `configs/`, and the documentation are licensed under the MIT Licence
(see `LICENSE`), © 2026 Sunflower AI.

## Treebank-derived data and released models — CC BY-SA 4.0

The relabelled treebanks committed here (`*.relabeled*.conllu`), the per-language gold sets, and
the **released model wheels** are derivative works of Surface-Syntactic Universal Dependencies
(SUD) treebanks, which are themselves derived from Universal Dependencies. They are distributed
under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, the licence of
the underlying treebanks. You must give attribution and share derivatives alike.

| Component | Source treebank | Licence |
|-----------|-----------------|---------|
| `en_sud_ewt` model, `assets/en_ewt-sud-*.conllu` | SUD_English-EWT | CC BY-SA 4.0 |
| `zh_sud_gsd_simp_trad` model, `assets_zh/.../*.relabeled*.conllu` | SUD_Chinese-GSD + SUD_Chinese-GSDSimp | CC BY-SA 4.0 |
| `ko_sud_gsd` model, `assets_ko/.../*.relabeled*.conllu` | SUD_Korean-GSD | CC BY-SA 4.0 |
| `id_sud_gsd` model, `assets_id/.../*.relabeled*.conllu` | SUD_Indonesian-GSD | CC BY-SA 4.0 |
| `la_sud_ittb_proiel_perseus` model, `assets_la/la_ittbproiel-sud-*.conllu` | SUD_Latin-ITTB + SUD_Latin-PROIEL + SUD_Latin-Perseus | **CC BY-NC-SA** (NonCommercial — see below) |

Each source treebank's own `LICENSE.txt` is retained alongside its data.

## Digital Corpus of Sanskrit (DCS) — CC BY 4.0

The Sanskrit **CSLiser** (`models/sa_presegment*`, the saṃhitā → CSL pre-tokeniser) is trained on
classical and epic text from the **Digital Corpus of Sanskrit**, Oliver Hellwig et al.,
<https://github.com/OliverHellwig/sanskrit> (`dcs/data/conllu`), licensed **CC BY 4.0**.

| Component | Source | Licence |
|-----------|--------|---------|
| `models/sa_presegment_dcs`, `data_samhita/dcs.jsonl`, `models/sa_lexicon` | DCS — Rāmāyaṇa, Mahābhārata, Kathāsaritsāgara, Bhāgavatapurāṇa, Bṛhatkathāślokasaṃgraha, Hitopadeśa, Daśakumāracarita, Harṣacarita, Kumārasaṃbhava | CC BY 4.0 |

CC BY 4.0 requires attribution but not share-alike, so it composes with the CC BY-SA 4.0 above
(the combined work stays CC BY-SA 4.0). The DCS checkout itself (`assets_dcs/`) is a sparse clone
and is not redistributed here. Cite DCS as: Oliver Hellwig, *The Digital Corpus of Sanskrit (DCS)*,
2010–2026.

## NonCommercial — the Latin model (`la_sud_ittb_proiel_perseus`)

The Latin model is trained on the union of three SUD Latin treebanks, **all NonCommercial**:
**ITTB** (CC BY-NC-SA 3.0), **PROIEL** (CC BY-NC-SA), and **Perseus** (CC BY-NC-SA 2.5). Unlike
the other models — which are kept free of NonCommercial sources to stay commercially usable — the
Latin model and its derived data (`assets_la/la_ittbproiel-sud-*.conllu` and the released wheel)
are therefore licensed **CC BY-NC-SA (NonCommercial)**. Use it for non-commercial purposes only.

## Latin macronisation — builder shipped, lookup table NOT shipped

`scripts/la_macronise.py` restores vowel-length macrons (`token._.macron` / `doc._.macron`) by
looking each word up in a table keyed on the form plus this project's own predicted UPOS/FEATS.
**That table is never committed and never included in a wheel.** Its vowel-length data originates
in **Morpheus** (Perseus Project, **CC BY-SA 3.0 US**), reached via Johan Winge's
**latin-macronizer** (**GPL-3.0**).

CC BY-SA permits commercial use but forbids imposing further restrictions on the work. The Latin
model is CC BY-**NC**-SA (forced by its three NonCommercial treebanks, above), so bundling
Morpheus-derived content into that wheel would add precisely the restriction BY-SA rules out.
Rather than resolve that tension by guesswork, the repository ships only the **builder**, which is
our own code (MIT):

```bash
bash scripts/build_la_macron.sh     # macronise locally -> harvest table -> attach to a local model
```

The released wheel **does** ship the component's code (ours, MIT) via `spacy package --code`, so the
factory is registered and the component is opt-in — but it ships **no table**:

```python
nlp = spacy.load("la_sud_ittb_proiel_perseus")          # 5 pipes, no macroniser
nlp.add_pipe("la_macronise", config={"lut": "scripts/la_macron_lut.json.gz"})
nlp("Gallia est omnis divisa in partes tres.")._.macron
# -> 'Gallia est omnis dīvīsa in partēs trēs.'
```

`scripts/la_macron_lut.json.gz` and `build_la_macron/` are gitignored; the component raises rather
than silently returning unmacronised text if loaded without a table. A model you build this way
contains Morpheus-derived data — **keep it local and do not redistribute it**.

The macroniser's own tagger, **RFTagger** (Schmid & Laws), is licensed for education, research and
other **non-commercial** use only. It is used solely to label your treebank offline; the component
uses this project's morphologiser at inference, so RFTagger is not a runtime dependency.

## NonCommercial exclusion — SUD_English-GUM

The English EWT+GUM development setup used **SUD_English-GUM**, which is **CC BY-NC-SA 4.0
(NonCommercial)**. To keep the published English model and data free of the NonCommercial
restriction, **GUM is excluded entirely**: the shipped `en_sud_ewt` model is retrained on
EWT only, and no GUM-derived sentences, gold, or metrics are committed. The development-time
EWT+GUM figures quoted in the docs are reported for method context only.

## Attribution

- Surface-Syntactic Universal Dependencies — https://surfacesyntacticud.github.io/
- Universal Dependencies — https://universaldependencies.org/
- Morpheus (Perseus Project, CC BY-SA 3.0 US) — https://github.com/PerseusDL/morpheus
- latin-macronizer, Johan Winge (GPL-3.0) — https://github.com/Alatius/latin-macronizer
- RFTagger, Helmut Schmid & Florian Laws (non-commercial) —
  https://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/
- Please cite the individual UD/SUD treebanks when using these models; their authors are credited
  in each treebank's `LICENSE.txt`.
