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
| `zh_sud_gsdboth` model, `assets_zh/.../*.relabeled*.conllu` | SUD_Chinese-GSD + SUD_Chinese-GSDSimp | CC BY-SA 4.0 |
| `ko_sud_gsd` model, `assets_ko/.../*.relabeled*.conllu` | SUD_Korean-GSD | CC BY-SA 4.0 |
| `id_sud_gsd` model, `assets_id/.../*.relabeled*.conllu` | SUD_Indonesian-GSD | CC BY-SA 4.0 |
| `la_sud_ittbproielperseus` model, `assets_la/la_ittbproiel-sud-*.conllu` | SUD_Latin-ITTB + SUD_Latin-PROIEL + SUD_Latin-Perseus | **CC BY-NC-SA** (NonCommercial — see below) |

Each source treebank's own `LICENSE.txt` is retained alongside its data.

## NonCommercial — the Latin model (`la_sud_ittbproielperseus`)

The Latin model is trained on the union of three SUD Latin treebanks, **all NonCommercial**:
**ITTB** (CC BY-NC-SA 3.0), **PROIEL** (CC BY-NC-SA), and **Perseus** (CC BY-NC-SA 2.5). Unlike
the other models — which are kept free of NonCommercial sources to stay commercially usable — the
Latin model and its derived data (`assets_la/la_ittbproiel-sud-*.conllu` and the released wheel)
are therefore licensed **CC BY-NC-SA (NonCommercial)**. Use it for non-commercial purposes only.

## NonCommercial exclusion — SUD_English-GUM

The English EWT+GUM development setup used **SUD_English-GUM**, which is **CC BY-NC-SA 4.0
(NonCommercial)**. To keep the published English model and data free of the NonCommercial
restriction, **GUM is excluded entirely**: the shipped `en_sud_ewt` model is retrained on
EWT only, and no GUM-derived sentences, gold, or metrics are committed. The development-time
EWT+GUM figures quoted in the docs are reported for method context only.

## Attribution

- Surface-Syntactic Universal Dependencies — https://surfacesyntacticud.github.io/
- Universal Dependencies — https://universaldependencies.org/
- Please cite the individual UD/SUD treebanks when using these models; their authors are credited
  in each treebank's `LICENSE.txt`.
