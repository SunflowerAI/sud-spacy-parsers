#!/usr/bin/env python3
"""Build a TRADITIONAL jieba dictionary from jieba's own, for the zh segmenter's BMES channel.

WHY THIS EXISTS. jieba's shipped `dict.txt` is simplified — 349 046 entries, essentially none of
them traditional — while `zh_sud_gsd` is traditional end to end. The first fix was to ask jieba
about the `t2s` rendering of the whole chunk and keep the per-character answer for the original
text (`--jieba-t2s`), which recovered the vocabulary but left the channel answering a question
about a DIFFERENT STRING than the one being segmented: `t2s` is many-to-one, so 乾/幹/干 all reach
jieba as 干 and any distinction the traditional text draws is invisible to the lookup.

This builds the dictionary in the script the model actually works in, so the lookup — the part that
carries the vocabulary — sees the traditional text itself. Only jieba's OOV HMM still consults the
`t2s` rendering, because its emission probabilities are per CHARACTER and were estimated on
simplified text; that is handled in `zh_jieba_feature.set_dictionary(hmm_t2s=True)`.

**s2tw, not s2t** — the same conversion `zh_script.ZhTradTokenizer` applies to incoming simplified
input, so the dictionary is in the orthography the segmenter is actually handed, and it is GSD's
own: OpenCC's plain `s2t` writes 爲什麼 and 臺灣 where GSD (and `s2tw`) write 為什麼 and 台灣.

WHAT ELSE WAS MEASURED, on jieba's boundary decisions over the traditional GSD test (529 chunks):

    stock dict, traditional text                     F 0.8931     the defect
    stock dict, whole chunk via t2s  (0.2.0 ships)   F 0.9236
    jieba's own extra_dict/dict.txt.big              F 0.9176     its traditional half is s2t, and
                                                                  its HMM still reads traditional
    this dictionary, HMM on the raw text             F 0.9203     the HMM alone is the whole gap
    this dictionary + HMM via t2s                    F 0.9237     ships
    s2t ∪ s2tw ∪ s2hk variants, HMM via t2s          F 0.8990     more spellings is NOT more
                                                                  coverage — the spurious long
                                                                  matches cost more than the
                                                                  variants recover

⚠ Do NOT harvest anything here from the treebank. This dictionary is a conversion of an EXTERNAL
resource, which is what lets the channel skip the jackknifing the corpus lexicon needs: jieba's
vocabulary was not derived from our training split, so its train-time reliability already equals
its test-time reliability. A traditional word list harvested from GSD train would reintroduce
exactly the leak `--jackknife` exists to remove.

Licence: jieba is MIT (`scripts/vendor_jieba.py` carries the notice the wheel ships); a converted
dictionary is a derivative of it and travels under the same terms.

    python scripts/build_jieba_trad_dict.py [-o models/jieba_dict_trad.txt] [--config s2tw]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def build(out_path, config="s2tw"):
    import opencc
    import zh_jieba_feature as jf

    jieba = jf._import_jieba()
    src = pathlib.Path(jieba.__file__).parent / "dict.txt"
    if not src.is_file():
        sys.exit(f"no dict.txt beside {jieba.__file__} — nothing to convert")
    conv = opencc.OpenCC(config)

    entries, converted = {}, 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(" ")
        word, freq = parts[0], int(parts[1])
        tag = parts[2] if len(parts) > 2 else ""
        trad = conv.convert(word)
        converted += trad != word
        # Frequencies are jieba's own and carry over unchanged; where two simplified words converge
        # on one traditional spelling, the larger frequency wins rather than their sum, so no word
        # is made commoner than jieba ever saw it.
        prev = entries.get(trad)
        if prev is None or freq > prev[0]:
            entries[trad] = (freq, tag)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(f"{w} {f} {t}".rstrip() + "\n" for w, (f, t) in entries.items()),
        encoding="utf-8")
    print(f"  {src} ({sum(1 for _ in src.open(encoding='utf-8'))} entries, {config})")
    print(f"  -> {out_path}: {len(entries)} entries, {converted} spellings converted, "
          f"{out_path.stat().st_size / 1e6:.2f} MB")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="models/jieba_dict_trad.txt")
    ap.add_argument("--config", default="s2tw",
                    help="OpenCC config (default s2tw — what zh_script converts input with)")
    a = ap.parse_args()
    build(a.out, a.config)


if __name__ == "__main__":
    main()
