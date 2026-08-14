#!/usr/bin/env python3
"""Build `fa_vocalise`'s table by aligning a Persian pronunciation lexicon onto its spellings.

Persian has no `Vform`. Unlike Arabic -- where PADT ships the gold vocalisation beside the gold
morphology and the table is a straight extract -- there is no vocalised Persian text in this
project at all, and SUD_Persian-PerDT's `Translit=` is a mechanical romanisation of the
CONSONANTAL spelling (`nšst` for نشست), so it records no short vowels either. The table therefore
has to be RECONSTRUCTED from a pronunciation lexicon, which is what `fa_align.align` does.

SOURCES, cascaded. **KaamelDict** (116 629 words, GPL, MahtaFetrat/KaamelDict on HuggingFace) is
the coverage layer; **Tihu** (47 149 words, redistributed inside the MIT-licensed `PersianG2p`) is
the consistency layer and WINS on any word both hold, because it is one editor's single scheme.

⚠ KaamelDict's homograph metadata is thinner than its README implies, and this is worth knowing
before relying on it. Of 116 629 entries only **3 261** carry more than one pronunciation and only
**580** carry any POS at all -- and the POS list is not always aligned 1:1 with the pronunciation
list (شکر has four readings and two POS tags). The `prob` column is also suspect: for در it reads
`[10.0, 90.0]` against readings `[dar, dorr]`, i.e. it makes the rare literary *dorr* the likely
one, contradicting the README's own account of that very word. So POS is used ONLY where it aligns
one-to-one with the readings, `prob` is not used to pick, and everything else falls back to Tihu or
to the first reading. The genuine homograph disambiguation this component can do is therefore
small, and confined to the ~580 POS-annotated entries.

Tihu was chosen over WikiPron's Persian scrape deliberately: WikiPron has 40 183 rows
but only 10 728 distinct words, and it mixes Iranian, Dari, Tajik and Classical transcriptions
with mutually incompatible vowel systems (`ɔː`, `eː`, `oː`, aspirated stops, `ɹ` and `w`), so the
same word arrives with several pronunciations that are not alternatives in one variety. Tihu is
one consistent Iranian Persian scheme. `--wikipron` is available for anyone who wants to widen
coverage and can accept that mixture.

⚠ IT IS A ONE-PRONUNCIATION-PER-WORD DICTIONARY, and that caps what this component can be. The
thing that makes `ar_vocalise` worth putting in a *pipeline* -- morphology choosing between rival
vocalisations of one spelling -- has no data to work on here: Tihu is a JSON object, so a spelling
has exactly one reading by construction, and Persian's genuine homographs (کرم kerm/karam/kerem,
شکر šekar/šokr) are simply absent or arbitrarily resolved. So this table is a LOOKUP that happens
to live in a pipeline, not a disambiguator. Say so rather than implying otherwise.

TWO RUNGS:
    F   the form itself, aligned
    L   the LEMMA aligned, with its diacritics transferred onto a form it prefixes -- نِشَست +
        ند -> نِشَستند. Worth ~8 points of token coverage, and safe because it only ever adds marks
        to the stretch the lemma actually spells and leaves the affix bare.

The data is NOT bundled, on the same footing as `la_macronise`'s Morpheus table and
`ar_vocalise`'s PADT table: run this script to build one.

    python scripts/build_fa_vocalise_lut.py                     # needs PersianG2p installed
    python scripts/build_fa_vocalise_lut.py --tihu path/to/tihudictBIG.json
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_align import align   # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "fa_vocalise_lut.json.gz"
DIAC = set("َُِّْ")

# WikiPron's IPA -> Tihu's ASCII scheme. Deliberately lossy on the marks that encode a DIFFERENT
# variety (length on mid vowels, aspiration, stress) rather than a different phoneme of Iranian
# Persian: keeping them would multiply the inventory without adding a distinction the diacritics
# can express.
IPA = [("t͡ʃ", "C"), ("d͡ʒ", "j"), ("ɑ́ː", "A"), ("ɒ́ː", "A"), ("ɑː", "A"), ("ɒː", "A"),
       ("ɔ́ː", "o"), ("ɔː", "o"), ("íː", "i"), ("úː", "u"), ("iː", "i"), ("uː", "u"),
       ("eː", "e"), ("oː", "o"), ("t̪ʰ", "t"), ("t̪", "t"), ("d̪", "d"), ("kʰ", "k"),
       ("pʰ", "p"), ("ʃ", "S"), ("ʒ", "Z"), ("ʔ", "?"), ("ɾ", "r"), ("ɹ", "r"),
       ("ɡ", "g"), ("χ", "x"), ("ʁ", "q"), ("ɣ", "q"), ("ɢ", "q"), ("ɦ", "h"),
       ("ä", "a"), ("æ", "a"), ("á", "a"), ("ǽ", "a"), ("ɔ́", "o"), ("ɔ", "o"),
       ("ɪ́", "i"), ("ɪ", "i"), ("ʊ́", "u"), ("ʊ", "u"), ("é", "e"), ("í", "i"),
       ("ú", "u"), ("ó", "o"), ("ɵ", "o"), ("ŋ", "n"), ("w", "v"), ("j", "y")]


def to_tihu(ipa_row):
    out = []
    for sym in ipa_row.split():
        s = sym.replace("ʰ", "").replace("ʱ", "").replace("̥", "").replace("ː", "")
        for a, b in IPA:
            s = s.replace(a, b)
        s = "".join(c for c in s if c.isascii() and c.isalpha() or c == "?")
        if s:
            out.append(s)
    return out


def load_tihu(path):
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        import PersianG2p
        p = Path(PersianG2p.__file__).parent / "data" / "tihudictBIG.json"
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(
            "no Tihu dictionary. Either `pip install PersianG2p` (MIT; it redistributes Tihu) "
            "or pass --tihu path/to/tihudictBIG.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tihu", help="path to tihudictBIG.json; default = inside PersianG2p")
    ap.add_argument("--kaamel", help="path to KaamelDict.csv (GPL; fetch it yourself)")
    ap.add_argument("--wikipron", help="optional fas_arab_*.tsv to merge in (mixed varieties)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    entries = {w: p.split() for w, p in load_tihu(a.tihu).items()}
    print(f"tihu: {len(entries)} words")
    pos_entries = {}
    if a.kaamel:
        import ast
        import csv
        add = 0
        for r in csv.DictReader(open(a.kaamel, encoding="utf-8")):
            w = r["grapheme"]
            try:
                reads = [list(x) for x in ast.literal_eval(r["phoneme"] or "[]")]
                poss = ast.literal_eval(r["POS"] or "[]")
            except Exception:
                continue
            if not reads:
                continue
            # POS is only usable when there is exactly one tag per reading; see the note above.
            if len(poss) == len(reads) and len(set(poss)) > 1:
                for tag, ph in zip(poss, reads):
                    if tag and tag != "-":
                        pos_entries.setdefault((w, tag), ph)
            if w not in entries:        # Tihu wins on overlap
                entries[w] = reads[0]
                add += 1
        print(f"kaamel: +{add} words not already in tihu; "
              f"{len(pos_entries)} (word, POS) readings usable for homographs")
    if a.wikipron:
        added = 0
        for line in open(a.wikipron, encoding="utf-8"):
            w, p = line.rstrip("\n").split("\t")
            if w not in entries:                 # Tihu wins: it is the consistent source
                ph = to_tihu(p)
                if ph:
                    entries[w] = ph
                    added += 1
        print(f"wikipron: +{added} words not already in tihu")

    voc, failed = {}, 0
    for w, ph in entries.items():
        r = align(w, ph)
        if r is None:
            failed += 1
            continue
        # The aligner must never change the spelling -- only add marks. Checked per entry rather
        # than trusted, because a table that silently respells words would corrupt every output.
        if "".join(c for c in r if c not in DIAC) != w:
            failed += 1
            continue
        # Store the entry even when it needed NO mark. A Persian word spelled entirely with long
        # vowels (این, که) is fully determined as it stands, and dropping it would make the
        # component report "unknown" for a word it knows perfectly well -- which understated
        # coverage on the test set by 27 points before this was fixed.
        voc[w] = r
    print(f"aligned {len(entries) - failed}/{len(entries)} = "
          f"{(len(entries) - failed) / len(entries):.2%}; {len(voc)} carry marks")
    if a.stats:
        return
    pos_voc = {}
    for (w, tag), ph in pos_entries.items():
        r = align(w, ph)
        if r and "".join(c for c in r if c not in DIAC) == w and voc.get(w) != r:
            pos_voc["|".join((w, tag))] = r
    print(f"POS-conditioned readings that DIFFER from the default: {len(pos_voc)}")
    blob = {"F": sorted(voc.items()), "P": sorted(pos_voc.items())}
    with gzip.open(a.out, "wb", compresslevel=9) as fh:
        fh.write(json.dumps(blob, ensure_ascii=False).encode("utf-8"))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
