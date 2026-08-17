#!/usr/bin/env python3
"""Hand gold segmentation of the Heart Sutra (T08n0251), in KYOTO'S convention.

⚠ THIS IS A HAND ANNOTATION BY AN LLM, NOT AN AUTHORITATIVE EDITION. It is grounded in Kyoto's
attested tokens (counts below, over train+dev+test) but the residue is judgment and should be
reviewed by a specialist before any published number rests on it. The uncertain calls are listed at
the bottom of this docstring.

SCOPE. Only <div type="jing"> -- the scripture proper, 330 characters. T08n0251 also carries two
prefaces in <div type="xu"> (794 + 156 chars) which are NOT the sutra and must not be scored.

THE CONVENTION, READ OFF KYOTO RATHER THAN ASSUMED. Classical Chinese in this treebank is
overwhelmingly one character = one token; it merges only names, honorifics, ethnonyms and numerals.
Checked against the treebank, every NATIVE disyllabic compound in this text is UNMERGED there --
一切 0, 諸法 0, 無明 0, 老死 0, 三世 0, 諸佛 0, 恐怖 0, 顛倒 0, 夢想 0, 究竟 0, 罣礙 0, 真實 0,
眼界 0, 意識 0 -- so they are split here. The Buddhist units ARE merged, and the merge list follows
Kyoto's own counts: 菩薩 43, 波羅蜜 18, 阿耨多羅 34, 三藐 30, 三菩提 30, 般若 13, 菩提 3, 涅槃 2.

Note 阿耨多羅 + 三藐 + 三菩提 is THREE tokens, not one: that is Kyoto's segmentation of
anuttara-samyaksambodhi (each attested ~30x), and a gold that merged them would be scoring the
model against a convention its training data does not use.

FOUR DELIBERATE DEPARTURES FROM ATTESTED KYOTO TOKENS, each because following the letter of the
treebank would encode an error:
  * 波羅蜜多 (paramita) -- Kyoto attests 波羅蜜 (18) because the Diamond Sutra writes the term without
    the final 多. Splitting 多 off as its own token would strand half a syllable of the
    transliteration. Merged as four characters.
  * 舍利子 (Sariputra) -- 0 in Kyoto (the Diamond Sutra's interlocutor is Subhuti, 須菩提, 136x), but
    it is exactly parallel to the attested name pattern 孔子 / 孟子 / 夫子, which Kyoto DOES merge.
  * 菩提薩埵 (bodhisattva, full transliteration) -- 0 in Kyoto, which has only the contracted 菩薩.
  * the mantra -- 揭帝 / 般羅 / 般羅僧 / 莎婆訶 are all 0 in Kyoto TRAIN. The segmentation is NOT a
    guess: Kyoto's own TEST split contains KR6c0127, Kumarajiva's translation of this same sutra,
    and its annotators segment the parallel mantra as
        竭帝 | 竭帝 | 波羅 竭帝 | 波羅僧 竭帝 | 菩提 僧莎呵
    i.e. paragate is TWO tokens (波羅 + 竭帝), not one. An earlier version of this file merged
    般羅揭帝 and 般羅僧揭帝 as whole mantra words, which contradicted the treebank. Fixed to match.

UNCERTAIN, FLAG FOR REVIEW:
  * 觀自在 (Avalokitesvara) -- merged here as a proper name, but unlike the others it is a TRANSLATED
    name whose characters are semantically transparent (觀 "contemplate" + 自在 "at ease"), so
    觀 + 自 + 在 is defensible. 0 in Kyoto either way.
  * 般羅揭帝 / 般羅僧揭帝 -- merged as whole mantra words; segmenting 般羅 + 揭帝 is defensible.
  * 五蘊 -- split here (五 + 蘊) as numeral + noun; Kyoto merges numeral compounds like 五十 but this
    is not one.

REVERSIBILITY is asserted: concatenating the tokens must reproduce the source line exactly.
"""
import pathlib, sys

# Longest-first: 菩提薩埵 must beat 菩提, and 般羅僧 must beat 般羅.
MERGE = ["菩提薩埵", "阿耨多羅", "波羅蜜多",
         "三菩提", "舍利子", "觀自在", "莎婆訶", "般羅僧",
         "般若", "菩薩", "菩提", "涅槃", "三藐", "揭帝", "般羅"]

def segment(line, units):
    units = sorted(units, key=len, reverse=True)
    out, i = [], 0
    while i < len(line):
        for u in units:
            if line.startswith(u, i):
                out.append(u); i += len(u); break
        else:
            out.append(line[i]); i += 1
    return out

def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets_cbeta/heart_sutra_jing.txt")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "assets_cbeta/heart_sutra_gold.txt")
    lines = src.read_text(encoding="utf-8").rstrip("\n").split("\n")
    rows, n_tok, n_multi = [], 0, 0
    for ln in lines:
        toks = segment(ln, MERGE)
        assert "".join(toks) == ln, f"round trip failed on: {ln[:20]}"
        rows.append(" ".join(toks)); n_tok += len(toks)
        n_multi += sum(1 for t in toks if len(t) > 1)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} paragraphs, {sum(len(l) for l in lines)} chars, "
          f"{n_tok} tokens, {n_multi} multi-char ({n_multi/n_tok:.2%})")
    print("  round-trip asserted per paragraph")

if __name__ == "__main__":
    main()
