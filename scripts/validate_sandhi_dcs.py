#!/usr/bin/env python3
"""Validate the forward sandhi engine against DCS's REAL editorial sandhied text.

WHY IT MATTERS BEYOND ANY AUGMENTER. `docs/sanskrit.md` states plainly that no gold sandhied text
exists for the Vedic half, so `external_sandhi.py` is rule-based generation "validated by round-trip
+ textbook unit tests" — and that the whole representation, released model included, currently rests
on it UNVERIFIED. DCS is the one place that can settle it: it carries real editorial sandhied text
in `# text` aligned with per-token `Unsandhied=`, under CC BY 4.0. The doc lists this check as
"noted, not done".

WHAT IS COMPARED. For each DCS sentence: take the `Unsandhied` forms in order, mark bound junctions
from `Compound=Yes` and the MWT ranges, run `apply_vedic_sandhi.generate`, strip CSL notation back
to the plain surface, and compare with the editorial `# text`.

⚠ CSL IS NOT THE SURFACE. CSL splits a coalesced vowel across the junction to stay reversible
(`rāj" ôvāca` for `rājovāca`), so the generated string must be de-CSLised before comparison or every
coalescence counts as a mismatch. `external_sandhi.COALESCE_SURFACE` is exactly that mapping, and it
is derived from the engine's own `_coalesce`, so it cannot drift from it.

    validate_sandhi_dcs.py corpus_sa_unsandhi/dcs_train.unsandhi.conllu [--limit N]
"""
import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import external_sandhi as ES          # noqa: E402
from conllu_misc import misc_dict     # noqa: E402
from apply_vedic_sandhi import generate  # noqa: E402


def csl_to_surface(pieces):
    """CSL pieces -> the plain sandhied CHARACTER SEQUENCE, whitespace removed.

    ⚠ COMPARE WITHOUT SPACES. DCS's editorial text runs words together across sandhi junctions
    (`sarvāstvanavadyāṅgyaḥ`) where CSL keeps them apart (`sarvās tv anavady' āṅgāḥ`), and it writes
    compounds solid (`doṣadarśī`) where CSL marks the join. Those are typographic conventions, not
    sandhi: comparing them as-is scored 14.7 % and measured the wrong thing entirely. What the
    engine is responsible for is WHICH CHARACTERS appear at each junction, so both sides are
    stripped of whitespace before comparison.

    The elision marker `'` is CSL notation for a dropped vowel, not a character an editor prints —
    except that DCS DOES print an avagraha (also `'`) for the same phenomenon, inconsistently. It is
    therefore dropped from both sides rather than from one.
    """
    s = " ".join(pieces)
    for mark in ES.COALESCE_MARKS:                 # longest first
        s = s.replace(mark, ES.COALESCE_SURFACE[mark])
    return _bare(s)


def _bare(s):
    """Strip everything that is TYPOGRAPHY rather than sandhi, from both sides.

    ⚠ Three conventions had to be normalised away before this measured the engine at all, and each
    one was first mistaken for an engine error:
      * CSL's coalescence markers `'` and `"` (elision / the right word's mark) are notation, not
        characters an editor prints — `path"ānena` vs `pathānena` was 8 of the top failures.
      * compound joins are written solid by DCS and marked in CSL.
      * final -m before a consonant: the engine writes anusvāra (`nītaṃ kālindīm`), DCS's `# text`
        writes plain `m`. Standard editions do the former; DCS does not. Both are collapsed to `m`.
    """
    s = re.sub(r"[\s'\u2019\"-]+", "", s)
    return s.replace("\u1e43", "m")          # ṃ -> m, an orthographic choice not a sandhi one


def read(path, limit=0):
    sent, text = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# text = "):
                text = line[9:].strip()
            elif not line.strip():
                if sent and text:
                    yield text, sent
                    if limit and limit <= 0:
                        return
                sent, text = [], None
            elif line and not line.startswith("#"):
                p = line.split("\t")
                if "-" in p[0] or "." in p[0]:
                    continue
                misc = misc_dict(p[9])
                sent.append((misc.get("Unsandhied") or p[1], p[5]))
    if sent and text:
        yield text, sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    n = exact = 0
    shown = 0
    diffs = collections.Counter()
    for text, toks in read(a.conllu):
        n += 1
        if n > a.limit:
            break
        words = [w for w, _ in toks]
        feats = [f for _, f in toks]
        internal = [("Compound=Yes" in f) for _, f in toks]
        try:
            pieces = generate(words, feats, internal)
        except Exception:
            diffs["engine raised"] += 1
            continue
        got = csl_to_surface(pieces)
        want = _bare(text)
        if got == want:
            exact += 1
        else:
            diffs["mismatch"] += 1
            if shown < a.show:
                shown += 1
                print(f"  want: {want}")
                print(f"  got : {got}\n")
    print(f"{n-1 if n > a.limit else n} DCS sentences")
    print(f"  exact match against the editorial surface: {exact} ({exact/max(n-1,1):.1%})")
    for k, v in diffs.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
