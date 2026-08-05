#!/usr/bin/env python3
"""``la_macronise`` -- restore vowel-length macrons to parsed Latin.

This is the "rule + lexicon" half of the Alatius macroniser, re-hosted on top of THIS pipeline's
own morphology. Alatius works by tagging with RFTagger and then looking each (form, tag) up in a
Morpheus-derived lexicon of macronised forms. Our released Latin model already predicts UPOS,
full FEATS and a lemma, so the tagging half is already done -- and done by a tagger trained on the
same treebank the lexicon is keyed to, rather than by a separate one that sometimes disagrees with
it. What remains is the lookup, which is what this component is.

Purely additive: spaCy tokens are immutable, so nothing here touches ``token.text``. The macronised
form is exposed as extensions, and the original ``doc.text`` is untouched:

    token._.macron   -- the macronised word form (str; == token.text when nothing was added)
    token._.macron_level -- which backoff level fired ("L1"/"L2"/"MP"/"L3"/"M"/"S4"/"S3"/None)
    doc._.macron     -- the macronised text, rebuilt with the doc's own whitespace

SHIPPED IN THE RELEASED WHEEL, WITH NO DATA IN IT. The component is part of the Latin model's
pipeline; the vowel lengths are not, and cannot be -- they come from Morpheus (CC BY-SA 3.0 US) and
the model is CC BY-NC-SA, so bundling them would impose exactly the restriction BY-SA forbids (see
NOTICE.md). So the pipe ships bare: until data arrives it passes every token through unchanged and
warns once, and the moment ``fetch_morpheus()`` (or a locally harvested table) lands it macronises
with no further configuration. ``require_data=True`` in the component config restores the old hard
failure for a caller who added the pipe on purpose and wants to be told.

TWO TABLES, CASCADED. The harvested one answers for the words it has; **Morpheus** answers for the
rest, fetched at runtime (see ``fetch_morpheus``):

    L1  (form, upos, feats)  the morphologiser disambiguating genuine homographs   } harvested
    L2  (form, upos)                                                               } from the
    MP  Morpheus, POS-SPLIT forms only -- jumped ahead of L3, see below            } treebank
    L3  (form)               a bare word list                                      }
    M   Morpheus, on a nine-slot morphology key with a backoff ladder              } fetched
    S4  (form[-4:], upos, feats)   ending-only; LEGACY, used only when M is absent
    S3  (form[-3:], upos, feats)
    --  otherwise the form is left bare (no macrons invented)

WHY `MP` INTERRUPTS THE CASCADE. L3 answers 90 % of tokens and is keyed on the STRING ALONE, so on a
word whose vowel length depends on its part of speech it returns the corpus majority and Morpheus --
which knows the difference -- is never reached: `malus` ADJ and `mālus` NOUN, `liber` "book" and
`līber` "free", were one question. `build_morpheus_table` marks the 4 094 forms where part of
speech alone settles the length and settles it DIFFERENTLY per part of speech, and for those, and
only those, the UPOS-aware answer goes first. Everywhere else the measured order above stands
untouched.

Three things keep it from doing harm, each of them found by a token it got wrong:
  * only a DECISIVE rung answers (`rung_mask`), never `mask`'s form-wide majority fallback -- which
    was giving vocative `canis` the `cānīs` of `cānus`, displacing a correct answer with a worse
    kind of majority than the one it replaced.
  * not inside an IDIOM. SUD gives an idiom's head the idiom's part of speech and says so in
    `ExtPos`, so `satis` in `satis facit` is tagged VERB while the word is still the adverb
    "enough"; reading that as the word's own POS made it `satīs`.
  * the two POS-bearing rungs sit at the END of `_RUNGS`, so for a token whose FEATS are full a more
    specific rung has already answered and nothing changes.

MEASURED. Agreement with Alatius barely moves (gold morphology 97.60 -> 97.59 whole-token; predicted
97.39 -> 97.34) and that is expected rather than disappointing: Alatius is RFTagger-predicted on
exactly these hard words, so it is a poor referee here -- of 24 held-out tokens where the new answer
differs from it, ~18 are ours right and its tagger wrong (`mēnse` not `mēnsē`, the third-declension
ablative being short; `ūtī` from `ūtor`, not the conjunction `utī`; `capī`, `ācer`, `audīte`). The
number that decides it takes Morpheus's GOLD-POS answer as the referee, on the 1 516 POS-split test
tokens that have one:

    old (L3 corpus majority)          86.02 %
    new (predicted UPOS + FEATS)      92.74 %

and that is with the tagger at its WEAKEST -- its UPOS accuracy on POS-split tokens is 87.92 %
against 92.35 % overall, since these are precisely the words that are hard to tag.

WHY THE CASCADE, MEASURED. Agreement with Alatius, gold morphology, held-out ITTB+PROIEL test
(48 792 tokens), split by whether the harvested table has the word at all:

                            harvest has it (92.1 %)      it does not (7.9 %)
    harvested L1/L2/L3           98.23 %                       --
    harvested S4/S3                 --                       52.46 %
    Morpheus                     93.98 %                     90.42 %

The harvest is near-perfect on its own vocabulary -- it was built from that treebank with exact
keys -- and the suffix levels that cover everything else are barely better than a coin toss. That is
the same finding this docstring used to state as prose ("the residue is overwhelmingly STEM length
on words the table has never seen ... covering them for arbitrary vocabulary needs Morpheus itself,
not a treebank-harvested table"), now acted on rather than noted: the ENDING is a function of the
paradigm and generalises, the STEM is lexical and does not, and only a real lexicon supplies it.

Taking each where it is strong: **97.61 %** whole-token in-domain, against the 94.32 % this table
scored alone. The gap widens with distance from the harvest's own corpus -- on the Perseus test
split (classical poetry, where the harvest's out-of-vocabulary share rises from 7.9 % to 23.8 %):

    Morpheus alone 95.75 %      harvested alone 87.02 %      cascaded 97.24 %

Morpheus alone is the ordinary case, and it is the better half: the harvested table cannot be
redistributed by anyone (see NOTICE.md), while Morpheus is one fetch away.

Two caveats worth stating plainly:
  * these numbers are AGREEMENT WITH ALATIUS, not gold vowel length. Alatius is ~98-99 % on vowels,
    so the ceiling here is its accuracy, not ours.
  * with PREDICTED rather than gold morphology, L1 fires on the morphologiser's output
    (``morph_acc`` ~0.83 on la dev), so real-world accuracy is below the figures above.
"""
import gzip
import json
import os
import unicodedata
import warnings
from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc, Token

LONG = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "y": "ȳ"}
LONG.update({k.upper(): v.upper() for k, v in list(LONG.items())})

for _ext, _target in (("macron", Token), ("macron_level", Token), ("macron", Doc)):
    if not _target.has_extension(_ext):
        _target.set_extension(_ext, default=None)


def strip_macron(s):
    n = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in n if c != "̄"))


# --- paradigm override -------------------------------------------------------------------------
# The lookup table memorises (form, morph) -> pattern pairs and CANNOT express a paradigm rule, so
# an unseen (form, morph) combination falls through to the form-only level, which is
# morphology-blind and can flatly contradict correctly-predicted morphology. That is how nominative
# `Gallia` came out `Galliā`: the treebank only ever attests the ablative, so the form-only
# majority is the ablative pattern, and it overrode a correct Case=Nom.
#
# These cells of the Latin paradigm fix the FINAL vowel's length absolutely, whatever the lexicon
# says. Keyed on (InflClass, Case, Number, final letter) -> is that vowel long.
#
# NB the harvested data DISAGREES with these rules on ~1500 training tokens (IndEurA/Nom/Sing/-a is
# marked long 12.9 % of the time; IndEurA/Abl/Sing/-a is marked long only 89.0 %). That is the
# Alatius macroniser's own RFTagger contradicting the treebank's gold morphology -- the rule is
# right and the data is wrong. Applying it therefore LOWERS measured agreement-with-Alatius while
# raising real accuracy; see scripts/eval_la_macronise.py --paradigm.
#
# Deliberately conservative: only cells that are exceptionless in classical Latin and that hinge on
# a final vowel. Third-declension ablative -e (short for consonant stems, -ī for i-stems) is left
# out precisely because it is NOT determined by InflClass alone.
_PARADIGM = {
    # a-stems (1st declension): nominative/vocative singular -a is short, ablative singular -ā long
    ("IndEurA", "Nom", "Sing", "a"): False,
    ("IndEurA", "Voc", "Sing", "a"): False,
    ("IndEurA", "Abl", "Sing", "a"): True,
    # o-stems (2nd declension): dative and ablative singular -ō are long
    ("IndEurO", "Dat", "Sing", "o"): True,
    ("IndEurO", "Abl", "Sing", "o"): True,
    # e-stems (5th declension): ablative singular -ē is long
    ("IndEurE", "Abl", "Sing", "e"): True,
}


def _feat(feats, key):
    """Read one feature out of a CoNLL-U FEATS string (also what str(token.morph) yields)."""
    for part in str(feats).split("|"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return ""


# When InflClass is absent -- routinely so for PROPN in ITTB/PROIEL -- the declension is still
# recoverable from the LEMMA's ending, which is how a Latinist reads it: a lemma in -a is an
# a-stem, one in -us/-um an o-stem. Restricted to nominals, and only used as a fallback.
_LEMMA_CLASS = (("a", "IndEurA"), ("us", "IndEurO"), ("um", "IndEurO"))


def _infl_class(feats, lemma, upos):
    ic = _feat(feats, "InflClass")
    if ic or upos not in ("NOUN", "PROPN", "ADJ"):
        return ic
    lem = strip_macron(str(lemma or "")).lower()
    for suf, cls in _LEMMA_CLASS:
        if lem.endswith(suf) and len(lem) > len(suf):
            return cls
    return ""


def paradigm_final(form, feats, lemma=None, upos=None):
    """Return True/False if the paradigm fixes the FINAL vowel's length, else None.

    Returns None whenever the cell is not covered -- including when InflClass is ABSENT, which is
    the common case for PROPN in ITTB/PROIEL. That is why nominative `Gallia` is still not fixed by
    this rule: it carries Case=Nom|Gender=Fem|Number=Sing and no InflClass at all, so there is
    nothing to key on. Inferring the declension from the lemma would be the next step, and is not
    attempted here.
    """
    if not form or form[-1] not in "aeiouyAEIOUY":
        return None
    return _PARADIGM.get((_infl_class(feats, lemma, upos), _feat(feats, "Case"),
                          _feat(feats, "Number"), form[-1].lower()))


# ── MORPHEUS ────────────────────────────────────────────────────────────────────────────────────
# The lexicon the harvest cannot be: 249 659 wordforms against its 42 817. Johan Winge commits
# Morpheus's output in latin-macronizer as `latin_macronizer/macrons.txt` -- 33 MB of
# `wordform TAB tag TAB lemma TAB accented`, about 4 MB on the wire -- and this fetches it on
# demand, compiles it, and caches the result.
#
# FETCHED, NOT SHIPPED, and for the same reason the harvested table is not shipped, arrived at from
# the other direction. The harvest cannot go in a wheel because it mixes CC BY-NC-SA treebank keys
# with CC BY-SA data. Morpheus has no treebank in it at all -- but it reaches us through a GPL-3.0
# repository, and GPL restricts DISTRIBUTION, not USE. A file the user's own machine fetches from
# upstream, and that never enters a build of this package, is not ours to license. Bundling either
# table would be a licensing question; fetching one is not.
MORPHEUS_URL = ("https://raw.githubusercontent.com/Alatius/latin-macronizer/"
                "master/latin_macronizer/macrons.txt")
MORPHEUS_CREDIT = ("Vowel lengths from Morpheus (Perseus Project, CC BY-SA 3.0 US) via "
                   "latin-macronizer by Johan Winge (GPL-3.0). Fetched, not redistributed.")
# 2 adds `source`/`credit` beside the payload: a table sitting in a cache directory with no
# attribution in it is exactly the file whose provenance gets forgotten, and this one has a licence
# attached to it. The F/K/S PAYLOAD IS UNCHANGED from 1 -- same 249,659 forms, same 480,384 rung
# keys, same 7,640 suffixes -- which is why `load` accepts both rather than making every existing
# cache a 4 MB re-download for two strings.
# 3 adds the two part-of-speech rungs to K and the `P` list of POS-split forms (see
# `build_morpheus_table`). A format 1 or 2 cache simply lacks them, so `_POS_SPLIT` is empty, the
# `MP` level never fires and the cascade is exactly what it was -- readable, just less informed,
# which is why an existing cache is not forced into a re-download.
MORPHEUS_FORMAT = 3
_MORPHEUS_READABLE = (1, 2, 3)


def morpheus_path():
    """Where the compiled table is cached. `$LA_MORPHEUS_TABLE` overrides."""
    env = os.environ.get("LA_MORPHEUS_TABLE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "sud-spacy" / "la_macron_morpheus.json.gz"


# The nine-slot morphology key. Slot order is the LDT tag's own -- pos, person, number, tense, mood,
# voice, gender, case, degree -- and "-" means "not stated". `macrons.txt` keys morphology as the
# Perseus/LDT nine-position tag while this pipeline emits UPOS + UD FEATS, so both are rendered into
# one alphabet and the two can meet.
_LDT_POS = {"n": "N", "v": "V", "t": "V", "a": "A", "d": "D", "p": "P",
            "m": "M", "c": "C", "r": "R", "i": "I", "e": "I", "g": "G", "u": "U"}
_UD_POS = {"NOUN": "N", "PROPN": "N", "VERB": "V", "AUX": "V", "ADJ": "A", "DET": "A",
           "ADV": "D", "PRON": "P", "NUM": "M", "CCONJ": "C", "SCONJ": "C", "ADP": "R",
           "INTJ": "I", "PART": "G", "PUNCT": "U"}
_UD_CASE = {"Nom": "n", "Gen": "g", "Dat": "d", "Acc": "a", "Abl": "b", "Voc": "v", "Loc": "l"}
_UD_NUM = {"Sing": "s", "Plur": "p"}
_UD_GEN = {"Masc": "m", "Fem": "f", "Neut": "n"}
_UD_MOOD = {"Ind": "i", "Sub": "s", "Imp": "m"}
_UD_VFORM = {"Inf": "n", "Part": "p", "Ger": "d", "Gdv": "g", "Sup": "u"}
_UD_VOICE = {"Act": "a", "Pass": "p"}
_UD_DEG = {"Cmp": "c", "Sup": "s"}
# UD splits across Tense and Aspect what the LDT packs into one slot, so the pair is read together:
# a Latin "past" is the imperfect or the perfect depending on aspect, and they take different
# endings. Aspect is defaulted because ITTB states it and PROIEL often does not.
_UD_TENSE = {("Pres", ""): "p", ("Pres", "Imp"): "p", ("Past", "Imp"): "i", ("Past", ""): "r",
             ("Past", "Perf"): "r", ("Pqp", ""): "l", ("Pqp", "Perf"): "l",
             ("Fut", ""): "f", ("Fut", "Imp"): "f", ("Fut", "Perf"): "t"}
# The backoff ladder: which slots survive at each rung, most specific first. Tense goes first
# because the Tense/Aspect pairing above is the least reliable half of the mapping; part of speech
# next, because it is what the tagger most often gets wrong (`cano` came back ADJ and `fortes` VERB
# on a sample); then gender, which a Latin ending most often leaves ambiguous. A LADDER rather than
# one exact key, because with one key every mis-tag becomes a total miss instead of a coarser hit.
# The last two rungs KEEP the part of speech, and sit at the END. For a token whose FEATS are full
# a more specific rung has already answered, so they change nothing there; they exist so that the
# table can be asked the one question the ladder above cannot put to it -- "does part of speech
# alone settle this word?" -- which is what `_POS_SPLIT` below is built from.
_RUNGS = ("012345678", "01245678", "1245678", "124567", "1247", "27", "027", "0")


def _slots(*vals):
    return "".join(v or "-" for v in vals)


def _rung(key, keep):
    """`key` with every slot outside `keep` blanked -- the stored form of one ladder rung."""
    return "".join(key[i] if str(i) in keep else "-" for i in range(9))


def ldt_key(tag):
    """A Perseus/LDT nine-position tag -> the shared key. Positional; only the part of speech is
    translated, since "-" already means "unspecified" in the source."""
    t = (tag or "").ljust(9, "-")[:9]
    return _slots(_LDT_POS.get(t[0], "-" if t[0] == "-" else "?"),
                  t[1] if t[1].isdigit() else "-", t[2] if t[2] in "sp" else "-",
                  t[3] if t[3] in "pirltf" else "-", t[4] if t[4] in "isnmpdgu" else "-",
                  t[5] if t[5] in "ap" else "-", t[6] if t[6] in "mfn" else "-",
                  t[7] if t[7] in "ngdabvl" else "-", t[8] if t[8] in "cs" else "-")


def ud_key(upos, feats):
    """UPOS + UD FEATS -> the shared key.

    A MULTI-VALUED feature is read as UNSTATED, not as its first value: the morphologiser writes
    `Gender=Fem,Masc` when the form genuinely does not distinguish them, and picking one would
    assert what the tagger explicitly declined to. Blanking the slot drops the lookup to a coarser
    rung instead, which is the honest answer to an ambiguity."""
    def one(val, table):
        return table.get(val, "") if val and "," not in val else ""
    mood = one(_feat(feats, "Mood"), _UD_MOOD) or _UD_VFORM.get(_feat(feats, "VerbForm"), "")
    tense = _UD_TENSE.get((_feat(feats, "Tense"), _feat(feats, "Aspect")), "")
    person = _feat(feats, "Person")
    return _slots(_UD_POS.get(upos or "", ""), person if person in "123" else "",
                  one(_feat(feats, "Number"), _UD_NUM), tense, mood,
                  one(_feat(feats, "Voice"), _UD_VOICE), one(_feat(feats, "Gender"), _UD_GEN),
                  one(_feat(feats, "Case"), _UD_CASE), one(_feat(feats, "Degree"), _UD_DEG))


def _split_accented(acc):
    """Morpheus's `a^ba_ctor` -> ("abactor", bitmask of the LONG vowels), indexed from the left."""
    plain, mask, i = [], 0, 0
    for ch in acc:
        if ch == "_":
            if i:
                mask |= 1 << (i - 1)
        elif ch == "^":
            pass
        else:
            plain.append(ch)
            i += 1
    return "".join(plain), mask


def build_morpheus_table(lines, progress=None):
    """Compile `macrons.txt` into the lookup below. Pure; takes any line iterable."""
    forms = {}
    seen = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        wf, tag, acc = parts[0], parts[1], parts[3]
        plain, mask = _split_accented(acc)
        # Only rows whose accented column really spells the wordform. Morpheus also emits entries
        # whose accented form differs (u/v and i/j normalisation), and a mask indexed against a
        # different string would lengthen the wrong vowel.
        if not wf or plain.lower() != wf.lower():
            continue
        key = ldt_key(tag)
        if "?" in key:
            continue
        forms.setdefault(wf.lower(), {}).setdefault(key, set()).add(mask)
        seen += 1
        if progress and seen % 200000 == 0:
            progress(f"read {seen:,} analyses")

    F, K, P = {}, {}, []
    for wf, by in forms.items():
        masks = set()
        for st in by.values():
            masks |= st
        if len(masks) == 1:
            F[wf] = next(iter(masks))
            continue
        tally = {}
        for st in by.values():
            for m in st:
                tally[m] = tally.get(m, 0) + 1
        F[wf] = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        # Every rung that IS decisive for this form -- and only those, so a rung never answers a
        # question it cannot settle.
        by_pos = {}
        for keep in _RUNGS:
            agg = {}
            for k, st in by.items():
                agg.setdefault(_rung(k, keep), set()).update(st)
            for rk, st in agg.items():
                if len(st) == 1:
                    K[wf + "\t" + rk] = next(iter(st))
            if keep == "0":
                by_pos = agg
        # A POS-SPLIT form: part of speech alone settles the vowel length, and settles it
        # DIFFERENTLY for at least two parts of speech -- `malus` ADJ short against `mālus` NOUN
        # long, `liber` "book" against `līber` "free". For these and only these the form-wide
        # majority is a coin flip, which is what `_lookup` uses this list to refuse. Groups whose
        # POS slot is blank are ignored: an unknown part of speech distinguishes nothing.
        decisive = {rk: next(iter(st)) for rk, st in by_pos.items()
                    if len(st) == 1 and rk[0] != "-"}
        if len(decisive) > 1 and len(set(decisive.values())) > 1:
            P.append(wf)

    # SUFFIX levels for a word Morpheus has never seen either. Kept near-unanimous only: a coin toss
    # here would invent macrons, which is worse than leaving the form bare.
    S = {}
    for k in (4, 3):
        agg2 = {}
        for wf, m in F.items():
            if len(wf) <= k:
                continue
            n = len(wf)
            sm = sum(1 << j for j in range(k) if (m >> (n - k + j)) & 1)
            d = agg2.setdefault(f"{k}\t{wf[-k:]}", {})
            d[sm] = d.get(sm, 0) + 1
        for sk, t in agg2.items():
            tot = sum(t.values())
            best, n = max(t.items(), key=lambda kv: (kv[1], -kv[0]))
            if n / tot >= 0.9:
                S[sk] = best
    if progress:
        progress(f"compiled {len(F):,} forms, {len(P):,} of them POS-split")
    return {"format": MORPHEUS_FORMAT, "source": "morpheus", "credit": MORPHEUS_CREDIT,
            "F": F, "K": K, "S": S, "P": sorted(P)}


def fetch_morpheus(dest=None, progress=print):
    """Download `macrons.txt` and compile it to `dest` (default `morpheus_path()`).

    Run once per machine. Nothing in this package ships the result, and nothing should."""
    import urllib.request
    dest = Path(dest or morpheus_path())
    dest.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"downloading {MORPHEUS_URL}")
    req = urllib.request.Request(MORPHEUS_URL, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
        if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
            raw = gzip.decompress(raw)
    table = build_morpheus_table(raw.decode("utf-8", "replace").splitlines(), progress=progress)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(table, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, dest)      # atomic: a half-written table must never be loadable
    if progress:
        progress(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


class Morpheus:
    """The compiled table, and the ladder walk over it."""

    def __init__(self, table):
        self.F = table.get("F") or {}
        self.K = table.get("K") or {}
        self.S = table.get("S") or {}
        #: forms whose vowel length part of speech alone settles, differently per part of speech
        self.P = frozenset(table.get("P") or ())

    @classmethod
    def load(cls, path=None):
        """The cached table, or None when there is none (or it was built by an older shape)."""
        p = Path(path or morpheus_path())
        if not p.is_file():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                t = json.load(fh)
            if int(t.get("format") or 0) not in _MORPHEUS_READABLE or not t.get("F"):
                return None
            return cls(t)
        except Exception:
            return None

    def rung_mask(self, form, upos, feats):
        """The mask from a DECISIVE rung only -- no form-wide majority, no suffix guess.

        `mask` below ends by returning `self.F[form]`, the majority across every reading, which is
        the right last resort when the alternative is nothing. It is the wrong one when the
        alternative is the harvested L3 level, whose majority at least comes from the treebank being
        macronised. Vocative `canis` has no rung at all in Morpheus and was taking `cānīs` (the
        dative-ablative plural of `cānus`) from that fallback, displacing a correct `canis`.
        """
        key = ud_key(upos, feats)
        for keep in _RUNGS:
            m = self.K.get(form + "\t" + _rung(key, keep))
            if m is not None:
                return m
        return None

    def mask(self, form, upos, feats):
        """The long-vowel mask for `form`, or None when nothing answers."""
        lower = form.lower()
        if lower in self.F:
            key = ud_key(upos, feats)
            for keep in _RUNGS:
                m = self.K.get(lower + "\t" + _rung(key, keep))
                if m is not None:
                    return m
            return self.F[lower]
        for k in (4, 3):
            if len(lower) > k:
                m = self.S.get(f"{k}\t{lower[-k:]}")
                if m is not None:
                    n = len(lower)
                    return sum(1 << (n - k + j) for j in range(k) if (m >> j) & 1)
        return None


def apply_mask(form, mask):
    """Lengthen the vowels whose bit is set, preserving the form's own case."""
    out = []
    for i, ch in enumerate(form):
        out.append(LONG.get(ch, ch) if (mask >> i) & 1 else ch)
    return "".join(out)


@Language.factory("la_macronise",
                  default_config={"lut": None, "paradigm": True, "morpheus": True,
                                  "require_data": False})
def make_la_macronise(nlp, name, lut, paradigm, morpheus, require_data):
    return LaMacronise(lut, paradigm, morpheus, require_data)


class LaMacronise:
    def __init__(self, lut=None, paradigm=True, morpheus=True, require_data=False):
        self.paradigm = paradigm
        # `require_data`: what to do when NO vowel-length data is present. The released wheel ships
        # the component with no table (see NOTICE.md -- Morpheus is CC BY-SA, the model is
        # CC BY-NC-SA), so for it the answer has to be "pass the doc through unchanged and say so
        # once", not "raise": a component in the default pipeline that throws would make every
        # ordinary `nlp(text)` fail for the 99 % of users who never asked for macrons. That is the
        # same degrade-don't-fail posture the rest of this project takes toward optional data.
        # True restores the old hard failure, for a caller that added the pipe ON PURPOSE and would
        # rather hear about a missing table than silently get its input back.
        self.require_data = require_data
        self._warned = False
        self.l1 = self.l2 = self.l3 = self.s4 = self.s3 = {}
        # `lut` is a BUILD-time convenience only. In a packaged model the table travels inside the
        # model directory and is restored by from_disk(), which runs after __init__ -- so a config
        # that still names a build-time path (or none at all) must not be fatal here, or the wheel
        # fails to load with FileNotFoundError before from_disk ever gets a chance.
        if lut and Path(lut).exists():
            self._load_blob(json.loads(gzip.open(lut, "rb").read().decode("utf-8")))
        # `morpheus`: True loads the cached table if there is one (and is silent when there is not
        # -- `fetch_morpheus` is an explicit act, never a surprise download inside `spacy.load`),
        # False disables it, a path names one.
        self.morpheus = None
        if morpheus:
            self.morpheus = Morpheus.load(None if morpheus is True else morpheus)

    def _load_blob(self, b):
        self.l1 = {(f, u, x): m for f, u, x, m in b["L1"]}
        self.l2 = {(f, u): m for f, u, m in b["L2"]}
        self.l3 = {f: m for f, m in b["L3"]}
        # S4/S3 are LEGACY: `build_la_macron_lut.py` no longer emits them, because Morpheus answers
        # the same question far better (52.46 % against 90.42 % on the tokens they exist for). Still
        # READ, so a table built before that change keeps working -- and still used, but only where
        # there is no Morpheus table at all.
        self.s4 = {(f, u, x): m for f, u, x, m in b.get("S4", [])}
        self.s3 = {(f, u, x): m for f, u, x, m in b.get("S3", [])}

    def _lookup(self, form, upos, feats):
        """Return (mask, level) for the lowercased form, or (None, None).

        THE CASCADE, in the order the docstring's measurements justify: the harvested table's exact
        levels first (98.23 % where they fire), then Morpheus (90.42 % on everything they miss),
        then -- only for a legacy table with no Morpheus beside it -- the suffix guess."""
        n = len(form)
        m = self.l1.get((form, upos, feats))
        if m is not None:
            return m, "L1"
        m = self.l2.get((form, upos))
        if m is not None:
            return m, "L2"
        # L3 answers 90 % of tokens and is keyed on the STRING ALONE, so on a word whose length
        # depends on its part of speech it returns the corpus majority and Morpheus -- which knows
        # the difference -- is never reached. That is a coin flip on `malus` ADJ vs `mālus` NOUN,
        # `liber` "book" vs `līber` "free". `P` is exactly the set of forms where part of speech
        # settles the question and settles it differently, so for those, and only those, the
        # UPOS-aware answer goes first. Everywhere else the measured order stands.
        # ... but NOT inside an idiom. SUD gives an idiom's head the IDIOM's part of speech and
        # records the fact in `ExtPos`, so in `satis facit` the token `satis` is tagged VERB while
        # the word is still the adverb "enough" -- reading that UPOS as the word's own turns it into
        # `satīs`, the dative-ablative plural of a participle. The treebank says so on the token, so
        # the guard is exact rather than a heuristic.
        if (self.morpheus is not None and form in self.morpheus.P
                and not _feat(feats, "ExtPos")):
            m = self.morpheus.rung_mask(form, upos, feats)
            if m is not None:
                return m, "MP"
        m = self.l3.get(form)
        if m is not None:
            return m, "L3"
        if self.morpheus is not None:
            m = self.morpheus.mask(form, upos, feats)
            if m is not None:
                return m, "M"
        for k, tab, lvl in ((4, self.s4, "S4"), (3, self.s3, "S3")):
            m = tab.get((form[-k:], upos, feats))
            if m is not None:
                # the stored mask is indexed from the right; shift it back onto the form
                return sum(1 << (i + n - k) for i in range(k)
                           if (m >> i) & 1 and 0 <= i + n - k < n), lvl
        return None, None

    def resolve(self, form, upos, feats, lemma=None):
        """(macronised form, level) for one token -- the single path used by __call__ AND the
        evaluator, so a measurement can never silently miss the paradigm override."""
        if not any(c.isalpha() for c in form):
            return form, None
        mask, level = self._lookup(strip_macron(form).lower(), upos, feats)
        mask = mask or 0
        if self.paradigm:
            fixed = paradigm_final(form, feats, lemma, upos)
            if fixed is not None:
                bit = 1 << (len(form) - 1)
                new = (mask | bit) if fixed else (mask & ~bit)
                if new != mask:
                    level = f"{level or 'none'}+P"
                mask = new
        base = strip_macron(form)
        return (apply_mask(base, mask) if mask else base), level

    #: Said once, whether the no-data case raises or merely warns -- the two differ in severity, not
    #: in what the user has to do about it, so there is one text.
    #: The fetch line is written with __name__ rather than a literal "la_macronise", because inside
    #: a packaged wheel this module is `la_sud_ittb_proiel_perseus.la_macronise` and an instruction
    #: naming the bare name would fail with ImportError for exactly the reader who most needs it.
    NO_DATA = (
        "la_macronise has no vowel-length data, and none is distributed with this model. "
        "The lengths come from Morpheus (CC BY-SA 3.0 US) and this model is CC BY-NC-SA, so "
        "bundling them would add exactly the restriction BY-SA forbids. Two ways to supply "
        "them, either of which is enough:\n"
        f"  python -c 'from {__name__} import fetch_morpheus; fetch_morpheus()'\n"
        "      one 4 MB download, covers 249,659 wordforms, needs nothing else; or\n"
        "  bash scripts/build_la_macron.sh   (in the SUD-spaCy repo)\n"
        "      harvests a table from your own treebank -- more accurate on ITS vocabulary "
        "and much less accurate off it, so it is worth having IN ADDITION rather than "
        "instead (see this module's docstring for the numbers, and NOTICE.md)."
    )

    def has_data(self):
        """Is there any vowel-length data to work from? Public, because a caller that offers
        macronisation in its own UI needs to know whether to offer it at all."""
        return bool(self.l3) or self.morpheus is not None

    def __call__(self, doc):
        if not self.has_data():
            if self.require_data:
                raise RuntimeError(self.NO_DATA)
            # No data: pass every token through unchanged, so `._.macron` is always readable and
            # always a string. Warned ONCE per component -- a per-doc warning would bury a batch
            # run's real output, and the condition cannot change between two calls to one instance.
            if not self._warned:
                self._warned = True
                warnings.warn(self.NO_DATA + "\n  (macronisation is off; token._.macron is the "
                              "unchanged form)", RuntimeWarning, stacklevel=2)
            for tok in doc:
                tok._.macron = tok.text
                tok._.macron_level = None
            doc._.macron = doc.text
            return doc
        pieces = []
        for tok in doc:
            out, level = self.resolve(tok.text, tok.pos_, str(tok.morph) or "_",
                                       tok.lemma_)
            tok._.macron = out
            tok._.macron_level = level
            pieces.append(out + tok.whitespace_)
        doc._.macron = "".join(pieces)
        return doc

    # -- serialisation: the table travels inside the model directory ------------------
    def to_disk(self, path, exclude=tuple()):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {
            "L1": [[f, u, x, m] for (f, u, x), m in self.l1.items()],
            "L2": [[f, u, m] for (f, u), m in self.l2.items()],
            "L3": [[f, m] for f, m in self.l3.items()],
            "S4": [[f, u, x, m] for (f, u, x), m in self.s4.items()],
            "S3": [[f, u, x, m] for (f, u, x), m in self.s3.items()],
        }
        with gzip.open(path / "lut.json.gz", "wb", compresslevel=9) as fh:
            fh.write(json.dumps(blob, ensure_ascii=False).encode("utf-8"))

    def from_disk(self, path, exclude=tuple()):
        p = Path(path) / "lut.json.gz"
        if p.exists():
            self._load_blob(json.loads(gzip.open(p, "rb").read().decode("utf-8")))
        return self

    def to_bytes(self, exclude=tuple()):
        blob = {
            "L1": [[f, u, x, m] for (f, u, x), m in self.l1.items()],
            "L2": [[f, u, m] for (f, u), m in self.l2.items()],
            "L3": [[f, m] for f, m in self.l3.items()],
            "S4": [[f, u, x, m] for (f, u, x), m in self.s4.items()],
            "S3": [[f, u, x, m] for (f, u, x), m in self.s3.items()],
        }
        return gzip.compress(json.dumps(blob, ensure_ascii=False).encode("utf-8"), 9)

    def from_bytes(self, data, exclude=tuple()):
        self._load_blob(json.loads(gzip.decompress(data).decode("utf-8")))
        return self
