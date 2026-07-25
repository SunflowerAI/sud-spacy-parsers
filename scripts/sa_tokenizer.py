#!/usr/bin/env python3
"""Input front-end for the Sanskrit model (`sa_sud_vedic_ufal_csl`).

The model is trained on **CSL-reverted** wordforms: sandhied Clay-Sanskrit-Library text with the
*notation-marked* sandhi undone (vowel coalescence and avagraha) but the unmarked consonant/visarga
sandhi (visarga → -o/-r, m → ṃ, t/n assimilation) left on the surface. This tokeniser maps raw CSL
input to that representation, so the parser sees clean, normalised wordforms.

It normalises input to the treebank's unaccented IAST before whitespace tokenisation:
  * Devanagari (देवनागरी) -> IAST via indic-transliteration (ळ ḻ→ḷ, anusvara → ṃ);
  * accented Vedic IAST -> the udātta/svarita accent marks are stripped (phonemic macrons and
    under/over-dots are kept);
  * plain unaccented IAST passes through unchanged;
then **reverses the CSL-marked sandhi** with `desandhi_csl` (see below).

Input must be **word-segmented** (space-separated padas) — like the treebank; continuous
saṃhitā/Devanagari needs sandhi splitting, which is out of scope. Runtime dependency:
`pip install indic-transliteration` (only needed when Devanagari is fed; pure-Python, MIT).
"""
import re
import unicodedata

from spacy.tokens import Doc
from spacy.util import registry

# Punctuation the treebanks tokenise as separate tokens (Devanagari daṇḍa ।॥, which the
# transliterator renders as |/||, plus the Latin marks the UFAL edition uses). A maximal
# run of the SAME punctuation char is ONE token, so the double daṇḍa ॥ -> "||" stays whole;
# `desandhi_csl` then normalises every DOUBLE daṇḍa (||, //, ॥, ।।) to the single char ‖ (U+2016)
# — see `_normalise_danda`. NB: hyphen-minus '-' is deliberately absent — it is the CSL/MWT-internal
# boundary marker handled by _HYPH below (a lone '-' becomes its own token there); the
# avagraha "'" and the CSL long-elision mark '"' (U+0022) stay attached to their word.
_PUNCT = "।॥|/.?!,;:–—«»‹›”“‘’…()[]‖"
_PCLASS = re.escape(_PUNCT)
# Sentence-MEDIAL punctuation — TRANSPARENT to the non-coalescent external sandhi (see
# `_next_word` and the desandhi section below): a comma / quotation mark / bracket is a purely
# typographic overlay a modern editor lays over a phonological chain that keeps running, so
# visarga, -s/-r, anusvara, stop and glide sandhi all apply straight across it (tataś, ca <-
# tataḥ ca;  kiṃ, bhadre <- kim bhadre). The sentence-final marks . ? ! … are a genuine pause
# (avasāna), at which the words on either side already stand in their pausa form: OPAQUE.
# A SINGLE daṇḍa is medial-or-final by the same DOCUMENT-DEPENDENT test `clause_parser` uses for
# sa (`sent_scheme = "danda"`): if the text closes its sentences with a DOUBLE daṇḍa, a single one
# is only a pāda / half-verse boundary — a metrical, not a phonological, break — so sandhi runs
# across it too; where there is no double daṇḍa the single one IS the sentence end, hence a pause.
# A double daṇḍa is always a pause.
_MEDIAL_PUNCT = set(",;:–—«»‹›()[]”“‘’")
_SPLIT = re.compile(r"[^%s]+|([%s])\1*" % (_PCLASS, _PCLASS))
# A CSL/MWT-internal hyphen stays attached to the element on its LEFT (śrī-śāradā ->
# 'śrī-', 'śāradā'); a lone hyphen (the dash PUNCT, e.g. "ucyate -") is its own token.
_HYPH = re.compile(r"[^-]+-|[^-]+|-")
# CSL prints compound division with a thin vertical line; accept | as a compound-internal
# separator (śrī|śāradā) and normalise it to a hyphen. Only a | that is immediately followed
# by a word character is a compound join — a sentence daṇḍa (।/॥ -> |/||) is always followed
# by space, end, or other punctuation, never directly by a letter.
_PIPE = re.compile(r"\|(?=[^\s%s])" % _PCLASS)
# Straighten typographic (curly) apostrophes and double-apostrophes to the ASCII ' and "
# used for the sandhi marks (avagraha / vowel elision), so smart-quoted input matches the
# model. (CSL quotation uses guillemets « », which are distinct and pass through.)
_STRAIGHTEN = {0x2018: "'", 0x2019: "'", 0x201B: "'", 0x2032: "'",
               0x201C: '"', 0x201D: '"', 0x201F: '"', 0x2033: '"'}

# Vedic pitch-accent combining marks to drop (keep macron U+0304, dot-below U+0323,
# dot-above U+0307). NB the combining circumflex U+0302 is deliberately NOT dropped:
# in the CSL scheme circumflex-on-vowel is a meaningful sandhi-coalescence mark
# (â ê î ô û / âi âu), not an accent, so it must survive normalisation.
_ACCENTS = {chr(cp) for cp in (0x0301, 0x0300, 0x0951, 0x0952, 0x1CDA, 0x0331)}
# transliterator output -> treebank IAST conventions
_FIX = {"ḻ": "ḷ", "Ḻ": "Ḷ"}  # ḻ/Ḻ -> ḷ/Ḷ (Vedic ळ)


def _has_devanagari(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def normalise(text):
    text = text.translate(_STRAIGHTEN)            # curly apostrophes/double-quotes -> ASCII ' "
    if _has_devanagari(text):
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        text = "".join(_FIX.get(c, c) for c in text)
    # Strip Vedic pitch accents (NFD, drop the accent marks, recompose) — but KEEP the
    # combining acute that is part of ś/Ś (s/S + U+0301): it is a phonemic consonant, not
    # an accent. Vedic udātta/svarita only fall on vowels (and vocalic ṛ/ḷ), never on a true
    # consonant like s, so an acute sitting directly on s/S can only be ś/Ś.
    out, base = [], ""
    for c in unicodedata.normalize("NFD", text):
        if not unicodedata.combining(c):
            base = c
            out.append(c)
        elif c in _ACCENTS and not (c == "́" and base in ("s", "S")):  # U+0301 = acute
            continue                      # drop a genuine Vedic accent mark
        else:
            out.append(c)                 # keep macron / dot-below / the ś acute
    return unicodedata.normalize("NFC", "".join(out))


# --------------------------------------------------------------------------------------------
# Reverse the CSL sandhi, normalising each word toward its **pre-pausal (pausa) form** — the shape
# it takes in isolation — so the parser sees ONE canonical wordform regardless of the following
# context (and training data + runtime tokeniser stay in step, since `revert_csl_sandhi.py` calls
# this same routine). Operates on the ordered token list (every junction is a two-token affair).
# Undone:
#   (1) the *notation-marked* junctions — vowel coalescence (' / " on the left, â ê î ô û / macron
#       on the right) and avagraha — reversible exactly (`_restore_pair` etc.);
#   (2) the DETERMINISTIC external sandhi — recoverable WITHOUT a lexicon, because CSL keeps word
#       boundaries and marks coalescence, and because pausa reductions that PRESERVE place of
#       articulation are unique (`_rev_visarga_vowel`, `_rev_final_consonant`). Applied:
#         • ayādi glide before a vowel  ->  the diphthong/mid vowel: -ay/-av/-āy/-āv -> -e/-o/-ai/-au
#           (te i- -> tay i-; tau a- -> tāv a-). Unambiguous because external_sandhi KEEPS the glide
#           (e+V -> ay V, not bare hiatus): a vowel-then-glide is ayādi, a CONSONANT-then-glide is
#           yaṇ, and a bare vowel is visarga;
#         • yaṇ glide before a vowel  ->  the short vowel: -Cy/-Cv -> -Ci/-Cu (ity a- -> iti a-,
#           tanv a- -> tanu a-). The i/ī, u/ū LENGTH is not recoverable from y/v, so the short vowel
#           is taken as the default (iti, not itī) — a deliberate, small loss for the common case;
#         • -a/-ā in HIATUS before a vowel  ->  -aḥ/-āḥ  (a genuine -a/-ā + V would have coalesced and
#           been marked, and ayādi keeps its glide, so a bare -a/-ā + V can only be a dropped visarga);
#         • the guṇa of a following vocalic ṛ/ḷ (-a/-ā + ṛ -> -ar, `_rev_guna_r`): a word2-initial r/l
#           before a CONSONANT  ->  ṛ/ḷ  (ca rṣiḥ -> ca ṛṣiḥ, etayā rcā -> etayā ṛcā). Word1 keeps its
#           own vowel in this junction, so — unlike coalescence — nothing is marked; the cue is on
#           word2, and it is unambiguous because no native Sanskrit word can begin r/l + consonant;
#         • -o before avagraha  ->  -aḥ  (namo 'stu -> namaḥ astu), and -o before a voiced consonant
#           ->  -aḥ  (vatso vira- -> vatsaḥ);  [both from -aḥ; the only collision is the -u-stem
#           vocative in -o (viṣṇo, ~0.8 %), accepted — the model being for classical prose];
#         • word-final -s and -r (after a vowel)  ->  visarga -ḥ  (tatas -> tataḥ, punar -> punaḥ,
#           agnir -> agniḥ);  every Sanskrit word-final s/r goes to visarga at pause;
#         • voiced stop -d/-g/-b  ->  -t/-k/-p  (dental/velar/labial voice neutralised: tad -> tat,
#           id -> it);  place is preserved, so this is unique;
#         • anusvara -ṃ before a non-sibilant consonant  ->  -m;  gemination -nn  ->  -n.
#         • CONTEXT-SENSITIVE sibilant/palatal junctions, gated by a small gold-derived lexicon of
#           genuine consonant-final / ch-initial stems (`_rev_sibilant_and_c`, lexica _SIB_FINAL /
#           _C_FINAL / _J_FINAL / _L_FINAL / _CH_INITIAL): word-final -ś before c/ch and -ṣ before ṭ/ṭh
#           are visarga (kratuś ca -> kratuḥ ca) unless a genuine -ś/-ṣ stem (diś, haviṣ); word-final
#           -c/-j/-l before their trigger are -t (tac ca -> tat ca, taj jal- -> tat jal-, tal l- ->
#           tat l-) unless a genuine stem (vāc, rāj, the -añc directionals); and the ch of a -c ch-
#           junction is ś (paṭhec chiva -> paṭhet śiva) unless a genuine ch-word (chāyā). -ñc is
#           structurally always genuine (t+c never yields -ñc).
#         • LAW OF FINALS / avasāna (stage 3, `_rev_law_of_finals` + `_LAW_OF_FINALS`): a genuine
#           consonant-final stem is normalised to its pausa form — vāc -> vāk, ṛc -> ṛk, pratyañc ->
#           pratyaṅ, diś -> dik, viś -> viṭ, rāj -> rāṭ, yuj -> yuk, haviṣ -> haviḥ, ṣaṣ -> ṣaṭ. Place
#           is LEXICAL (diś->k vs viś->ṭ; rāj->ṭ vs yuj->k), so it is a per-stem map (Whitney §141-2 +
#           gold). Applied to compound members too (prāc- -> prāk-) — one pre-pausal form per stem.
#   NOT reverted — genuinely ambiguous even WITH the lexicon: the remaining aspirate finals -h, the
#   hapax -ṣ stem dadhṛṣ (uncertain place), and a word-final segment before a non-triggering one (place
#   unrecoverable there: diś -> dik but viś -> viṭ), and a -ā/-a before a voiced consonant (a dropped
#   -āḥ/-aḥ is indistinguishable from a genuine final vowel there). These stay on the surface. Measured
#   on Vedic (round-trip through external_sandhi against gold): the bare-hiatus visarga rule is 100 %
#   clean (5922/5922 genuine visarga — ayādi no longer collides with it), the ayādi glide reversal
#   round-trips exactly, and the lexicon-gated sibilant/-c junctions are 100 % on the gold (word2
#   1668/1668; the -ś/-ṣ and -c guards 1230+438 with zero genuine-stem mangling). NB the paired forward
#   engine `external_sandhi.py` must keep the ayādi glide (e+V -> ay, not bare hiatus) for this to hold.
#
#   PUNCTUATION. Every junction rule above except the two COALESCENT ones (the marked vowel coalescence
#   of stage 1 and the guṇa ṛ/ḷ of `_rev_guna_r`) looks for its neighbour through sentence-medial
#   punctuation (`_MEDIAL_PUNCT`, `_next_word`): non-coalescent external sandhi applies right across an
#   editorial comma or quotation mark, which a CSL edition lays over a phonological chain that does not
#   pause there (tataś, ca <- tataḥ ca;  vatso, vipra- <- vatsaḥ vipra-;  kiṃ, bhadre <- kim bhadre).
#   The coalescent two are kept strictly adjacent, since coalescence fuses the two vowels into a single
#   syllable and no mark can sit inside it. A sentence-final mark or a daṇḍa is a PAUSE: it blocks every
#   rule, because the words flanking it already stand in pausa form — which also settles the anusvara: a
#   -ṃ at a pause is a GENUINE anusvara (oṃ) and is left alone, exactly as at end of input and before a
#   vowel, since an edition writes final m as -m before a pause (Devanagari virāma). NB this is a fix
#   in passing: before the pause/medial split a daṇḍa counted as an ordinary following consonant, so
#   `oṃ ‖` was reduced to `om` while a sentence-final `oṃ` was not.
_APOS, _DAPOS = "'", '"'
# a genuinely vowel-initial word starts with one of these plain vowels (the coalescence marks
# â ê î ô û / ē ō are what a *coalesced* right word begins with instead, and are excluded);
_PLAIN_VOWEL = set("aāiīuūṛṝḷeo")
# voiced consonants that trigger visarga -> -o/-r (external_sandhi.VOICED_C).
_VOICED_C = set("gjḍdbṅñṇnmyrlvh")
# Lexica of GENUINE consonant-final / ch-initial wordforms, harvested from the Vedic gold (pausa)
# treebank (sa_vedic-sud-{train,dev,test}.conllu, minus ś-words mis-stored with ch-). They gate the
# context-sensitive consonant reversions in `_rev_sibilant_and_c` so a real stem is never mangled:
#   • a surface word-final -ś (before c/ch) / -ṣ (before ṭ/ṭh) is visarga (kratuś ca <- kratuḥ ca)
#     UNLESS the word is a genuine -ś/-ṣ stem (diś, viś, haviṣ) that keeps its sibilant;
#   • a surface word-final -c (before c/ch) is t+c/t+ch/t+ś sandhi (-c <- -t: tac ca <- tat ca)
#     UNLESS the word is a genuine -c stem (vāc, ṛc, tvac, the -añc directionals — note -ñc can
#     NEVER arise from t+c sandhi, so it is treated as genuine structurally, without the list);
#   • the word2 of a -c ch- junction is t+ś (ch <- ś: paṭhec chiva <- paṭhet śiva) UNLESS it is a
#     genuine ch-initial word (chāyā, chandas, chid-, chāga …), a small closed class. Validated 100 %
#     against the forward engine on the gold (word2 1668/1668; the -c/-ś guards 438+1230/…).
_CH_INITIAL = frozenset("""
    chadayat chadayathaḥ chadayati chadiḥ chadma chadmabhiḥ chaitsīt chambaṭkurvanti chambaṭkāram
    chandasaḥ chandasi chandaskṛtam chandaskṛtaḥ chandasā chandasām chandasī chandati chandayase
    chandayāte chandaḥ chandaḥsu chandobhiḥ chandobhyaḥ chandogam chandogebhyaḥ chandogāḥ chandomāḥ
    chandonāmānām chandovicitiḥ chandāḥ chandāṃsi channaḥ channām chantsat chardayate chardayitvā
    chardiḥ chatra chatram chattram chattreṇa chattrāṇi chavyai chedi chetsyāmi chidra chidram
    chidraḥ chidre chidreṇa chidreṣu chidrāṇi chidyamānā chidyante chidyate chinadmi chinatti
    chinattu chindan chindanti chinddhi chinna chinnam chinnasya chinnaḥ chinne chinnāt chinnāḥ
    chinttam chitsi chittvā chubukena chuchundarī chutudrī chuvukena chyati chādayan chādayati
    chādayāmi chādyate chāga chāgasya chāyā chāyām chāyānām chāyāyām chāyāḥ chṛndantu chṛndhi
    chṛṇattu
""".split())
_C_FINAL = frozenset("""
    avāc avāñc ghṛtāñc nimruc parāc parāñc pratyañc prāc prāñc ruc sic sruc taijanitvac tvac udañc
    upapṛc vāc śuc ṛc
""".split())
_SIB_FINAL = frozenset("""
    dadhṛṣ dhīṣ divispṛś diś etādṛś haviṣ hṛdispṛś jyotiṣ niṣ saṃdṛś spaś tādṛś upadṛś vipāś viś
    yādṛś ṣaṣ
""".split())
_J_FINAL = frozenset("""
    abhoj asṛj bhiṣaj bhrāj bhāj dharmarāj nirṇij rej ruj rāj samrāj saṃvṛj sraj svarāj svāvṛj
    vanerāj vaṇij vibhrāj virāj yuj ūrj ṛtvij
""".split())
_L_FINAL = frozenset("""
    bāl
""".split())
# Law of finals (avasāna): a genuine consonant-final stem's PAUSA form — what it becomes before a
# pause / in isolation. Place of articulation is LEXICALLY determined (not recoverable from the
# surface: -ś -> k in diś/dṛś/spṛś but ṭ in viś; -j -> ṭ in the rāj/bhrāj roots but k elsewhere), so
# this is a per-stem lexicon harvested from the Vedic gold + Whitney §141-2/§218-9. The treebank is
# itself inconsistent (it writes ṛk/prāṅ/haviḥ/ṣaṭ AND ṛc/prāñc/haviṣ/ṣaṣ); mapping every genuine
# consonant-final to its avasāna collapses that to ONE canonical pausa form. Regular within a class
# (-c -> -k, -ñc -> -ṅ) except the -ś/-j place split; -s-stems (haviṣ/jyotiṣ/niṣ/dhīṣ) -> visarga -ḥ.
# The hapax `dadhṛṣ` is deliberately omitted (uncertain place) — it stays on the surface. Applied to
# EVERY member, compound-internal ones included (prāc- -> prāk-), so each stem has one pre-pausal form.
_LAW_OF_FINALS = {
    "abhoj": "abhok", "asṛj": "asṛk", "avāc": "avāk", "avāñc": "avāṅ", "bhiṣaj": "bhiṣak",
    "bhrāj": "bhrāṭ", "bhāj": "bhāk", "dharmarāj": "dharmarāṭ", "dhīṣ": "dhīḥ", "divispṛś":
    "divispṛk", "diś": "dik", "etādṛś": "etādṛk", "ghṛtāñc": "ghṛtāṅ", "haviṣ": "haviḥ",
    "hṛdispṛś": "hṛdispṛk", "jyotiṣ": "jyotiḥ", "nimruc": "nimruk", "nirṇij": "nirṇik", "niṣ":
    "niḥ", "parāc": "parāk", "parāñc": "parāṅ", "pratyañc": "pratyaṅ", "prāc": "prāk", "prāñc":
    "prāṅ", "rej": "rek", "ruc": "ruk", "ruj": "ruk", "rāj": "rāṭ", "samrāj": "samrāṭ", "saṃdṛś":
    "saṃdṛk", "saṃvṛj": "saṃvṛk", "sic": "sik", "spaś": "spaṭ", "sraj": "srak", "sruc": "sruk",
    "svarāj": "svarāṭ", "svāvṛj": "svāvṛk", "taijanitvac": "taijanitvak", "tvac": "tvak", "tādṛś":
    "tādṛk", "udañc": "udaṅ", "upadṛś": "upadṛk", "upapṛc": "upapṛk", "vanerāj": "vanerāṭ",
    "vaṇij": "vaṇik", "vibhrāj": "vibhrāṭ", "vipāś": "vipāṭ", "virāj": "virāṭ", "viś": "viṭ",
    "vāc": "vāk", "yuj": "yuk", "yādṛś": "yādṛk", "śuc": "śuk", "ūrj": "ūrk", "ṛc": "ṛk", "ṛtvij":
    "ṛtvik", "ṣaṣ": "ṣaṭ"
}
# t-assimilation family: a surface word-final -c/-j/-l before its trigger (all from an underlying -t,
# unless a genuine stem) -> revert to -t. Maps last-char -> (trigger-first-char, genuine-lexicon).
_TASSIM = {"c": ("c", _C_FINAL), "j": ("j", _J_FINAL), "l": ("l", _L_FINAL)}
_FAMILY_SHORT = {"a": "a", "i": "i", "u": "u"}
_FAMILY_LONG = {"a": "ā", "i": "ī", "u": "ū"}
# inverse of external_sandhi._coalesce: a right word's initial mark -> (left-vowel family, the
# right word's original initial vowel). The left word's final vowel is short/long per its '/" .
_MARK_INV = {
    "â": ("a", "a"), "ā": ("a", "ā"), "ê": ("a", "i"), "ē": ("a", "ī"),
    "ô": ("a", "u"), "ō": ("a", "ū"), "âi": ("a", "e"), "ai": ("a", "ai"),
    "âu": ("a", "o"), "āu": ("a", "au"),
    "î": ("i", "i"), "ī": ("i", "ī"),
    "û": ("u", "u"), "ū": ("u", "ū"),
}
_MARK_INV = {unicodedata.normalize("NFC", k): v for k, v in _MARK_INV.items()}
_MARKS_SORTED = sorted(_MARK_INV, key=len, reverse=True)         # longest first (âi/âu/āu/ai)
# circumflex-bearing marks are UNAMBIGUOUS (a circumflex vowel is never a genuine letter), so a
# word starting with one can be reverted even when its left partner is an unmarked particle.
_CIRC_MARKS = sorted([m for m in _MARK_INV if "̂" in unicodedata.normalize("NFD", m)],
                     key=len, reverse=True)


def _restore_pair(L, R):
    """Coalescence junction: L ends in '/" (maybe before a compound '-'), R starts with a mark."""
    tail, Lc = "", L
    if Lc.endswith("-"):
        tail, Lc = "-", Lc[:-1]
    if not Lc or Lc[-1] not in (_APOS, _DAPOS):
        return None
    short = Lc[-1] == _APOS
    for m in _MARKS_SORTED:
        if R.startswith(m):
            fam, v2 = _MARK_INV[m]
            v1 = (_FAMILY_SHORT if short else _FAMILY_LONG)[fam]
            return Lc[:-1] + v1 + tail, v2 + R[len(m):]
    return None


def _restore_circumflex_start(s):
    for m in _CIRC_MARKS:
        if s.startswith(m):
            return _MARK_INV[m][1] + s[len(m):]                  # restore the right word's vowel
    return s


def _restore_trailing(s):
    """An unpaired '/" — left word elided before an unmarked particle; restore the a-stem vowel."""
    tail = ""
    if s.endswith("-"):
        tail, s = "-", s[:-1]
    if s.endswith(_APOS):
        return s[:-1] + "a" + tail
    if s.endswith(_DAPOS):
        return s[:-1] + "ā" + tail
    return s + tail


def _restore_avagraha(s):
    if s.startswith(_APOS):
        return "a" + s[1:]                                       # avagraha: elided initial a
    if s.startswith(_DAPOS):
        return "ā" + s[1:]                                       # elided initial ā
    return s


def _split_tail(w):
    """Detach a trailing compound-join '-' (runtime member forms carry it; corpus forms do not)."""
    return (w[:-1], "-") if w.endswith("-") else (w, "")


def _first_char(w):
    wc, _ = _split_tail(w)
    return wc[0] if wc else ""


def _punct_kind(w, single_danda_medial=False):
    """Empty string if `w` is not a punctuation token, else "medial" (sandhi-transparent, per
    `_MEDIAL_PUNCT`) or "pause" (opaque: a sentence-final mark or a daṇḍa). `single_danda_medial`
    is set by `_next_word` when the text closes its sentences with a DOUBLE daṇḍa, which demotes a
    single daṇḍa to a pāda boundary — medial, so sandhi reads across it."""
    if not w or not all(c in _PUNCT for c in w):
        return ""
    if single_danda_medial and _danda_strokes(w) == 1:
        return "medial"
    return "medial" if all(c in _MEDIAL_PUNCT for c in w) else "pause"


def _next_word(out):
    """For each token, the index of the next WORD token reachable across sentence-MEDIAL punctuation
    only (None if a pause mark or the end of the input intervenes) — the neighbour the non-coalescent
    junction rules take, so that sandhi is reversed straight across an editorial comma / quotation
    mark (and across a pāda-boundary single daṇḍa) but never across a pause. See the section comment
    above. The single-daṇḍa test is document-dependent, so it is evaluated over the WHOLE token list:
    a single daṇḍa is medial exactly when some DOUBLE daṇḍa is present to close the sentences."""
    single_danda_medial = any(_danda_strokes(w) >= 2 for w in out)
    nxt: list = [None] * len(out)
    following = None
    for i in range(len(out) - 1, -1, -1):
        nxt[i] = following
        kind = _punct_kind(out[i], single_danda_medial)
        if not kind:
            following = i                        # a word: the neighbour for everything to its left
        elif kind == "pause":
            following = None                     # a pause: nothing to its right is a sandhi partner
    return nxt


def _rev_visarga_vowel(out, nxt):
    """STAGE 0 — restore a dropped visarga BEFORE the coalescence marks are undone. Run first so a
    coalescence-derived hiatus (L ends in the elision mark ', R in a circumflex/macron mark) is
    never mistaken for a dropped visarga (L in a plain -a/-ā, R in a plain vowel). Mutates `out`."""
    for i, j in enumerate(nxt):
        if j is None or _punct_kind(out[i]):
            continue
        Lc, tail = _split_tail(out[i])
        if not Lc:
            continue
        Rc = _split_tail(out[j])[0]
        r0 = Rc[0] if Rc else ""
        # A vowel hidden under a coalescence/avagraha mark ('/") on the NEXT token still triggered the
        # glide/visarga on THIS word, so it counts as a following vowel. This happens when the next
        # word is a single-vowel particle (preverb ā, emphatic u, a) that itself coalesces FORWARD, so
        # its vowel survives only as the mark and stage 0 (which runs before the marks are undone)
        # would otherwise see a non-vowel: nayatu ā agram -> nayatv " âgram (yaṇ), atha u iti -> ath' v
        # ' (u -> v), tau ā iha -> tāv " iha (ayādi). A bare glide particle (next token is just y/v, the
        # emphatic u / i reduced before a vowel) is likewise a following vowel (vai u X -> vāy v X).
        rvow = r0 in _PLAIN_VOWEL or r0 in (_APOS, _DAPOS) or Rc in ("y", "v")
        last = Lc[-1]
        # A bare glide token is itself the emphatic vowel particle (u -> v / i -> y before a vowel).
        # Restore the short vowel (length lost, as in yaṇ). Must precede the yaṇ branch, which assumes
        # a preceding consonant (len >= 2).
        if len(Lc) == 1 and Lc in ("y", "v") and rvow:
            out[i] = {"y": "i", "v": "u"}[Lc] + tail
            continue
        # ayādi glide before a vowel — unambiguous once external_sandhi keeps the glide (a genuine
        # -a/-ā + V would coalesce; yaṇ puts the glide after a CONSONANT: -Cy/-Cv, so a vowel before
        # the glide marks ayādi). Restore the diphthong/mid vowel.
        if rvow and len(Lc) >= 2 and Lc[-1] in "yv" and Lc[-2] in "aā":
            v = {"ay": "e", "av": "o", "āy": "ai", "āv": "au"}[Lc[-2:]]
            out[i] = Lc[:-2] + v + tail                   # -ay/-av/-āy/-āv + V  <-  -e/-o/-ai/-au
        elif rvow and len(Lc) >= 2 and Lc[-1] in "yv" and Lc[-2] not in _PLAIN_VOWEL:
            out[i] = Lc[:-1] + {"y": "i", "v": "u"}[last] + tail  # yaṇ -Cy/-Cv + V <- -Ci/-Cu (length
            #                                                      lost; short i/u the default: iti, not itī)
        elif last in ("a", "ā") and rvow:
            out[i] = Lc + "ḥ" + tail                     # -a/-ā + V  <-  -aḥ/-āḥ (dropped visarga)
        elif last == "o" and r0 == _APOS:
            out[i] = Lc[:-1] + "aḥ" + tail               # -o ' <- -aḥ a  (namo 'stu -> namaḥ astu)
        elif last == "o" and r0 in _VOICED_C and not rvow:
            out[i] = Lc[:-1] + "aḥ" + tail               # -o + voiced <- -aḥ  (vatso vira- ...)
    return out


def _rev_final_consonant(out, nxt):
    """STAGE 2 — deterministic final-consonant pausa reductions (place of articulation preserved,
    hence unique). Mutates `out`."""
    for i in range(len(out)):
        if _punct_kind(out[i]):
            continue
        Lc, tail = _split_tail(out[i])
        if not Lc:
            continue
        j = nxt[i]
        r0 = _first_char(out[j]) if j is not None else ""   # "" at a pause / end of input
        last = Lc[-1]
        vowel_before = len(Lc) >= 2 and Lc[-2] in _PLAIN_VOWEL
        new = None
        if last == "ṃ" and r0 and r0 not in _PLAIN_VOWEL and r0 not in "śṣs":
            new = Lc[:-1] + "m"                           # anusvara -ṃ + non-sibilant C  ->  -m
        elif Lc.endswith("nn"):
            new = Lc[:-1]                                 # gemination -nn (+ V)  ->  -n
        elif last in "sr" and vowel_before:
            new = Lc[:-1] + "ḥ"                           # word-final s/r  ->  visarga (tatas/agnir)
        elif last in "dgb":
            new = Lc[:-1] + {"d": "t", "g": "k", "b": "p"}[last]  # voiced stop -> voiceless (place kept)
        if new is not None:
            out[i] = new + tail
    return out


def _rev_sibilant_and_c(out, nxt):
    """STAGE 1.5 — context-sensitive sibilant/palatal junctions, lexicon-gated (see the lexica above).
    A two-token pass on the raw consonant surface (independent of vowel coalescence). Mutates `out`.
      • -ś before c/ch, -ṣ before ṭ/ṭh  ->  -ḥ  (visarga), unless a genuine -ś/-ṣ stem;
      • -c/-j/-l before their trigger (c/ch, j/jh, l)  ->  -t  (unless a genuine stem or -ñc), and for
        the -c case if word2 begins ch and is not a genuine ch-word, its ch  ->  ś (t + ś -> c ch)."""
    for i, j in enumerate(nxt):
        if j is None or _punct_kind(out[i]):
            continue
        Lc, tail = _split_tail(out[i])
        Rc, rtail = _split_tail(out[j])
        if not Lc or not Rc:
            continue
        last, r0 = Lc[-1], Rc[:1]
        if last == "ś" and r0 == "c" and Lc not in _SIB_FINAL:
            out[i] = Lc[:-1] + "ḥ" + tail                # -ś + c/ch  <-  visarga (kratuś ca -> kratuḥ ca)
        elif last == "ṣ" and r0 == "ṭ" and Lc not in _SIB_FINAL:
            out[i] = Lc[:-1] + "ḥ" + tail                # -ṣ + ṭ/ṭh  <-  visarga
        elif last in _TASSIM and r0 == _TASSIM[last][0] \
                and Lc not in _TASSIM[last][1] and not Lc.endswith("ñc"):
            out[i] = Lc[:-1] + "t" + tail                # -c/-j/-l + trigger  <-  -t  (tac ca -> tat ca)
            if last == "c" and Rc[:2] == "ch" and Rc not in _CH_INITIAL:
                out[j] = "ś" + Rc[2:] + rtail            # -c ch-  <-  -t ś-  (paṭhec chiva -> paṭhet śiva)
    return out


def _rev_guna_r(out):
    """STAGE 0.5 — restore the vocalic ṛ/ḷ that the guṇa junction -a/-ā + ṛ -> -ar devocalised on WORD2
    (ca ṛṣiḥ -> ca rṣiḥ, etayā ṛcā -> etayā rcā; the forward rule is in `external_sandhi.join_pair`).
    Word1 keeps its own vowel here, so nothing is marked and the only cue is word2's initial r/l before
    a consonant — unambiguous, since no native Sanskrit word begins with r/l + consonant. Adjacency-only
    (like the other COALESCENT junction, stage 1: the two vowels merge into one syllable, so no editorial
    mark can intervene), and run AFTER stage 0, whose dropped-visarga rule must still see this word2 as
    consonant-initial: an unreduced ṛ- after -a marks a dropped visarga (-aḥ + ṛ- -> -a ṛ-), whereas the
    reduced r- shows word1's -a is genuine. Mutates `out`."""
    for i in range(1, len(out)):
        Rc, rtail = _split_tail(out[i])
        if len(Rc) < 2 or Rc[0] not in "rl":
            continue
        if Rc[1] in _PLAIN_VOWEL or Rc[1] in (_APOS, _DAPOS) or unicodedata.combining(Rc[1]):
            continue                                     # r/l + vowel (or a marked vowel): a genuine word
        Lc = _split_tail(out[i - 1])[0]
        if Lc and Lc[-1] in ("a", "ā"):
            out[i] = {"r": "ṛ", "l": "ḷ"}[Rc[0]] + Rc[1:] + rtail
    return out


def _rev_law_of_finals(out):
    """STAGE 3 — normalise a GENUINE consonant-final stem to its avasāna (pausa) form (`_LAW_OF_FINALS`:
    vāc->vāk, diś->dik, rāj->rāṭ, haviṣ->haviḥ). Applied to EVERY member, compound-internal ones
    included (a compound join marker -/| is stripped, the reduction applied, the marker reattached) so
    each stem gets ONE canonical pre-pausal form regardless of position (prāc- -> prāk-). Mutates `out`."""
    for i, w in enumerate(out):
        tail, base = ("", w)
        if base and base[-1] in "-|":
            tail, base = base[-1], base[:-1]
        red = _LAW_OF_FINALS.get(base)
        if red is not None:
            out[i] = red + tail
    return out


def _danda_strokes(text):
    """Total daṇḍa strokes if `text` is WHOLLY daṇḍa marks (| / ‖ or an Indic-script daṇḍa ।॥ …),
    else 0. | // and a double-daṇḍa char count 2; a single | / ।  counts 1."""
    if not text:
        return 0
    total = 0
    for c in text:
        if c in "|/":
            total += 1
        elif c == "‖":                                   # ‖ U+2016 DOUBLE VERTICAL LINE
            total += 2
        else:
            try:
                name = unicodedata.name(c)
            except ValueError:
                return 0
            if not name.endswith("DANDA"):
                return 0
            total += 2 if "DOUBLE DANDA" in name else 1
    return total


def _normalise_danda(out):
    """Normalise a DOUBLE daṇḍa (|| // ॥ ।। or any ≥2-stroke run) to the single char ‖ (U+2016); a
    single daṇḍa (|) is left unchanged. Runs BOTH in the runtime tokeniser and (via `desandhi_csl` in
    revert_csl_sandhi.py) in the corpus build, so training data and inference stay in step."""
    for i, w in enumerate(out):
        if _danda_strokes(w) >= 2:
            out[i] = "‖"
    return out


def desandhi_csl(words):
    """Undo the CSL sandhi across a token list, preserving token count. Reverses the notation-marked
    vowel coalescence + avagraha AND the unambiguous subset of unmarked consonant/visarga external
    sandhi (see the section comment above), then normalises daṇḍa marks (|| -> ‖). Returns a new list."""
    out = [unicodedata.normalize("NFC", w) for w in words]
    nxt = _next_word(out)                                         # neighbour across MEDIAL punctuation
    _rev_visarga_vowel(out, nxt)                                  # STAGE 0 (on the raw surface)
    _rev_guna_r(out)                                              # STAGE 0.5 (-a + ṛ- -> -a r-, adjacent)
    _rev_sibilant_and_c(out, nxt)                                 # STAGE 1.5 (sibilant/palatal junctions)
    for i in range(len(out) - 1):                                 # STAGE 1: vowel coalescence (adjacent —
        #                                                           a fused syllable admits no punctuation)
        res = _restore_pair(out[i], out[i + 1])
        if res:
            out[i], out[i + 1] = res
    for i in range(len(out)):
        out[i] = _restore_circumflex_start(out[i])
        out[i] = _restore_trailing(out[i])
        out[i] = _restore_avagraha(out[i])
    _rev_final_consonant(out, nxt)                                # STAGE 2 (on the de-vowelled surface)
    _rev_law_of_finals(out)                                       # STAGE 3 (genuine finals -> avasāna)
    _normalise_danda(out)                                         # STAGE 4 (double daṇḍa -> ‖)
    return out


@registry.tokenizers("sa.SanskritInputTokenizer.v1")
def make_sanskrit_input_tokenizer():
    def create(nlp):
        return SanskritInputTokenizer(nlp.vocab)
    return create


class SanskritInputTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        norm = _PIPE.sub("-", normalise(text))       # CSL compound | -> hyphen (not the daṇḍa |)
        words, spaces = [], []
        for chunk in norm.split():
            toks = []
            for m in _SPLIT.finditer(chunk):
                if m.group(1) is not None:                # a punctuation run (||, |, , ? …)
                    toks.append(m.group(0))
                else:                                      # a word run: split internal hyphens
                    toks.extend(_HYPH.findall(m.group(0)))
            for j, tk in enumerate(toks):
                words.append(tk)
                spaces.append(j == len(toks) - 1)          # space only after a chunk's last token
        if spaces:
            spaces[-1] = False
        if not words:
            words, spaces = [norm or text], [False]
        words = desandhi_csl(words)                    # reverse CSL-marked sandhi (vowel/avagraha)
        # Drop the compound-join marker so members are clean wordforms, matching the training
        # data: internally a compound member is split with a trailing '-' (from _HYPH), which is
        # removed here (śuka- -> śuka). The grouping is recoverable from the Compound=Yes FEAT
        # (samāsa members) plus token adjacency (no SpaceAfter between members). A lone dash (the
        # '-' PUNCT, length 1) is a genuine dash — left as is.
        words = [w[:-1] if len(w) > 1 and w.endswith("-") else w for w in words]
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
