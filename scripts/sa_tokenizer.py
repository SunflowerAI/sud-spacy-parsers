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

**Source offsets.** Because the tokeniser *rewrites* what it reads (Devanagari -> IAST, accents
stripped, sandhi reverted), `doc.text` is NOT the input string and no token form need be a
substring of it — so the usual `token.idx` is an offset into the reconstructed text and is useless
to a caller that wants to highlight the input. Each token therefore also carries the character span
of the RAW INPUT it came from, as spaCy extensions (registered below):

    doc._.src_text          the exact string this tokeniser was handed
    doc._.src_spans         list, one entry per token: (start, end) into `src_text`, or None
    token._.src_span        that token's entry

The spans are **purely additive** — the tokens themselves (and hence the parser's input) are
byte-identical to what this module produced before, which is what lets the wheel be repackaged
without retraining. `None` is reported wherever the span cannot be established honestly rather
than a guess (see `_normalise_aligned`).
"""
import re
import unicodedata

from spacy.language import Language
from spacy.tokens import Doc, Token
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
    return _normalise_body(text, _has_devanagari(text))


def _normalise_body(text, deva):
    """`normalise` minus the (index-preserving) quote straightening, with the Devanagari test
    passed in so a SEGMENT can be normalised exactly as the whole string would be — see
    `_normalise_aligned`, which needs a piecewise normalisation to recover source offsets."""
    if deva:
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
# SOURCE OFFSETS. `normalise` is not length-preserving (Devanagari -> IAST is many-to-many, an
# accent mark vanishes), so an index into its output is not an index into the input. Aligning it
# character by character would mean tracking the transliterator's akshara arithmetic — but nothing
# needs that resolution: EVERY token boundary this tokeniser can produce falls on a character that
# `normalise` maps 1:1 anyway. `norm.split()` cuts at whitespace, `_SPLIT` cuts at `_PUNCT`, and
# `_HYPH` cuts at `-`; so if the input is segmented at exactly those three classes of character
# ("anchors"), no token ever straddles a segment and a per-SEGMENT alignment is already exact. A
# maximal run of non-anchor characters is at most one token (a Devanagari word has no interior cut
# point), so per-segment == per-token there.
#
# Anchors are tested on the STRAIGHTENED text, since `_STRAIGHTEN` maps the curly quotes — which
# are in `_PUNCT`, hence anchors — onto ' and ", which are not. That translate is 1:1, so it does
# not disturb indices. The Indic daṇḍas ।/॥ are anchors too and so are transliterated on their own
# (।->| 1:1, ॥->|| 1:2), each mapping wholly to its one source character.
#
# The piecewise result is CHECKED against `normalise(text)` and the offsets are abandoned (all
# None) if the two disagree, so an exotic input can cost the spans but can never change the tokens.
def _segments(s):
    """Split `s` at anchors: each whitespace / '-' / `_PUNCT` character is its own segment, and a
    maximal run of everything else is one segment. Returns a list of (start, end) into `s`."""
    segs, i, n = [], 0, len(s)
    while i < n:
        if s[i].isspace() or s[i] == "-" or s[i] in _PUNCT:
            segs.append((i, i + 1))
            i += 1
        else:
            j = i + 1
            while j < n and not (s[j].isspace() or s[j] == "-" or s[j] in _PUNCT):
                j += 1
            segs.append((i, j))
            i = j
    return segs


def _normalise_aligned(text):
    """Return (norm, start_of, end_of) where `norm` is exactly `normalise(text)` and the two dicts
    map a boundary position in `norm` to the corresponding boundary position in `text`
    (`start_of[a]` = where the segment starting at `a` starts in the source; `end_of[b]` = where
    the segment ending at `b` ends). A token's normalised range [a, b) is convertible iff BOTH
    boundaries are present — which is the honest test, since a range that begins or ends inside a
    segment has no exact source counterpart. `start_of`/`end_of` are None when the piecewise
    normalisation fails to reproduce `norm` (then no token gets a span)."""
    norm = normalise(text)
    s0 = text.translate(_STRAIGHTEN)                    # 1:1, so indices into s0 == indices into text
    deva = _has_devanagari(s0)
    pieces, start_of, end_of, pos = [], {}, {}, 0
    for a, b in _segments(s0):
        out = _normalise_body(s0[a:b], deva)
        pieces.append(out)
        # First writer wins for a start and last writer wins for an end, so a segment that
        # normalises to nothing (a lone accent mark) is absorbed into its neighbour's span
        # rather than dropped out of every span.
        start_of.setdefault(pos, a)
        end_of[pos + len(out)] = b
        pos += len(out)
    if "".join(pieces) != norm:                          # exotic input: keep the tokens, drop the spans
        return norm, None, None
    return norm, start_of, end_of


def _iter_chunks(s):
    """`s.split()` with the offset of each chunk — same `str.isspace()` test `split()` itself uses,
    so the chunk sequence is identical to the one the tokeniser used before offsets existed."""
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        j = i
        while j < n and not s[j].isspace():
            j += 1
        if j > i:
            yield i, s[i:j]
        i = j


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


# NB the lexeme affix windows are spaCy's defaults (PREFIX = form[:1], SUFFIX = form[-3:]) and are
# deliberately NOT overridden here. Widening them for Sanskrit was tried (PREFIX 3 / SUFFIX 6) and
# regressed everything but the tagger — see the "do NOT widen sa PREFIX/SUFFIX" negative result in
# CLAUDE.md. They are overridable per language via `Sanskrit.Defaults.lex_attr_getters`, but any
# change requires retraining every component, since the models read them as input features.

# Join members that are NOT samāsa members, so they must not be stamped Compound=Yes. The CSL
# representation hyphen-joins three different things: compound (samāsa) members, verb preverbs
# (upasarga), and the privative a-/an-. Only the first is Compound=Yes in the treebank; the other
# two are separate ADV/PART/ADP tokens carrying no Compound feat. They are a closed class, and this
# is it — the classical upasarga inventory plus the privative, harvested from the training treebank
# as the join-member types that are predominantly NOT Compound=Yes. With this exclusion the rule
# "carried a join marker and is not in this set" predicts the treebank's Compound=Yes at precision
# 0.9998 / recall 0.9997 over non-elided tokens (TP 6463, FP 1, FN 2); the bare join marker alone
# scores only 0.775/0.713. `sam` and `su` each occur once as a genuine compound member and are lost,
# which is the whole cost. (The 2 598 remaining gold Compound=Yes tokens the rule misses all have
# FORM `_` — elided tokens that exist only in the treebank, never in raw input.) `‖` is here because
# a handful of double daṇḍas pick up a join marker in the source; punctuation is never a compound.
_NON_COMPOUND_JOIN = frozenset(
    "a abhi adhi an anu apa apā ati ava ni niḥ pari parā pra prati "
    "sam saṃ su upa ut vi ā ‖".split()
)

# Source-offset extensions. Registered at import (this module is the wheel's `--code` payload, so
# importing the model registers them) and guarded, because loading two models in one process — or
# reloading one — imports it twice and `set_extension` raises on a duplicate.
# `force=` is deliberately NOT used: it would silently stomp on a caller's own extension.
if not Doc.has_extension("src_text"):
    Doc.set_extension("src_text", default=None)          # the raw string handed to the tokeniser
if not Doc.has_extension("src_spans"):
    Doc.set_extension("src_spans", default=None)         # per token: (start, end) into src_text, or None
if not Token.has_extension("src_span"):
    Token.set_extension("src_span", getter=lambda t: (
        (t.doc._.src_spans[t.i] if t.i < len(t.doc._.src_spans) else None)
        if t.doc._.src_spans is not None else None))
if not Doc.has_extension("compound_flags"):
    # Per token: did the tokeniser stamp Compound=Yes? The morphologizer overwrites token.morph
    # with its own prediction, and clause_parser rebuilds the doc for its re-parse, so the
    # tokeniser's decision has to survive somewhere neither of them touches.
    Doc.set_extension("compound_flags", default=None)


def _is_punct_text(w):
    return bool(w) and all(c in _PUNCT or c == "-" for c in w)


def _add_compound(morph, flag):
    """Set/clear Compound=Yes in a FEATS string, leaving every other feature alone."""
    feats = [f for f in str(morph).split("|") if f and not f.startswith("Compound=")]
    if flag:
        feats.append("Compound=Yes")
    return "|".join(sorted(feats))


@Language.factory("sa_compound")
def make_sa_compound(nlp, name):
    return SaCompound()


class SaCompound:
    """Guarantee the `Compound=Yes` INPUT feature is present however the Doc was built.

    The models read MORPH as an input feature (worth +1.30 LAS — see CLAUDE.md), and normally the
    tokeniser supplies it from the CSL join marker. But a caller who hands the pipeline TOKENS
    rather than raw text never invokes the tokeniser — `Doc(vocab, words=[...])`, pre-tokenised
    input, `spacy evaluate` — and the feature is then silently absent, costing ~6.8 LAS with no
    error. This component runs FIRST and closes that hole.

    When the tokeniser did run it defers to it (`doc._.compound_flags` is already set). Otherwise it
    re-derives the feature from token adjacency: a token bound to the next one with no intervening
    space is a samāsa member, unless it is one of the upasarga/privative join types
    (`_NON_COMPOUND_JOIN`) or either side of the junction is punctuation.

    On real text it is EXACT: over 300 test documents it reproduces the tokeniser's decision on
    19 584 / 19 584 tokens, zero disagreements. Precision against the treebank is 1.0000 — it never
    marks a non-compound.

    KNOWN LIMIT, and it bites harder than it looks. It cannot see a compound member that is
    **elided** — the treebank writes those with FORM `_` and a trailing space, so there is no
    adjacency to read (282 in the test split, and every single unrecoverable token is a `_`). Such
    tokens cannot occur in real input, so this is a treebank-only gap. But a PARTIALLY supplied
    feature turns out to be worse for the parser than none at all — on the Vedic test set, token
    input scores LAS 0.5169 with no Compound, 0.4826 with this fallback, 0.5601 with the full
    feature — because training always had the feature on every compound member, so an unmarked one
    is positive evidence of "not a compound" rather than absence of evidence. Hence: **evaluate on a
    treebank with `scripts/eval_sa_compound.py` (which supplies the feature from the reference), NOT
    with this component.** Marking every `_` token is not a fix: only 81 % of them are compounds, so
    the rule would trade the precision-1.0 property for a worse error.

    Only the Compound feature is touched; any other FEATS already on the token are preserved.
    """

    def __call__(self, doc):
        if doc._.compound_flags is not None:
            return doc                      # the tokeniser already decided; do not second-guess it
        n = len(doc)
        flags = [
            bool(tok.whitespace_ == "" and i + 1 < n
                 and tok.text not in _NON_COMPOUND_JOIN
                 and not _is_punct_text(tok.text)
                 and not _is_punct_text(doc[i + 1].text))
            for i, tok in enumerate(doc)
        ]
        for tok, flag in zip(doc, flags):
            tok.set_morph(_add_compound(tok.morph, flag))
        doc._.compound_flags = flags
        return doc


@registry.tokenizers("sa.SanskritInputTokenizer.v1")
def make_sanskrit_input_tokenizer():
    def create(nlp):
        return SanskritInputTokenizer(nlp.vocab)
    return create


class SanskritInputTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # `norm` is what gets tokenised; `start_of`/`end_of` translate a boundary in it back to the
        # raw input (None/None when that could not be done — see `_normalise_aligned`). `_PIPE.sub`
        # is a 1:1 character substitution, so it leaves those boundaries where they are.
        norm, start_of, end_of = _normalise_aligned(text)
        norm = _PIPE.sub("-", norm)                  # CSL compound | -> hyphen (not the daṇḍa |)
        words, spaces, at = [], [], []               # `at`: each token's [start, end) within `norm`
        for c0, chunk in _iter_chunks(norm):
            toks = []
            for m in _SPLIT.finditer(chunk):
                if m.group(1) is not None:                # a punctuation run (||, |, , ? …)
                    toks.append((m.group(0), c0 + m.start()))
                else:                                      # a word run: split internal hyphens
                    p = c0 + m.start()
                    for piece in _HYPH.findall(m.group(0)):   # the pieces tile the run exactly,
                        toks.append((piece, p))               # so their lengths give the offsets
                        p += len(piece)
            for j, (tk, off) in enumerate(toks):
                words.append(tk)
                spaces.append(j == len(toks) - 1)          # space only after a chunk's last token
                at.append((off, off + len(tk)))
        if spaces:
            spaces[-1] = False
        if not words and (norm or text):
            # Nothing but whitespace: one token standing for the whole input. (Empty input falls
            # through to an empty Doc — spaCy rejects a '' token with E031, so the old
            # `words = [norm or text]` raised on it.)
            words, spaces, at = [norm or text], [False], [(0, len(norm))]
        # The two transforms below both preserve the token COUNT (`desandhi_csl` rewrites in place;
        # the join-marker strip is a per-token edit), which is what lets the spans stay aligned even
        # though the forms change out from under them.
        words = desandhi_csl(words)                    # reverse CSL-marked sandhi (vowel/avagraha)
        # Drop the compound-join marker so members are clean wordforms, matching the training
        # data: internally a compound member is split with a trailing '-' (from _HYPH), which is
        # removed here (śuka- -> śuka). The grouping is recoverable from the Compound=Yes FEAT
        # (samāsa members) plus token adjacency (no SpaceAfter between members). A lone dash (the
        # '-' PUNCT, length 1) is a genuine dash — left as is.
        joined = [len(w) > 1 and w.endswith("-") for w in words]
        words = [w[:-1] if j else w for w, j in zip(words, joined)]
        # A join marker means "bound to the token on the right", which is a samāsa member EXCEPT for
        # the preverbs and the privative (see _NON_COMPOUND_JOIN). Stamping the feat here makes it
        # deterministic instead of predicted — the morphologizer scored only F 0.889 on Compound —
        # and, because the training corpora carry the same feat on the predicted doc (via
        # `sud.CompoundCorpus.v1`), it is also an INPUT feature for every component downstream.
        compound = [j and w not in _NON_COMPOUND_JOIN for w, j in zip(words, joined)]
        doc = Doc(self.vocab, words=words, spaces=spaces,
                  morphs=["Compound=Yes" if c else "" for c in compound])
        doc._.compound_flags = compound
        doc._.src_text = text
        # A span is emitted only when BOTH ends of the token's normalised range land on a segment
        # boundary; anything else would be a guess, and a caller can live with a hole but not with
        # a wrong span.
        doc._.src_spans = [None] * len(words) if start_of is None else [
            None if (a not in start_of or b not in end_of) else (start_of[a], end_of[b])
            for a, b in at]
        return doc

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
