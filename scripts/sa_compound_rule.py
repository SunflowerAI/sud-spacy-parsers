#!/usr/bin/env python3
"""When a Sanskrit multiword token's non-final member is a compound member the annotator did not mark.

THE RULE. A samāsa member appears in STEM form: the FINAL member carries the compound's case,
number and gender, and every member before it is morphologically bare. So inside a multiword token,
a non-final member that

  (a) belongs to a word class capable of nominal morphology — NOUN, PROPN, DET, NUM, ADJ, PRON, or
      a PARTICIPLE — and
  (b) carries no morphological features PROPER, and
  (c) sits in a multiword token whose FINAL member can END a compound

is a compound member whether or not `Compound=Yes` is on it: a nominal with nothing to show for its
inflection is a stem, and a stem inside a word is samāsa.

`VerbForm` is not a morphological feature for the purposes of (b). It names the word class —
participle, converb, finite verb — rather than saying how the word is inflected, so a participle
whose only FEATS is `VerbForm=Part` is morphologically bare and clause (b) admits it. It is also
what identifies clause (a)'s participle in the first place, since neither treebank gives one a UPOS
of its own: both write `VERB` + `VerbForm=Part`. See `POS_SUBTYPE_FEATS`.

WHY IT IS WORTH RUNNING. The token it recovers in UFAL is `dharma` in `dharmopārjitabhūrivibhavo`
(panc1.s6) — a bahuvrīhi whose other two non-final members are both `Compound=Yes`. The treebank DID
record it, in the XPOS column, one field left of where it belongs; `normalise_sa_xpos.py` cleaned
that cell and deliberately left FEATS alone, so the only record of the membership was deleted and
the compound has been a member short since. Without the mark, `sa_csl_prep` renders the first member
as a separate coalesced word (`dharm' ôpārjita-…`) rather than hyphen-joining it, and the tokeniser
downstream never sees a compound at all. DCS adds five: three bare nominals in one nāmāvalī passage
whose FEATS were never filled in (`svasti-daḥ`, `bhāga-karaḥ`, `sarva-dehinām`), and two participles
that only clause (b)'s treatment of `VerbForm` lets through (`vṛkta-barhiṣam`, `svayaṃvara-āgatā-
tyāgāt`).

WHY CLAUSE (c) IS NEEDED. "No morphological features" means "is a bare stem" only if the treebank
annotates morphology everywhere it exists. It does not, and an unannotated inflected word is
indistinguishable from a stem by FEATS alone. Clause (a)+(b) on their own fire on 11 tokens across
UFAL and DCS, of which 5 are inflected words in a multiword token that is not a compound at all:

  * four `X-aḥ + ca` sandhi joins in the same DCS passage — `svastibhāvaḥ`, `utsaṅgaḥ`, `mahāṅgaḥ`,
    `suvarṇaḥ`, all visarga-final nominatives with empty FEATS, enclitic `ca` for a host;
  * UFAL's `dūra` in `dūrādevāśṛṇot` (panc1.s84), where the orthographic word is `dūrāt eva aśṛṇot`
    and the FORM has simply lost its ablative ending.

Clause (c) rejects all five and none of the six true members, because it is the same premise the
rule rests on read from the other end: if the final member carries the compound's morphology, then
a conjunction, a particle or a FINITE verb cannot be one. Measured over both sources: 11 tokens
without it, 6 of them right; 6 tokens with it, all 6 right.

`looks_inflected` survives as a TRIPWIRE rather than a test — it is what caught the four `ca` joins
before clause (c) existed. Under the rule as it now stands it should never fire; a caller that
reports it will hear about the next annotation gap of that shape instead of absorbing it silently.
"""

# Capable of nominal morphology, by UPOS alone. A PARTICIPLE is capable of it too and is in scope
# by exactly the same argument, but neither treebank gives it a UPOS of its own — both write it
# `VERB` + `VerbForm=Part` — so it is recognised by `is_participle` rather than from this set.
NOMINAL_UPOS = frozenset(("ADJ", "DET", "NOUN", "NUM", "PRON", "PROPN"))

# `VerbForm` states WHICH KIND OF WORD this is — participle, converb, finite verb — not how that
# word is inflected. It is a POS subtype, so it is NOT a morphological feature for the purposes of
# clause (b) and `morph_feats` drops it: a participle whose only FEATS is `VerbForm=Part` is
# morphologically bare, and inside a multiword token that makes it a stem like any other. Two DCS
# tokens turn on this and both are right — `vṛkta` in `vṛkta-barhiṣam` (the Rigvedic bahuvrīhi) and
# `āgatā` in `svayaṃvara-āgatā-tyāgāt`, which sits between a `Case=Cpd` member and an ablative final
# member. Counting `VerbForm` as morphology would have silently excluded both.
POS_SUBTYPE_FEATS = ("VerbForm",)

# Word-final shapes that are a case ending rather than the end of a stem. Visarga is the giveaway in
# practice (nom./abl./gen. sg. of the commonest stem classes); `-m` is the acc. sg. / nom.-acc. neut.
_INFLECTED_ENDINGS = ("ḥ", "ः", "म्")


def morph_feats(feats):
    """FEATS that are MORPHOLOGY PROPER — inflection, and nothing else.

    Dropped: the compound-membership flag under either spelling (`Compound=Yes` here, `Case=Cpd` in
    DCS before `dcs_to_training.fix_feats` renames it), and the POS-subtype feats in
    `POS_SUBTYPE_FEATS`. Accepts CoNLL-U's `_` and the empty string DCS writes in its place.
    """
    out = []
    for f in (feats or "").split("|"):
        if not f or f == "_" or f == "Case=Cpd" or f.startswith("Compound="):
            continue
        if f.split("=", 1)[0] in POS_SUBTYPE_FEATS:
            continue
        out.append(f)
    return out


def has_compound(feats):
    """Whether the compound-membership flag is already present, under either spelling."""
    parts = (feats or "").split("|")
    return "Compound=Yes" in parts or "Case=Cpd" in parts


def verb_form(feats):
    """The `VerbForm` value, or None. Read directly — `morph_feats` deliberately drops it."""
    for f in (feats or "").split("|"):
        if f.startswith("VerbForm="):
            return f.split("=", 1)[1]
    return None


def is_participle(feats):
    """Whether this token is a participle — the one nominal-capable word class with no UPOS."""
    return verb_form(feats) == "Part"


def nominal_capable(upos, feats):
    """Clause (a): capable of nominal morphology — a nominal UPOS, or a participle."""
    return upos in NOMINAL_UPOS or is_participle(feats)


def can_end_compound(upos, feats):
    """Clause (c): whether this token could be a samāsa's FINAL member — the one that inflects.

    The same word classes as clause (a): a nominal or a participle can end a compound; a
    conjunction, a particle, an adverb or a FINITE verb cannot, and a multiword token ending in one
    is not a compound at all.
    """
    return nominal_capable(upos, feats)


def is_implicit_member(upos, feats):
    """Clauses (a) and (b): nominal-capable, no morphology proper, no flag yet.

    Clause (c) is a property of the multiword token, not of the token, so the caller applies it —
    see `can_end_compound`.
    """
    return nominal_capable(upos, feats) and not has_compound(feats) and not morph_feats(feats)


def add_compound(feats):
    """`feats` with `Compound=Yes` added, keeping everything else.

    NB it keeps the POS-subtype feats too, which `morph_feats` does not: a participle member is
    stamped `Compound=Yes|VerbForm=Part`, and dropping the `VerbForm` would throw away the only
    thing recording that the member is a participle at all.
    """
    keep = [f for f in (feats or "").split("|")
            if f and f != "_" and f != "Case=Cpd" and not f.startswith("Compound=")]
    return "|".join(sorted(keep + ["Compound=Yes"]))


def looks_inflected(form):
    """Tripwire: whether `form` still shows a case ending, so it is a word rather than a stem."""
    return bool(form) and form.endswith(_INFLECTED_ENDINGS)


def is_range(tid):
    """Whether a CoNLL-U ID column names a multiword-token range."""
    return "-" in tid and tid.split("-")[0].isdigit()


def stamp_rows(rows):
    """Apply the whole rule to one sentence of CoNLL-U column lists, IN PLACE.

    `rows` must include the MWT range lines — the rule is about position inside a multiword token,
    so a reader that skips them cannot apply it. Returns the token rows that were stamped, for
    reporting. Every caller that works in raw columns goes through this, so the two DCS derivations
    (`dcs_to_training.py` for the morphologiser/lemmatiser/unsandhi, `dcs_to_samhita.py` for the
    CSLiser's pairs) cannot drift apart on which members are compound members.
    """
    by_id = {int(c[0]): c for c in rows if not is_range(c[0]) and "." not in c[0]}
    stamped = []
    for c in rows:
        if not is_range(c[0]):
            continue
        a, b = (int(x) for x in c[0].split("-"))
        last = by_id.get(b)
        if last is None or not can_end_compound(last[3], last[5]):
            continue
        for i in range(a, b):
            t = by_id.get(i)
            if t is None or not is_implicit_member(t[3], t[5]):
                continue
            t[5] = add_compound(t[5])
            stamped.append(t)
    return stamped
