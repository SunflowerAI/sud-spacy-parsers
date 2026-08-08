"""Lexicons and character classes shared by the `Reported` gold builder and the runtime component.

`sud_reported_gold.py` applies these to CoNLL-U dicts to build the bootstrapped gold;
`sud_reported_rule.py` applies the same tests to a predicted `Doc` at inference. They must not
drift apart, so both import from here.

The speech/writing verb lists were seeded from each treebank's own head-lemma distribution over
complements bearing direct-speech evidence (the shape of `_derive_comp_frames`), then curated --
the raw ranking also surfaces `be`/`have`/`do` in English and modals in Latin, because quotation
marks mark scare quotes and titles as well as speech.
"""

# An ARM name is not always a language. `en_gum` is a second English arm (EWT + the ten
# non-NonCommercial GUM genres), so every lexicon below should answer for English. This lives here,
# beside the lexicons, because the gold builder, the runtime rule and the eval all key into them --
# and a rule handed an unknown key gets an EMPTY set, fires on nothing, and scores a flat 0.00 that
# reads as a finding. That is what it did for en_gum until 2026-08-08.
BASE_LANG = {"en_gum": "en"}


def base_lang(lang):
    """The language whose lexicons an arm should use."""
    return BASE_LANG.get(lang, lang)


SPEECH_VERBS = {
    "en": {"say", "tell", "ask", "reply", "answer", "state", "write", "add", "explain",
           "declare", "announce", "note", "remark", "question", "complain", "conclude",
           "insist", "claim", "argue", "report", "respond", "shout", "whisper", "mention",
           "recall", "admit", "observe", "comment", "warn", "suggest"},
    "ar": {"قَال", "أَضَاف", "أَوضَح", "أَكَّد", "شَدَّد", "تَابَع", "أَشَاد", "تَسَاءَل", "صَرَّح",
           "ذَكَر", "أَعلَن", "كَتَب", "سَأَل", "أَجَاب", "رَوَى", "أَفَاد", "لَفَت", "أَشَار", "رَدّ"},
    "fa": {"گفت", "نوشت", "پرسید", "افزود", "اظهار", "پاسخ", "نامید", "خواند", "اعلام"},
    "la": {"dico", "aio", "inquam", "loquor", "respondeo", "scribo", "narro", "fateor",
           "refero", "nuntio", "clamo", "praedico", "interrogo", "confiteor", "adicio",
           "subiungo", "praecipio"},
    "sa": {"vac", "ah", "brū", "vad", "bhāṣ", "kath", "gad", "ūc", "abhivad", "prabrū"},
}

# Overt complementisers. SUD is functional-head, so where one is present it IS the complement
# token -- always test that token, never the subtree (a subordinate clause *inside* a verbatim
# quote must not count as evidence of indirectness).
COMPLEMENTISERS = {
    "en": {"that", "whether", "if"},
    "ar": {"أَنّ", "إِنّ", "أَن"},
    "fa": {"که"},
    "la": {"quod", "quia", "ut", "an", "si", "num", "utrum", "quin", "quominus"},
    "sa": {"yad", "yathā"},
}

# Latin interrogatives. Used ONLY to withhold a positive, never to commit a negative: `qui` is
# overwhelmingly the relative pronoun, so its presence alone proves nothing.
LA_INTERROGATIVE = {"quis", "quid", "qui", "quomodo", "cur", "ubi", "unde", "quando",
                    "qualis", "quantus", "quare", "quotiens", "uter", "quantum", "quo",
                    "numquid", "quisnam"}

QUOTES = set('"“”‘’«»„「」『』‹›')

# Sanskrit's quotative particle: closes a verbatim quote, so it does the work of a closing quote.
SA_QUOTATIVE = {"iti"}
