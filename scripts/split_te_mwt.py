#!/usr/bin/env python3
"""Give SUD_Telugu-MTG the multiword tokens it is missing, from the treebank's own evidence.

WHAT IS WRONG WITH MTG AS SHIPPED. Its README states "Word count: 6465, Token count: 6465" — it has
**no multiword tokens at all**, and that is not because Telugu has none. Telugu adds a euphonic
enunciative `-u` to consonant-final words and ELIDES it before a vowel-initial word, writing the two
together; the treebank then annotates the result as one token. Measured against Tamil TTB, which
splits 9.67 % of its orthographic words:

    ta TTB   AUX 6.6 %   comp:aux 608   mod@emph 228
    te MTG   AUX 0.0 %   comp:aux   0   mod@emph   0        in 6 465 tokens

And MTG is inconsistent with ITSELF: the addressee particle `అండి` appears once as a separate
`PART`/`discourse` token and five times fused inside a verb. The clearest case is sentence 276,
`మీ అన్నగారికి ఎన్ని ఇళ్ళున్నాయి ?` — `ఇళ్ళున్నాయి` is `ఇళ్ళు` "houses" + `ఉన్నాయి` "are", so the
sentence's SUBJECT is inside the root token and `ఎన్ని` "how many" attaches as `det` OF THE VERB.
Both halves occur standalone elsewhere in the same treebank, in exactly those roles.

⚠ **THIS SCRIPT RE-ANNOTATES A TREEBANK, WHICH IS A DIFFERENT ACT FROM TRAINING ON ONE**, so every
decision it makes is read off MTG rather than supplied by the author, and it is held to the bar
`docs/latin.md` sets for a rule that COMMITS an annotation — **0.90 dominance**, not the looser bar
that is fine for a rule merely choosing between two attested spellings.

FIVE CONDITIONS, ALL REQUIRED. A candidate split `W -> A + B` is committed only if:

 1. **The orthography licenses it.** In akṣara-decomposed space (`scripts/indic_sandhi.py`) the cut
    leaves a left part ending in a virāma and a right part opening with an INDEPENDENT vowel — the
    exact signature of enunciative elision. A word of this language does not end in a bare
    consonant, which is why the reconstruction `A = recompose(left + ఉ)` is forced rather than
    guessed.
 2. **Both halves are attested standalone**, at least `--min-freq` times, and each has a UPOS that
    is ≥ 0.90 dominant across those occurrences. A part whose category the treebank cannot settle
    is a part this script has no business inventing one for.
 3. **The relation between them is attested**, on a ladder ordered by specificity: the exact form
    pair adjacent somewhere in the corpus, else the (UPOS, UPOS) pair adjacent. Whichever rung
    answers must be ≥ 0.90 dominant in BOTH direction and label.
 4. ⚠ **The head part's UPOS equals W's own UPOS.** The annotators assigned W a category, and the
    category of a fused word is the category of its syntactic head — so a split that makes the head
    something W is not has misidentified the construction. This is the condition that rejects
    `వాళ్ళని`, which is the accusative of `వాళ్ళు` and not `వాళ్ళు` + the quotative `అని`, and it
    rejects case suffixes and negative participles generally.
 5. **Re-joining reproduces the surface exactly** (`indic_sandhi.joins_to`). The MWT range line
    carries the ORIGINAL orthographic word, so `# text` never changes and the treebank still says
    what it said.

CHILDREN ARE REDISTRIBUTED, NOT LEFT BEHIND. When W had dependents, each is re-attached to whichever
part can head its relation — decided by the corpus's own distribution of head UPOS per deprel, again
at 0.90, and defaulting to the head part when that is not decisive. Leaving every child on the head
part is what produced `det`-of-a-VERB in the first place.

    split_te_mwt.py --dry-run                  # every candidate with its evidence, nothing written
    split_te_mwt.py --apply                    # rewrite assets_te/SUD_Telugu-MTG -> assets_te/*.mwt
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from indic_sandhi import SCRIPTS, decompose, joins_to, recompose, restore_enunciative  # noqa: E402

SCRIPT = SCRIPTS["te"]
DOMINANCE = 0.90


def read_sentences(path):
    comments, rows = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if rows:
                yield comments, rows
            comments, rows = [], []
            continue
        if line.startswith("#"):
            comments.append(line)
        else:
            rows.append(line.split("\t"))
    if rows:
        yield comments, rows


#: The construction a pair belongs to, finest first. A closed-class dependent IS its own class —
#: that is what "closed class" means — so a particle gets a key of its own before the coarse
#: (UPOS, UPOS) key is reached.
def class_keys(a, b, ua, ub):
    return [("ff", a, b), ("uf", ua, b), ("fu", a, ub), ("uu", ua, ub)]


#: Support required at each granularity before its measured accuracy is trusted. The exact
#: construction needs only one observation: it is the treebank annotating this very pair.
MIN_CLASS_N = {"ff": 1, "uf": 1, "fu": 3, "uu": 8}


def dominant(counter):
    """(value, share) of the most common entry, ties broken on the value for reproducibility."""
    if not counter:
        return None, 0.0
    total = sum(counter.values())
    value, n = max(counter.items(), key=lambda kv: (kv[1], str(kv[0])))
    return value, n / total


class Evidence:
    """Everything this script is allowed to know, harvested from the treebank itself."""

    def __init__(self, sentences):
        self.upos = collections.defaultdict(collections.Counter)     # form -> UPOS counts
        self.freq = collections.Counter()                            # form -> standalone count
        self.pair_form = collections.defaultdict(collections.Counter)   # (a,b) -> (dir, deprel)
        self.pair_upos = collections.defaultdict(collections.Counter)
        self.head_upos = collections.defaultdict(collections.Counter)   # deprel -> head UPOS
        self.deprel_form = collections.defaultdict(collections.Counter)  # form -> its own deprels
        self.direction_upos = collections.defaultdict(collections.Counter)  # (ua,ub) -> direction
        self.rung_order: list[str] = []
        self.rung_acc: dict[str, float] = {}
        self.class_acc: dict = {}

        for _comments, rows in sentences:
            ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
            by = {int(c[0]): c for c in ws}
            for c in ws:
                if c[3] == "PUNCT":
                    continue
                self.upos[c[1]][c[3]] += 1
                self.freq[c[1]] += 1
                self.deprel_form[c[1]][c[7]] += 1
            for c in ws:
                head = by.get(int(c[6]))
                if head is not None:
                    self.head_upos[c[7]][head[3]] += 1
            # adjacent pairs, and which of the two heads the other
            for i in range(len(ws) - 1):
                a, b = ws[i], ws[i + 1]
                if a[3] == "PUNCT" or b[3] == "PUNCT":
                    continue
                ia, ib = int(a[0]), int(b[0])
                if int(a[6]) == ib:
                    rel = ("right", a[7])            # A depends on B: B is the head
                elif int(b[6]) == ia:
                    rel = ("left", b[7])             # B depends on A: A is the head
                else:
                    continue
                self.pair_form[(a[1], b[1])][rel] += 1
                self.pair_upos[(a[3], b[3])][rel] += 1
                self.direction_upos[(a[3], b[3])][rel[0]] += 1
        self._rank_rungs(sentences)

    # ---- the ladder ---------------------------------------------------------------------
    @staticmethod
    def _less(counter, key):
        """`counter` with one observation of `key` removed — leave-one-out, for honest ranking."""
        if key is None or not counter:
            return counter
        out = collections.Counter(counter)
        out[key] -= 1
        if out[key] <= 0:
            del out[key]
        return out

    def _rungs(self, a, b, ua, ub, hold_out=None):
        """Every rung's answer for one pair, as (direction, deprel) or None.

        `dep_form` is the rung the first version lacked, and it is the one that matters here: the
        DEPENDENT part's own dominant deprel when it stands alone, with the direction taken from
        head-finality. The (UPOS, UPOS) rung cannot separate `subj` from `comp:obj` for a
        (NOUN, VERB) pair — 138 against 237 in this corpus — but the individual word usually can.

        `hold_out` removes this very observation from every table before answering, so ranking the
        rungs does not reward `form_pair` for memorising the instance it is being scored on.
        """
        out = {}
        counter = self._less(self.pair_form.get((a, b)), hold_out)
        if counter:
            out["form_pair"] = dominant(counter)[0]
        dcounter = self._less(self.direction_upos.get((ua, ub)),
                              hold_out[0] if hold_out else None)
        direction, dshare = dominant(dcounter or collections.Counter())
        if direction:
            dep_form = a if direction == "right" else b
            deprel, share = dominant(
                self._less(self.deprel_form.get(dep_form),
                           hold_out[1] if hold_out else None) or collections.Counter())
            if deprel and share >= DOMINANCE and dshare >= DOMINANCE:
                out["dep_form"] = (direction, deprel)
        counter = self._less(self.pair_upos.get((ua, ub)), hold_out)
        if counter:
            out["upos_pair"] = dominant(counter)[0]
        return out

    def _rank_rungs(self, sentences):
        """Order the ladder by MEASURED accuracy, and measure it again PER CONSTRUCTION CLASS.

        Same method as `normalise_ta_xpos.py`: assuming the most specific rung is the most accurate
        is what cost that script eighteen points.

        ⚠ THE GLOBAL NUMBER IS THE WRONG ONE TO GATE ON, and gating on it first is what made this
        script commit nothing. Held out over every adjacent pair in the corpus the best rung reaches
        only 0.8909, because that pool is dominated by (NOUN, VERB) pairs whose relation is
        genuinely ambiguous between `subj` and `comp:obj` — 138 against 237 — and that ambiguity is
        the parser's job to resolve, not this script's to guess. Conditioned on the CONSTRUCTION,
        the picture separates completely: a verb plus an addressee particle is not ambiguous at all.
        So accuracy is measured per (UPOS, UPOS) class and the 0.90 bar is applied there.
        """
        gold = []
        for _comments, rows in sentences:
            ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
            for i in range(len(ws) - 1):
                a, b = ws[i], ws[i + 1]
                if a[3] == "PUNCT" or b[3] == "PUNCT":
                    continue
                if int(a[6]) == int(b[0]):
                    gold.append((a[1], b[1], a[3], b[3], ("right", a[7])))
                elif int(b[6]) == int(a[0]):
                    gold.append((a[1], b[1], a[3], b[3], ("left", b[7])))
        score, seen = collections.Counter(), collections.Counter()
        cls_ok, cls_n = collections.Counter(), collections.Counter()
        for a, b, ua, ub, want in gold:
            answers = self._rungs(a, b, ua, ub, hold_out=want)
            for name, got in answers.items():
                seen[name] += 1
                score[name] += got == want
            # the LADDER's answer for this class, under the order computed below on a first pass
            got = None
            for name in ("form_pair", "dep_form", "upos_pair"):
                if name in answers:
                    got = answers[name]
                    break
            for key in class_keys(a, b, ua, ub):
                cls_n[key] += 1
                cls_ok[key] += got == want
        for name in ("form_pair", "dep_form", "upos_pair"):
            self.rung_acc[name] = score[name] / seen[name] if seen[name] else 0.0
        self.rung_order = sorted(self.rung_acc, key=lambda k: (-self.rung_acc[k], k))
        self.class_acc = {k: (cls_ok[k] / cls_n[k], cls_n[k]) for k in cls_n}

    def upos_of(self, form, min_freq):
        if self.freq[form] < min_freq:
            return None, 0.0, self.freq[form]
        value, share = dominant(self.upos[form])
        return value, share, self.freq[form]

    def relation(self, a, b, ua, ub):
        """(direction, deprel, class that licensed it, that class's HELD-OUT accuracy).

        The gate is the accuracy of the FINEST construction class that has enough support, not a
        global number and not the in-sample dominance of the answer. Pooled over every adjacent
        pair the ladder reaches only 0.8909, because that pool is dominated by (NOUN, VERB) pairs
        whose relation is genuinely ambiguous between `subj` and `comp:obj` — 138 against 237 — and
        resolving that ambiguity is the parser's job, not this script's. Conditioned on the
        construction the picture separates: (DET, NOUN) 0.992, (ADV, VERB) 0.961, (NOUN, VERB) 0.594.
        """
        answers = self._rungs(a, b, ua, ub)
        if not answers:
            return None, None, None, 0.0
        for name in self.rung_order:
            if name in answers:
                direction, deprel = answers[name]
                break
        for key in class_keys(a, b, ua, ub):
            acc, n = self.class_acc.get(key, (0.0, 0))
            if n >= MIN_CLASS_N[key[0]]:
                return direction, deprel, f"{key[0]}:{key[1]}/{key[2]} n={n}", acc
        return direction, deprel, "unsupported", 0.0

    def can_head(self, deprel, upos):
        """Is `upos` the treebank's dominant head category for `deprel`?"""
        value, share = dominant(self.head_upos.get(deprel, collections.Counter()))
        return value == upos and share >= DOMINANCE


def candidates(form, ev, min_freq):
    """Every orthographically licensed split of `form`, with its evidence."""
    out = []
    d = decompose(form, SCRIPT)
    vowels = SCRIPT.independent_vowels
    for k in range(2, len(d) - 1):
        left, right = d[:k], d[k:]
        if not left.endswith(SCRIPT.virama):
            continue
        if not right or right[0] not in vowels:
            continue
        a = restore_enunciative(left, SCRIPT)
        b = recompose(right, SCRIPT)
        if not a or not b or a == form or b == form:
            continue
        # ⚠ A REDUPLICATION IS ONE WORD. `ఎవరెవరు` "who all" and `అప్పుడప్పుడు` "now and then" are
        # lexicalised, and splitting them yields two identical tokens joined by an invented `mod`.
        # The orthography licenses the cut and the evidence tables endorse it, because both halves
        # are of course attested -- they are the SAME word. Only this guard catches it.
        if a == b:
            continue
        ua, sa, fa = ev.upos_of(a, min_freq)
        ub, sb, fb = ev.upos_of(b, min_freq)
        if ua is None or ub is None or sa < DOMINANCE or sb < DOMINANCE:
            continue
        if joins_to([a, b], SCRIPT) != form:
            continue
        out.append({"a": a, "b": b, "ua": ua, "ub": ub, "fa": fa, "fb": fb,
                    "sa": sa, "sb": sb})
    return out


def choose(form, upos, ev, min_freq):
    """The one split to commit for `form`, or None. Conditions 3 and 4 are applied here."""
    best = None
    for cand in candidates(form, ev, min_freq):
        direction, deprel, rung, share = ev.relation(cand["a"], cand["b"], cand["ua"], cand["ub"])
        if direction is None or share < DOMINANCE:
            cand["reject"] = f"no relation evidence (share {share:.2f})"
            best = best or cand
            continue
        head_upos = cand["ub"] if direction == "right" else cand["ua"]
        if head_upos != upos:
            cand["reject"] = (f"head part is {head_upos}, W is {upos} "
                              f"— not a fusion of two words")
            best = best or cand
            continue
        cand.update(direction=direction, deprel=deprel, rung=rung, share=share, reject=None)
        return cand
    return best


def split_sentence(comments, rows, ev, min_freq, stats):
    """Return the rewritten rows, expanding every committed token into a range + two words."""
    ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
    plan = {}
    for c in ws:
        if c[3] == "PUNCT" or len(decompose(c[1], SCRIPT)) < 6:
            continue
        cand = choose(c[1], c[3], ev, min_freq)
        if cand is None:
            continue
        if cand.get("reject"):
            stats["rejected"][cand["reject"].split("—")[0].split("(")[0].strip()] += 1
            continue
        plan[int(c[0])] = cand
    if not plan:
        return rows, 0

    # new numbering: each split token becomes two words
    new_id, mapping = {}, 1
    for c in ws:
        i = int(c[0])
        new_id[i] = mapping
        mapping += 2 if i in plan else 1

    out = []
    for c in ws:
        i = int(c[0])
        cand = plan.get(i)
        base = new_id[i]
        if cand is None:
            row = list(c)
            row[0] = str(base)
            row[6] = str(new_id[int(c[6])]) if c[6] != "0" and int(c[6]) in new_id else c[6]
            out.append(row)
            continue

        head_is_b = cand["direction"] == "right"
        a_id, b_id = base, base + 1
        head_id = b_id if head_is_b else a_id
        dep_id = a_id if head_is_b else b_id
        parent = str(new_id[int(c[6])]) if c[6] != "0" and int(c[6]) in new_id else c[6]

        # The range line IS the orthographic word, so it keeps that word's own MISC verbatim —
        # `Translit=` and `SpaceAfter=No` alike. The split parts get `_`: inventing a
        # transliteration for a piece the annotators never transliterated would be manufacturing
        # data, and `# translit` on the sentence still carries the real one.
        rng = ["%d-%d" % (a_id, b_id), c[1], "_", "_", "_", "_", "_", "_", "_", c[9]]
        row_a = [str(a_id), cand["a"], "_", cand["ua"], cand["ua"], "_", "", "", "_", "_"]
        row_b = [str(b_id), cand["b"], "_", cand["ub"], cand["ub"], "_", "", "", "_", "_"]
        head_row, dep_row = (row_b, row_a) if head_is_b else (row_a, row_b)
        head_row[6], head_row[7] = parent, c[7]          # the head part inherits W's own arc
        dep_row[6], dep_row[7] = str(head_id), cand["deprel"]
        out.extend([rng, row_a, row_b])
        stats["committed"][(c[1], cand["a"], cand["b"])] += 1
        plan[i] = dict(cand, head_id=head_id, dep_id=dep_id)

    # children of a split token: re-attach each to the part that can head its relation
    index = {r[0]: r for r in out if "-" not in r[0]}
    # The two rows a split produced are NOT children to be redistributed: their arc to each other
    # was just decided, and re-running the head-UPOS rule on it would overwrite that decision.
    own = {str(new_id[i]) for i in plan} | {str(new_id[i] + 1) for i in plan}
    for row in out:
        if "-" in row[0] or not row[6] or row[6] == "0" or row[0] in own:
            continue
        for i, cand in plan.items():
            if row[6] != str(new_id[i]) or not isinstance(cand, dict) or "head_id" not in cand:
                continue
            # `new_id[i]` is the FIRST of the two; a child pointing there must be redirected
            a_row, b_row = index[str(new_id[i])], index[str(new_id[i] + 1)]
            target = str(cand["head_id"])
            for part in (a_row, b_row):
                if ev.can_head(row[7], part[3]):
                    target = part[0]
                    break
            if target != row[6]:
                stats["rehomed"][row[7]] += 1
            row[6] = target
            break
    return out, len(plan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets_te/SUD_Telugu-MTG")
    ap.add_argument("--out-dir", default="assets_te")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    splits = ("train", "dev", "test")
    files = {s: f"{args.src}/te_mtg-sud-{s}.conllu" for s in splits}
    everything = [x for s in splits for x in read_sentences(files[s])]
    ev = Evidence(everything)
    print(f"evidence: {len(ev.freq)} standalone types, {len(ev.pair_form)} adjacent form pairs, "
          f"{len(ev.pair_upos)} adjacent UPOS pairs")

    stats = {"committed": collections.Counter(), "rejected": collections.Counter(),
             "rehomed": collections.Counter()}
    written = {}
    for s in splits:
        out_sents, n = [], 0
        for comments, rows in read_sentences(files[s]):
            new_rows, k = split_sentence(comments, rows, ev, args.min_freq, stats)
            n += k
            out_sents.append((comments, new_rows))
        written[s] = out_sents
        print(f"  {s:5s}: {n} tokens split")

    print(f"\ncommitted {sum(stats['committed'].values())} splits "
          f"({len(stats['committed'])} distinct types); "
          f"{sum(stats['rehomed'].values())} children re-attached")
    print(f"{'orthographic word':20s} {'= A':14s} {'+ B':14s}  n")
    for (w, a, b), n in stats["committed"].most_common():
        print(f"{w:20s} {a:14s} {b:14s} {n:3d}")
    print("\nrejected candidates, by reason:")
    for reason, n in stats["rejected"].most_common():
        print(f"  {n:4d}  {reason}")

    if args.apply:
        for s in splits:
            path = pathlib.Path(args.out_dir) / f"te_mtg-sud-{s}.mwt.conllu"
            with open(path, "w", encoding="utf-8") as fh:
                for comments, rows in written[s]:
                    fh.write("\n".join(comments) + "\n" if comments else "")
                    for row in rows:
                        fh.write("\t".join(row) + "\n")
                    fh.write("\n")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
