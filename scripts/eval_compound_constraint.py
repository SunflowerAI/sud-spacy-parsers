#!/usr/bin/env python3
"""Does forbidding out-of-compound heads for non-final compound members improve the parse?

`check_sa_compound_signal.py` answers whether the rule is TRUE. This answers whether enforcing it
PAYS, which is a different question: forbidding an arc does not say where the token should go
instead, and the re-decode may attach it somewhere equally wrong.

Reported per condition: UAS/LAS over all tokens, plus the constraint's own hit rate — how often it
fired, and how often the token it fired on ended up with the RIGHT head afterwards.

⚠ ONLY VIOLATING SENTENCES ARE RE-DECODED, AND THAT IS EXACT, NOT AN APPROXIMATION. ArcEager arcs
are final once taken, so a banned action that the greedy decoder would have chosen shows up as a
violating arc in the baseline tree. A sentence whose baseline parse violates nothing therefore never
had the mask bind, and its constrained parse is the same tree. `--check N` re-decodes N of those
anyway and reports any disagreement, which also catches this module's Python greedy loop drifting
from spaCy's own.

    eval_compound_constraint.py MODEL TEST.spacy [--allow flat] [--strict-span] [--limit N]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from sud_constrained_parse import _decode, compound_spans, parse_with_compound  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test")
    ap.add_argument("--allow", default="flat",
                    help="comma-separated relations the ban exempts; '' bans every label")
    ap.add_argument("--strict-span", action="store_true",
                    help="use the bare Compound=Yes run, not the orthographic word")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=50,
                    help="non-firing sentences to re-decode as a sanity check")
    ap.add_argument("--control", action="store_true",
                    help="also re-decode every sentence with the mask disabled, so the constraint "
                         "is scored against THIS decoder rather than against spaCy's own")
    a = ap.parse_args()
    allowed = frozenset(x for x in a.allow.split(",") if x)
    extend = not a.strict_span

    nlp = spacy.load(a.model)
    parser = nlp.get_pipe("parser")
    blank = spacy.blank("sa")
    refs = list(DocBin().from_disk(a.test).get_docs(blank.vocab))
    sents = [s for d in refs for s in d.sents]
    if a.limit:
        sents = sents[: a.limit]

    def prep(s):
        """The doc the parser sees in DEPLOYMENT: predicted tag/morph/lemma with the tokeniser's own
        `Compound` re-imposed over the morphologiser's, which is what `clause_parser` does."""
        d = Doc(nlp.vocab, words=[t.text for t in s], spaces=[bool(t.whitespace_) for t in s])
        for pt, rt in zip(d, s):
            pt.norm_ = rt.norm_
            if rt.morph.get("Compound"):
                pt.set_morph("Compound=Yes")
        for name, pipe in nlp.pipeline:
            if name == "parser":
                break
            d = pipe(d)
        for pt, rt in zip(d, s):
            if rt.morph.get("Compound") and not pt.morph.get("Compound"):
                pt.set_morph((str(pt.morph) + "|Compound=Yes").lstrip("|"))
        return d

    tot = 0
    base = [0, 0]
    ctl = [0, 0]
    con = [0, 0]
    fired = fixed = broke = same = failed = 0
    members = residual = rooted = base_rooted = 0
    checked = disagreed = 0
    for s in sents:
        gold = [(t.head.i - s.start, t.dep_) for t in s]
        d = prep(s)
        b = parser(d.copy())
        spans = compound_spans(b, extend_to_word=extend)
        hits = [i for i, sp in enumerate(spans)
                if sp is not None and not (sp[0] <= b[i].head.i <= sp[1])
                and b[i].dep_.split("||")[0] not in allowed]
        c = b
        if a.control:
            # The SAME Python greedy loop with nothing masked. spaCy parses in C off precomputed
            # weights, so the two decoders can part company on a near-tie; scoring the constraint
            # against spaCy's tree would then charge it for that drift as well as for the mask.
            u = _decode(parser, d, lambda s0, b0: (False, False))
            if u is not None:
                for i in range(len(s)):
                    ctl[0] += u[i].head.i == gold[i][0]
                    ctl[1] += u[i].head.i == gold[i][0] and u[i].dep_ == gold[i][1]
        if a.control or hits or checked < a.check:
            o = parse_with_compound(parser, d, allowed=allowed, extend_to_word=extend)
            if o is None:
                failed += 1
            elif hits or a.control:
                # Under --control the constrained tree comes from THIS decoder on every sentence,
                # not just the firing ones, so control-vs-constrained differs only by the mask.
                c = o
            if o is not None and not hits and checked < a.check:
                checked += 1
                disagreed += any(o[i].head.i != b[i].head.i for i in range(len(b)))
        fired += len(hits)
        # Does the mask actually deliver the property? Counted on the OUTPUT, not assumed: a member
        # can still end up out of its compound by becoming a root, which is not a maskable action.
        for i, sp in enumerate(compound_spans(c, extend_to_word=extend)):
            if sp is None:
                continue
            members += 1
            if not (sp[0] <= c[i].head.i <= sp[1]) and c[i].dep_.split("||")[0] not in allowed:
                residual += 1
            rooted += c[i].head.i == i
            base_rooted += b[i].head.i == i
        for i, (gh, gl) in enumerate(gold):
            tot += 1
            base[0] += b[i].head.i == gh
            base[1] += b[i].head.i == gh and b[i].dep_ == gl
            con[0] += c[i].head.i == gh
            con[1] += c[i].head.i == gh and c[i].dep_ == gl
        for i in hits:
            was, now = b[i].head.i == gold[i][0], c[i].head.i == gold[i][0]
            fixed += (not was) and now
            broke += was and (not now)
            same += was == now

    exempt = ", ".join(sorted(allowed)) or "nothing"
    print(f"{len(sents)} sentences / {tot} tokens; span "
          f"{'Compound run' if a.strict_span else 'orthographic word'}; exempting {exempt}")
    print(f"  constraint fired on {fired} arcs; {failed} decodes failed")
    print(f"  baseline    UAS {base[0]/tot:.4f}  LAS {base[1]/tot:.4f}   (spaCy's own C decode)")
    if a.control:
        print(f"  control     UAS {ctl[0]/tot:.4f}  LAS {ctl[1]/tot:.4f}"
              f"   (this decoder, mask disabled: {100*(ctl[0]-base[0])/tot:+.3f} UAS)")
    print(f"  constrained UAS {con[0]/tot:.4f}  LAS {con[1]/tot:.4f}"
          f"   ({100*(con[0]-base[0])/tot:+.3f} UAS, {100*(con[1]-base[1])/tot:+.3f} LAS)")
    print(f"  on the arcs it fired on: fixed {fixed}, broke {broke}, unchanged {same}")
    print(f"  property check: {residual} of {members} non-final members still take a head outside "
          f"their compound")
    print(f"  escape hatch: {rooted} take NO head at all — they come out as roots, which is not an "
          f"action and so cannot be masked (baseline {base_rooted})")
    print(f"  sanity: {checked} non-firing sentences re-decoded, {disagreed} disagreed with greedy")


if __name__ == "__main__":
    main()
