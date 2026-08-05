"""Where `Shared` can occur, and the backoff key it is decided on. Shared by every user of it.

`Shared=Yes|No` says whether a dependent of a conjunct is shared with the other conjuncts:

    identifying and breaking up terror cells
      `up`    compound@prt of `breaking`   Shared=No    -- belongs to the second conjunct alone
      `cells` comp:obj    of `breaking`    Shared=Yes   -- the object of both

So it is a fact about a COORDINATION, and it is only defined inside one. Three users need to agree
on exactly where that is -- the harvester that derives the rule table, the rule component, and the
evaluation -- so the definition lives here and nowhere else, in the manner of `sud_reported_data`.

THE CANDIDATE MASK. A token is a candidate iff

  1. its head belongs to a coordination -- i.e. the head is a conjunct: either it carries a `conj`
     relation itself or something carries one to it (SUD CHAINS conjuncts, each to the previous,
     so the group is recovered by walking the chain up to the first conjunct);
  2. its own relation is neither `cc` (the coordinator is not a dependent that could be shared)
     nor `conj` (a further conjunct is a member, not a dependent of one); and
  3. it lies OUTSIDE the span between the first and last conjunct. This is the part that is easy
     to miss and does most of the work: a dependent sitting between two conjuncts is inside the
     territory of its own conjunct and cannot be shared, and SUD does not mark it. `his` in
     `... his wife and X` gets no `Shared` at all; `terror cells`, which follows the last
     conjunct, does. Measured on SUD_English-EWT train, the mask reaches 92.9 % of the tokens
     that carry `Shared` while cutting the field from 30 168 tokens to 15 499.

The mask is a RECALL device, not a rule: 39 % of what it admits carries no `Shared` in the gold,
so whatever consumes it still has to decide presence as well as value. Its job is to say where the
question is even asked, which is what stops a component emitting the feature across the whole
sentence -- the failure the morphologiser makes today.

POSITION is carried through as `before`/`after` (relative to the first conjunct) because it is the
single most informative cue for the value: material to the LEFT of a coordination is usually shared
by all conjuncts (a subject, a preposed adjunct), while material to the right is more often local
to the last one.
"""

# Relations that are structurally part of the coordination rather than a dependent of a conjunct.
_NOT_A_DEPENDENT = ("cc",)


def _base(deprel):
    """The relation without its `@`-suffixed deep feature (`comp:obj@x` -> `comp:obj`)."""
    return deprel.split("@", 1)[0]


def _is_conj(deprel):
    return _base(deprel).split(":", 1)[0] == "conj"


def coordination_of(heads, deprels):
    """Map each token index to the sorted conjunct indices of the coordination it HEADS, if any.

    `heads` is 0-based: `heads[i]` is the index of i's head, and a root points at itself. Returns
    a dict from a conjunct's index to the tuple of every index in its group, so a lookup on
    `heads[i]` answers "is my head a conjunct, and if so who are its fellows".
    """
    groups = {}
    for i, dep in enumerate(deprels):
        if not _is_conj(dep):
            continue
        # Walk up the chain to the first conjunct. `seen` guards against a cycle, which a
        # PREDICTED parse can produce even though a gold tree cannot.
        cur, seen = i, {i}
        while _is_conj(deprels[cur]):
            nxt = heads[cur]
            if nxt == cur or nxt in seen:
                break
            cur = nxt
            seen.add(cur)
        groups.setdefault(cur, set()).update({cur, i})
    members = {}
    for first, ids in groups.items():
        ordered = tuple(sorted(ids))
        for i in ids:
            members[i] = ordered
    return members


def candidates(heads, deprels):
    """Yield `(i, position)` for every token index where `Shared` is a live question.

    `position` is "before" or "after", relative to the first conjunct of the coordination.
    """
    members = coordination_of(heads, deprels)
    for i, dep in enumerate(deprels):
        group = members.get(heads[i])
        if not group or heads[i] == i:
            continue
        base = _base(dep)
        if base in _NOT_A_DEPENDENT or _is_conj(dep):
            continue
        if group[0] <= i <= group[-1]:
            continue                      # inside the coordination's own span -- see the docstring
        yield i, ("before" if i < group[0] else "after")


def doc_candidates(doc):
    """`candidates` over a spaCy `Doc`, using the parser's heads and relations."""
    heads = [t.head.i for t in doc]
    deprels = [t.dep_ for t in doc]
    return candidates(heads, deprels)


def backoff_keys(deprel, head_pos, position):
    """The lookup ladder the rule table is keyed on, most specific first.

    An exact key alone is too brittle -- the deprel and the head's UPOS are both PREDICTED at
    inference, so one mis-tag would turn into a total miss. Same reasoning as the la macroniser's
    nine-slot ladder.
    """
    base = _base(deprel)
    return (
        f"{base}\t{head_pos}\t{position}",
        f"{base}\t{position}",
        f"{position}",
    )
