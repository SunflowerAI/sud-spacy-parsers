#!/usr/bin/env python
"""The left-corner dependency transition system of Noji & Miyao (COLING 2014,
`aclanthology.org/C14-1202`), plus its static oracle -- the substrate for the incremental English
parser used for psycholinguistic modelling.

WHY THIS SYSTEM AND NOT spaCy's. spaCy's parser is arc-eager, and arc-eager stack depth CANNOT
predict the difficulty of centre-embedding: its cost never increases as long as the tokens on the
stack form one connected component, which is exactly the situation a centre-embedded clause
creates. Left-corner is the one strategy known to get that contrast right (Abney & Johnson 1991;
Resnik 1992), and Noji & Miyao carry it over to dependencies. Nothing in spaCy's config surface
reaches this -- the transition system is Cython and has no left-corner option -- so this is a
standalone module over CoNLL-U, in the spirit of `sud_arcfactored_parser.py`'s hand-rolled decoder.

CONFIGURATION. c = (sigma, beta, A). Each stack element is a RIGHT SPINE: the descending path from
a subtree's root taking the rightmost child at each step. Nodes off that path are not lost, they
just live in A. A spine element is a token index, or the single DUMMY node x(lam), where lam holds
the roots of subtrees already known to be left dependents of the token that will eventually fill x.
A spine is INCOMPLETE iff it carries a dummy. The dummy, when present, is always the LAST node of
the spine -- every rule that creates or keeps one puts it there -- which is what makes this
representation a flat list rather than a tree.

THE SIX ACTIONS (Figure 2 of the paper), with i = the node above the dummy, if any:

    SHIFT       push <j>                                                      buffer j consumed
    INSERT      <s|i|x(lam)>          -> <s|i|j>        + (i,j) + (j,k) k in lam
    LEFT-PRED   <r, ...>              -> <x({r})>
    RIGHT-PRED  <r, ...>              -> <r, x({})>
    LEFT-COMP   <s|x(lam)> <r, ...>   -> <s|x(lam + {r})>
    RIGHT-COMP  <s|i|x(lam)> <r, ...> -> <s|i|r|x({})>  + (i,r) + (r,k) k in lam

Shift-type and reduce-type actions STRICTLY ALTERNATE, starting and ending with a shift-type one, so
a sentence of n tokens (the artificial root included) has exactly n shift-type and n-1 reduce-type
actions. Two invariants follow and are asserted below: #SHIFT = #COMP + 1 and #INSERT = #PRED.

TWO THINGS THE PAPER'S PRINTED RULES LEAVE IMPLICIT, both of which are load-bearing and both of
which the round-trip test catches immediately if got wrong:

  1. RIGHT-COMP ADDS THE ARC (i, r). Figure 2's right-hand side lists only the arcs from r to lam,
     but the oracle condition for the same action reads "(i, sigma_11) in A_g or ...", so the arc
     from i is required to be gold and must therefore be emitted. Without it every subtree attached
     by composition comes out headless.
  2. RIGHT-PRED TAKES *AT LEAST* ONE REMAINING DEPENDENT, RIGHT-COMP *EXACTLY* ONE. The paper writes
     "has one more dependent in beta" for both, and reading both as "at least one" deadlocks on the
     first head that has two right dependents and is not the root of its own stack element. The
     asymmetry is forced by where each action leaves the node: RIGHT-PRED hangs the new slot off the
     spine ROOT, which stays the root and can be given another slot later; RIGHT-COMP buries r one
     level down the spine, so r never gets a second chance and must be down to its last dependent.
     The same reasoning fixes INSERT's own asymmetry, which the paper DOES state: when i exists the
     inserted token is buried and must have no dependents left, and when the dummy is the whole
     spine (so the token becomes the root) it may still have some.

COVERAGE. Projective trees only; Noji & Miyao remove non-projective sentences and so does the
driver here. `oracle` raises `Underivable` rather than returning a wrong derivation.
"""
import argparse
import collections
import sys

SHIFT, INSERT, LEFT_PRED, RIGHT_PRED, LEFT_COMP, RIGHT_COMP = (
    "SHIFT", "INSERT", "LEFT-PRED", "RIGHT-PRED", "LEFT-COMP", "RIGHT-COMP")
SHIFT_ACTIONS = (SHIFT, INSERT)
REDUCE_ACTIONS = (LEFT_PRED, RIGHT_PRED, LEFT_COMP, RIGHT_COMP)
ACTIONS = SHIFT_ACTIONS + REDUCE_ACTIONS


class Underivable(Exception):
    """The gold tree is outside the system -- non-projective, or the oracle wedged."""


class Spine:
    """A right spine. `nodes` are token indices; `lam` is None for a complete spine, otherwise the
    left dependents pending on the dummy that sits immediately after `nodes`."""

    __slots__ = ("nodes", "lam", "dummy_label")

    def __init__(self, nodes, lam=None, dummy_label=None):
        self.nodes = list(nodes)
        self.lam = None if lam is None else list(lam)   # [(root, label), ...]
        self.dummy_label = dummy_label                  # label of (above_dummy -> filler)

    @property
    def incomplete(self):
        return self.lam is not None

    @property
    def root(self):
        # The head of the subtree. `nodes` is empty only for <x(lam)> straight out of LEFT-PRED,
        # where the root IS the dummy and no token is available.
        return self.nodes[0] if self.nodes else None

    @property
    def above_dummy(self):
        """`i` in the rules: the node the dummy hangs off, or None when the dummy is the root."""
        return self.nodes[-1] if self.nodes else None

    def copy(self):
        return Spine(self.nodes, self.lam, self.dummy_label)

    def __repr__(self):
        body = ",".join(str(n) for n in self.nodes)
        if self.incomplete:
            body += ("," if body else "") + "x(%s)" % ",".join(str(k) for k, _ in self.lam)
        return "<%s>" % body


class Config:
    """(sigma, beta, A). `beta` is the index of the next unconsumed token."""

    __slots__ = ("stack", "beta", "arcs", "n", "history")

    def __init__(self, n):
        self.stack = []
        self.beta = 1
        self.arcs = {}          # dependent -> head
        self.n = n              # highest token index, the artificial root included
        self.history = []

    @property
    def cost(self):
        """Noji & Miyao's memory cost: the number of elements on the stack."""
        return len(self.stack)

    @property
    def terminal(self):
        return self.beta > self.n and len(self.stack) == 1 and not self.stack[0].incomplete

    def legal(self, action):
        want_shift = not self.history or self.history[-1][0] in REDUCE_ACTIONS
        if (action in SHIFT_ACTIONS) != want_shift:
            return False
        if action in SHIFT_ACTIONS and self.beta > self.n:
            return False
        top = self.stack[-1] if self.stack else None
        second = self.stack[-2] if len(self.stack) > 1 else None
        if action == SHIFT:
            return True
        if action == INSERT:
            return top is not None and top.incomplete
        if action in (LEFT_PRED, RIGHT_PRED):
            return top is not None and not top.incomplete
        if action in (LEFT_COMP, RIGHT_COMP):
            return (top is not None and not top.incomplete
                    and second is not None and second.incomplete)
        raise ValueError(action)

    def apply(self, action):
        action, label = action if isinstance(action, tuple) else (action, None)
        if not self.legal(action):
            raise Underivable("illegal action %s in %s" % (action, self))
        if action == SHIFT:
            self.stack.append(Spine([self.beta]))
            self.beta += 1
        elif action == INSERT:
            top = self.stack[-1]
            j, i, lam = self.beta, top.above_dummy, top.lam
            if i is not None:
                self._add_arc(i, j, top.dummy_label)
            for k, kl in lam:
                self._add_arc(j, k, kl)
            top.nodes.append(j)
            top.lam = None
            top.dummy_label = None
            self.beta += 1
        elif action == LEFT_PRED:
            top = self.stack[-1]
            self.stack[-1] = Spine([], [(top.root, label)])
        elif action == RIGHT_PRED:
            top = self.stack[-1]
            self.stack[-1] = Spine([top.root], [], label)
        elif action == LEFT_COMP:
            top = self.stack.pop()
            self.stack[-1].lam.append((top.root, label))
        elif action == RIGHT_COMP:
            top = self.stack.pop()
            second = self.stack[-1]
            r, i, lam = top.root, second.above_dummy, second.lam
            if i is not None:
                self._add_arc(i, r, second.dummy_label)
            for k, kl in lam:
                self._add_arc(r, k, kl)
            second.nodes.append(r)
            second.lam = []
            second.dummy_label = label
        self.history.append((action, label))
        return self

    def _add_arc(self, head, dep, label):
        if dep in self.arcs:
            raise Underivable("token %d given a second head" % dep)
        self.arcs[dep] = (head, label)

    def copy(self):
        """A fresh configuration sharing nothing mutable -- one beam hypothesis per copy."""
        c = Config.__new__(Config)
        c.stack = [s.copy() for s in self.stack]
        c.beta, c.n = self.beta, self.n
        c.arcs = dict(self.arcs)
        c.history = list(self.history)
        return c

    @property
    def open_slots(self):
        """Predictions the parser is currently holding: one per incomplete spine, plus every
        subtree already parked as a left dependent of a node it has not yet seen. This is the
        left-corner analogue of an open-node count, and unlike `cost` it grows with unresolved
        LEFT-PREDs rather than only with stack elements."""
        return sum(1 + len(s.lam) for s in self.stack if s.incomplete)

    def __repr__(self):
        return "Config(sigma=[%s], beta=%d, cost=%d)" % (
            " ".join(repr(s) for s in self.stack), self.beta, self.cost)


def replay(actions, n):
    """Run an action sequence and return {dependent: (head, label)}. The inverse of `oracle`."""
    c = Config(n)
    for a in actions:
        c.apply(a)
    if not c.terminal:
        raise Underivable("action sequence ended in a non-terminal configuration: %r" % c)
    return c.arcs


def oracle(heads, n, labels=None):
    """Static oracle. `heads[d] = h` over 1..n, where n is the artificial root (which has no head).

    Returns (actions, costs). Each action is a (name, label) pair: shift-type actions carry None,
    and each reduce action carries the label of the ARC IT COMMITS TO. Every arc is labelled exactly
    once, at the moment a slot is predicted for its dependent rather than when the two ends finally
    meet -- LEFT-PRED and LEFT-COMP label the subtree they park as a left dependent, RIGHT-PRED
    labels the spine root's next right dependent, and RIGHT-COMP labels the one right dependent its
    own subtree has left. That is the property that makes the model usable for reading-time work:
    a grammatical function is committed as soon as the slot is opened, not retroactively.

    costs[t] is the memory cost of the configuration reached after action t. Raises `Underivable`
    on a tree the system cannot derive."""
    deps = collections.defaultdict(list)
    for d in range(1, n + 1):
        if d in heads:
            deps[heads[d]].append(d)
    for h in deps:
        deps[h].sort()

    def pending(v, beta):
        """v's gold dependents that are still in the buffer, in order."""
        return [d for d in deps.get(v, ()) if d >= beta]

    labels = labels or {}
    c = Config(n)
    costs = []
    while not c.terminal:
        want_shift = not c.history or c.history[-1][0] in REDUCE_ACTIONS
        if want_shift:
            action = (_shift_oracle(c, heads, pending), None)
        else:
            name = _reduce_oracle(c, heads, pending)
            action = (name, _reduce_label(c, name, heads, labels, pending))
        c.apply(action)
        costs.append(c.cost)
        if len(c.history) > 4 * n + 4:
            raise Underivable("oracle did not terminate")
    return c.history, costs


def _reduce_label(c, name, heads, labels, pending):
    """The label the reduce action commits to. Reads only configuration state and the gold, so it
    never second-guesses `_reduce_oracle`'s choice."""
    r = c.stack[-1].root
    if name in (LEFT_PRED, LEFT_COMP):
        return labels.get(r)
    slot_for = pending(r, c.beta)          # RIGHT-PRED: next right dependent; RIGHT-COMP: the last
    return labels.get(slot_for[0]) if slot_for else None


def _shift_oracle(c, heads, pending):
    if c.beta > c.n:
        raise Underivable("a shift is due but the buffer is empty")
    j = c.beta
    top = c.stack[-1] if c.stack else None
    if top is not None and top.incomplete:
        i = top.above_dummy
        if i is not None:
            # j lands buried in the spine, so it must be i's dependent and be finished with the
            # buffer; any left dependents already parked on the dummy must be j's.
            if heads.get(j) == i and not pending(j, j + 1) and all(heads.get(k) == j for k, _ in top.lam):
                return INSERT
        else:
            # j becomes the root of the spine, so it may still collect right dependents later.
            if top.lam and all(heads.get(k) == j for k, _ in top.lam):
                return INSERT
    return SHIFT


def _reduce_oracle(c, heads, pending):
    top = c.stack[-1]
    second = c.stack[-2] if len(c.stack) > 1 else None
    r = top.root
    r_pending = pending(r, c.beta)

    if second is not None and second.incomplete:
        i, lam = second.above_dummy, second.lam
        # Whoever fills the dummy: i's next still-unattached dependent, or the shared head of the
        # left dependents already parked there.
        if i is not None:
            filler_deps = pending(i, c.beta)
            filler = filler_deps[0] if filler_deps else None
        else:
            filler = heads.get(lam[0][0]) if lam else None
        # LEFT-COMP buries r in lam, so r must be finished and headed by the filler.
        if not r_pending and filler is not None and heads.get(r) == filler:
            return LEFT_COMP
        # RIGHT-COMP puts r at the dummy's position and hangs r's LAST dependent slot off it.
        if len(r_pending) == 1 and (heads.get(r) == i or (lam and all(heads.get(k) == r for k, _ in lam))):
            return RIGHT_COMP
    if r_pending:
        return RIGHT_PRED
    return LEFT_PRED


def read_conllu(path):
    """Yield (forms, heads, labels, n) per sentence, with an artificial root appended at index n.

    The root goes at the END, as Noji & Miyao and Ballesteros & Nivre (2013) place it: a left-corner
    parser reading a final root treats the sentence's own root as a left dependent it has been
    holding, which is the analysis the memory cost is meant to reflect. Multiword-token ranges and
    empty nodes are skipped, so indices are renumbered onto the surface tokens."""
    forms, heads, labels = [], {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if forms:
                    root = len(forms) + 1
                    heads = {d: (root if h == 0 else h) for d, h in heads.items()}
                    yield forms + ["<ROOT>"], heads, labels, root
                forms, heads, labels = [], {}, {}
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            forms.append(cols[1])
            heads[len(forms)] = int(cols[6]) if cols[6] != "_" else 0
            labels[len(forms)] = cols[7]
    if forms:
        root = len(forms) + 1
        heads = {d: (root if h == 0 else h) for d, h in heads.items()}
        yield forms + ["<ROOT>"], heads, labels, root


def is_projective(heads, n):
    arcs = [(min(h, d), max(h, d)) for d, h in heads.items()]
    for a, b in arcs:
        for c, d in arcs:
            if a < c < b < d or c < a < d < b:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("conllu", nargs="+")
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--show-failures", type=int, default=3)
    args = ap.parse_args()

    total = nonproj = derived = 0
    action_counts = collections.Counter()
    cost_hist = collections.Counter()
    max_cost_hist = collections.Counter()
    failures = []
    for path in args.conllu:
        for forms, heads, labels, n in read_conllu(path):
            total += 1
            if args.max_sents and total > args.max_sents:
                total -= 1
                break
            if not is_projective(heads, n):
                nonproj += 1
                continue
            try:
                actions, costs = oracle(heads, n, labels)
                got = replay(actions, n)
                gold = {d: (h, labels.get(d)) for d, h in heads.items()}
                if got != gold:
                    raise Underivable("round-trip mismatch on %d arcs"
                                      % sum(1 for d in gold if got.get(d) != gold[d]))
                names = [a for a, _ in actions]
                n_shift = sum(names.count(a) for a in SHIFT_ACTIONS)
                assert n_shift == n, "shift count %d != %d tokens" % (n_shift, n)
                assert names.count(SHIFT) == names.count(RIGHT_COMP) + names.count(LEFT_COMP) + 1
                assert names.count(INSERT) == names.count(LEFT_PRED) + names.count(RIGHT_PRED)
            except (Underivable, AssertionError) as exc:
                if len(failures) < args.show_failures:
                    failures.append((" ".join(forms), exc))
                continue
            derived += 1
            action_counts.update(names)
            cost_hist.update(costs)
            max_cost_hist[max(costs)] += 1

    proj = total - nonproj
    print("sentences            %7d" % total)
    print("non-projective       %7d  (%5.2f %%)" % (nonproj, 100.0 * nonproj / max(total, 1)))
    print("derived + round-trip %7d  (%5.2f %% of projective)" % (derived, 100.0 * derived / max(proj, 1)))
    if failures:
        print("\nfirst failures:")
        for text, exc in failures:
            print("  %s\n    -> %s" % (text[:90], exc))
    if not derived:
        return 1
    print("\naction distribution (%d actions)" % sum(action_counts.values()))
    for a in ACTIONS:
        k = action_counts[a]
        print("  %-11s %8d  %5.2f %%" % (a, k, 100.0 * k / sum(action_counts.values())))
    print("\nmemory cost, cumulative %% of configurations")
    running, tot = 0, sum(cost_hist.values())
    for d in sorted(cost_hist):
        running += cost_hist[d]
        print("  depth <= %-2d %8.4f %%" % (d, 100.0 * running / tot))
        if running / tot > 0.99995:
            break
    print("\nper-sentence maximum cost, cumulative %%")
    running, tot = 0, sum(max_cost_hist.values())
    for d in sorted(max_cost_hist):
        running += max_cost_hist[d]
        print("  max depth <= %-2d %8.4f %%" % (d, 100.0 * running / tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
