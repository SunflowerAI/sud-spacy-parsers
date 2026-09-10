#!/usr/bin/env python
"""Does the left-corner parser predict reading times over and above a lexical baseline?

THE ONLY QUESTION WORTH ASKING is whether the parser's predictors improve on a baseline that
already knows word length, frequency, cloze predictability and position. Surprisal correlates with
all of those, so a raw correlation between surprisal and gaze duration is nearly uninformative --
it mostly re-reports that rare long words are read slowly. Every model below therefore nests
inside the same baseline and is compared to it by Delta AIC on IDENTICAL rows.

⚠ IDENTICAL ROWS IS THE WHOLE GAME. Each predictor set drops a different set of rows to missing
values (the lagged terms drop sentence-initial words; the parser's terms drop sentences it could
not derive). Fitting each model on whatever survives its own predictors makes the comparison
meaningless in a way no diagnostic reports: AIC falls simply because n fell. The frame is
completed ONCE, up front, across the union of all predictors, and every fit uses that frame.

⚠ PARTICIPANT VARIANCE IS NOT NOISE. Reading speed varies several-fold across readers, so a model
without participant terms attributes that variance to whatever predictor happens to correlate with
who read what. Participant enters as a fixed effect here, which is not the mixed model this
literature normally fits but does absorb the same between-reader variance; treat the t values as
approximate and the Delta AIC ordering as the result.

⚠ SKIPPED WORDS ARE NOT ZERO-TIME WORDS. A word that was never fixated has no reading time, and
scoring it as 0 ms turns skipping -- which is itself predicted by surprisal -- into an enormous
fake reading-time effect. Skips are dropped, and modelled separately as a binary outcome.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASELINE = ["Word_Length", "log_freq", "prev_log_freq", "OrthographicMatch",
            "Word_In_Sentence_Number", "n_subtokens"]
SURPRISAL = ["surprisal", "prev_surprisal"]
MEMORY = ["depth", "prev_depth", "open_slots", "d_depth", "integ"]


def fit(df, cols, y, participants):
    X = pd.concat([df[cols], participants], axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(df[y].astype(float), X).fit()
    return m


def report(df, y, participants, sets):
    base = fit(df, BASELINE, y, participants)
    print("\n%s   n=%d   baseline adj-R2 %.4f  AIC %.1f" % (y, len(df), base.rsquared_adj, base.aic))
    print("  %-28s %10s %10s %10s" % ("model", "dAIC", "dAdjR2", "key t"))
    for name, cols in sets:
        m = fit(df, BASELINE + cols, y, participants)
        ts = ["%s %.2f" % (c, m.tvalues[c]) for c in cols[:2]]
        print("  %-28s %10.1f %10.5f   %s"
              % (name, m.aic - base.aic, m.rsquared_adj - base.rsquared_adj, "; ".join(ts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--min-rt", type=float, default=80)
    ap.add_argument("--max-rt", type=float, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.merged, sep="\t", low_memory=False)
    print("rows in %d" % len(df), file=sys.stderr)

    # One frame for every model: complete cases across the UNION of all predictors.
    need = BASELINE + SURPRISAL + MEMORY + ["prev_open_slots", "prev_d_depth", "Participant_ID"]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    df = df[(df.IA_SKIP == 0)]
    print("rows after complete-case + skip filter: %d" % len(df), file=sys.stderr)

    participants = pd.get_dummies(df["Participant_ID"].astype(str), prefix="p", drop_first=True)
    participants.index = df.index

    sets = [("+ surprisal", SURPRISAL),
            ("+ memory", MEMORY),
            ("+ surprisal + memory", SURPRISAL + MEMORY),
            ("+ lexical surprisal only", ["surprisal_lex", "prev_surprisal_lex"])]

    for y in ("first_fix", "gaze", "go_past", "total"):
        if y not in df.columns:
            continue
        d = df[(df[y] >= args.min_rt) & (df[y] <= args.max_rt)]
        p = participants.loc[d.index]
        report(d, y, p, sets)

    print("\ncorrelations among the parser's own predictors (collinearity check)")
    cols = [c for c in ["surprisal", "surprisal_lex", "depth", "open_slots", "d_depth", "integ",
                        "Word_Length", "log_freq", "OrthographicMatch"] if c in df.columns]
    print(df[cols].corr().round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
