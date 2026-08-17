"""Shared bits for the two analyser probes: key variants and the output format."""
import csv

def key_variants(w):
    """Both analysers store the PRE-PAUSAL word: vidyut's docs say visarga-final entries end in
    `s` or `r` (rAmaH is stored as rAmas), and Inria's XML is the same shape. So a padapāṭha form
    ending in visarga must be probed as itself AND as its two underlying finals, or every
    visarga-final noun in the corpus reads as unknown."""
    out = [w]
    if w.endswith("H"):
        out += [w[:-1] + "s", w[:-1] + "r"]
    if w.endswith("M"):
        out += [w[:-1] + "m"]
    return out

def read_keys(path):
    keys = set()
    with open(path) as f:
        for surf, pada, lem, morph, comp in csv.reader(f, delimiter="\t"):
            keys.add(pada)
    return sorted(keys)

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "found", "lemmas", "Case", "Number", "Gender", "Person"])
        w.writerows(rows)
