"""
parse_studio_table.py
=====================
Makes sense of a table copied out of Classiq Studio's Execute tab, then tells you
whether the circuit you ran can actually support the claim you want to make.

Studio renders the results grid as `#  counts  probability  bitstring`. When the
table is copy-pasted it often arrives as one long line with the column
boundaries lost, which is what this script handles: it finds the bitstrings
(tokens that are pure 0/1 and all the same length), then works out what the run
was, and what it can and cannot prove.

Usage
-----
    python3 code/parse_studio_table.py results/studio_counts/paste.txt
    python3 code/parse_studio_table.py paste.txt --circuit bb84_batch
    python3 code/parse_studio_table.py paste.txt --json          # save counts

It also accepts a proper whitespace/tab separated table (*.tsv, *.txt) or a
.json/.csv export -- in that case the counts are recovered exactly and the
statistics are computed.

No third-party imports.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

# the fixed example pattern used by qasm/bb84_batch.qasm (numpy default_rng(7))
BB84_BITS = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
BB84_BASES = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]


def known_patterns():
    """Circuits whose bit pattern is fixed by construction."""
    pats = {}
    if len(BB84_BITS) == 16:
        pats["bb84_batch"] = (BB84_BITS, BB84_BASES)
    return pats


# ---------------------------------------------------------------------------
def extract_bitstrings(text: str):
    """All pure-binary tokens; keeps the most common length."""
    toks = re.findall(r"(?<![01.])[01]{4,}(?![01])", text)
    if not toks:
        return []
    lens = collections.Counter(len(t) for t in toks)
    L = lens.most_common(1)[0][0]
    return [t for t in toks if len(t) == L]


def extract_counts(text: str, bitstrings, shots_hint=None):
    """Best-effort: for each bitstring, the integer that follows it."""
    counts = {}
    for m in re.finditer(r"(?<![01])([01]{4,})(?![01])", text):
        after = text[m.end(): m.end() + 12]
        n = re.match(r"\s*[^\d]{0,3}(\d{1,6})(?![\d.])", after)
        if n:
            v = int(n.group(1))
            counts[m.group(1)] = counts.get(m.group(1), 0) + v
    return counts


def parse_tabular(text: str):
    """Parse a properly separated table: header + rows."""
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if len(rows) < 3:
        return None
    hdr = [c.strip().lower() for c in re.split(r"[\t,;|]+|\s{2,}", rows[0])]
    if "bitstring" not in hdr:
        return None
    bi = hdr.index("bitstring")
    ci = next((hdr.index(c) for c in ("counts", "count", "frequency") if c in hdr), None)
    if ci is None:
        return None
    ri = next((hdr.index(c) for c in ("result", "decimal") if c in hdr), None)
    out, dec_ok, dec_bad = {}, 0, 0
    for ln in rows[1:]:
        f = re.split(r"[\t,;|]+|\s{2,}", ln.strip())
        if len(f) <= max(bi, ci):
            continue
        try:
            out[f[bi].strip()] = out.get(f[bi].strip(), 0) + int(float(f[ci]))
        except ValueError:
            continue
        if ri is not None and ri < len(f):
            try:
                v = int(float(f[ri]))
                if v < 2 ** len(f[bi].strip()):
                    if v == int(f[bi].strip(), 2):
                        dec_ok += 1
                    else:
                        dec_bad += 1
            except ValueError:
                pass
    if dec_ok or dec_bad:
        print(f"`result` column == int(bitstring, 2): {dec_ok} rows confirm, {dec_bad} do not")
    return out or None


# ---------------------------------------------------------------------------
def diagnose(bitstrings, counts, circuit=None):
    n = len(bitstrings[0])
    distinct = len(set(bitstrings))
    rep = {"bitstring_length": n, "rows_seen": len(bitstrings),
           "distinct_outcomes": distinct, "counts_recovered": bool(counts)}
    if counts:
        rep["shots"] = sum(counts.values())

    pat = known_patterns().get(circuit) if circuit else None
    if pat is None and n == 16:
        # only assume the batched-BB84 pattern if it actually explains the data;
        # a qrng_rotation run on 16 qubits is fully random and must not be
        # mistaken for the fixed-pattern circuit
        cand = known_patterns().get("bb84_batch")
        if cand:
            bits_c, bases_c = cand
            det_c = {q: bits_c[q] for q in range(16) if bases_c[q] == 0}
            agree = max(sum(1 for r in bitstrings
                            if all(int(ch) == det_c[q_of(j)]
                                   for j, ch in enumerate(r) if q_of(j) in det_c))
                        for q_of in (lambda j: 15 - j, lambda j: j))
            pat = cand if agree >= 0.8 * len(bitstrings) else None

    if pat and n == len(pat[0]):
        bits, bases = pat
        det = {q: bits[q] for q in range(n) if bases[q] == 0}
        rand = [q for q in range(n) if bases[q] == 1]
        rep["fixed_pattern_qubits_random"] = len(rand)
        rep["fixed_pattern_expected_outcomes"] = 2 ** len(rand)
        # check in the convention that actually fits the data
        best = None
        for label, q_of in (("big-endian", lambda j: n - 1 - j),
                            ("little-endian", lambda j: j)):
            ok = sum(1 for r in bitstrings
                     if all(int(ch) == det[q_of(j)]
                            for j, ch in enumerate(r) if q_of(j) in det))
            if best is None or ok > best[1]:
                best = (label, ok)
        rep["pattern_match"] = {"convention": best[0],
                                "strings_agreeing": best[1],
                                "strings_seen": len(bitstrings)}
        rep["verdict"] = ("fixed-pattern circuit: only %d of %d bits are random, so this run "
                          "cannot evidence entropy" % (len(rand), n))
    elif distinct >= 0.5 * len(bitstrings):
        rep["verdict"] = ("high-diversity output (%d distinct in %d rows) -- consistent "
                          "with an all-random register" % (distinct, len(bitstrings)))
    return rep


def completeness(rows_seen, n_outcomes_expected):
    """A partial table (rows missing) cannot support a rate/entropy statistic."""
    if n_outcomes_expected is None:
        return None
    return rows_seen >= n_outcomes_expected


def uniformity(counts, n_outcomes_expected):
    """Chi-square against a uniform distribution over n_outcomes_expected bins."""
    import math as _m
    obs = list(counts.values())
    n = sum(obs)
    if n == 0:
        return None
    exp = n / n_outcomes_expected
    chi2 = sum((o - exp) ** 2 / exp for o in obs)
    dof = max(len(obs) - 1, 1)
    # Wilson-Hilferty normal approximation to the chi-square survival function
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / _m.sqrt(2 / (9 * dof))
    p = 0.5 * _m.erfc(z / _m.sqrt(2))
    return {"chi2": round(chi2, 2), "dof": dof,
            "p_uniform": round(p, 4),
            "mean_count": round(n / len(obs), 2),
            "expected_if_all_bins": round(exp, 2)}


def stats(bitstrings, counts):
    """Expand to bits and report P(1), bias, min-entropy, monobit p."""
    if not counts:
        return None
    bits = []
    for s, c in counts.items():
        for _ in range(int(c)):
            bits.extend(int(ch) for ch in s)
    N = len(bits)
    if N == 0:
        return None
    ones = sum(bits)
    p1 = ones / N
    s = abs(2 * ones - N) / math.sqrt(N)
    p_max = max(p1, 1 - p1)
    pu = min(1.0, p_max + 2.576 * math.sqrt(p_max * (1 - p_max) / max(N - 1, 1)))
    return {"n_bits": N, "P_1": round(p1, 6), "bias": round(abs(p1 - 0.5), 6),
            "min_entropy_mcv": round(-math.log2(pu), 5) if pu < 1 else 0.0,
            "monobit_p_value": round(math.erfc(s / math.sqrt(2)), 4),
            "sigma_on_P1": round(math.sqrt(0.25 / N), 6)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--circuit", default=None,
                    help="bb84_batch | qrng_rotation | qrng_ghz | chsh")
    ap.add_argument("--json", action="store_true", help="write a parsed JSON next to the input")
    ap.add_argument("--shots", type=int, default=None,
                    help="Num Shots used in Studio; checks the distinct-outcome count "
                         "against the ideal uniform prediction (a strong correctness test)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"no such file: {args.file}")
    text = open(args.file, errors="replace").read()

    counts = parse_tabular(text)
    bitstrings = extract_bitstrings(text)
    if counts:
        bitstrings = list(counts)
        print(f"parsed a proper table: {len(counts)} outcomes, "
              f"{sum(counts.values())} shots")
    else:
        print(f"table looks copy-pasted without separators: "
              f"{len(bitstrings)} bitstrings recovered, counts NOT recoverable")
        c = extract_counts(text, bitstrings)
        if c and (sum(c.values()) > 0):
            print(f"best-effort counts: {len(c)} outcomes, sum={sum(c.values())} "
                  f"(treat as unreliable unless it equals your Num Shots)")

    if not bitstrings:
        sys.exit("no bitstrings found -- is this the results table?")


    rep = diagnose(bitstrings, counts if counts else None, args.circuit)
    st = None
    if counts:
        exp_out = rep.get("fixed_pattern_expected_outcomes")
        complete = completeness(len(counts), exp_out)
        rep["table_complete"] = complete
        if complete is False:
            rep["statistics_suppressed"] = (
                f"only {len(counts)} of {exp_out} outcomes present -- this is a partial "
                f"table (Studio sorts by counts, so it is the tail), and P(1)/entropy "
                f"computed on it would be badly biased. Paste all {exp_out} rows or "
                f"download the CSV.")
        else:
            st = stats(bitstrings, counts)
            if st:
                rep["statistics"] = st
            u = uniformity(counts, exp_out) if exp_out else None
            if u:
                rep["uniformity"] = u

    print()
    for k, v in rep.items():
        if k == "statistics":
            print("\nstatistics (from recovered counts):")
            for kk, vv in v.items():
                print(f"  {kk:18s} {vv}")
        elif isinstance(v, dict):
            print(f"{k}: " + ", ".join(f"{a}={b}" for a, b in v.items()))
        else:
            print(f"{k}: {v}")

    if args.shots:
        bins = 2 ** rep["bitstring_length"]
        seen = rep["distinct_outcomes"]
        expected = bins * (1 - (1 - 1 / bins) ** args.shots)
        dev = 100 * (seen - expected) / expected
        print(f"\ndistinct-outcome check (shots={args.shots}): "
              f"observed {seen}, ideal-uniform prediction {expected:.0f} "
              f"({dev:+.2f} %)")
        if seen < 0.5 * expected:
            print("  -> table is an excerpt (few rows), so this check cannot be applied; "
                  "use the full table or the CSV")
        elif abs(dev) < 2:
            print("  -> consistent with an unbiased uniform sampler")
        else:
            print("  -> NOT what a uniform sampler predicts: check the shot count, "
                  "the circuit, and whether all rows are present")

    if args.json:
        out = os.path.splitext(args.file)[0] + "_parsed.json"
        with open(out, "w") as f:
            json.dump({"report": rep,
                       "counts": counts or None}, f, indent=2)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
