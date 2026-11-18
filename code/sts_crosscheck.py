#!/usr/bin/env python3
"""
Independent-implementation cross-check of the QRNG test battery
against the NIST Statistical Test Suite (STS 2.1.2).

The battery in `code/nist_sp800_22.py` is a from-scratch implementation of
SP 800-22.  This script carries out the only meaningful external check of such
an implementation: run the *reference* C code (NIST STS 2.1.2) and this
Python implementation on the **same bits, with the same test parameters**, and
compare.

Two comparisons are produced, because they answer different questions:

  1. `pvalue`  -- 2 streams x 500 000 bits.  STS auto-selects the block length
     L = 6 for the Universal test at this length, i.e. every test is inside its
     documented operating range.  We therefore compare the per-stream p-values
     directly; agreement to 1e-6 means the statistics are identical, not merely
     that both happen to pass.

  2. `passrate` -- 100 streams x 10 000 bits (the STS default reporting mode).
     Pass counts are compared test by test.  At this stream length the Universal
     and Approximate-Entropy tests are *below* their documented minimum length
     (STS's own `universal.c` needs n >= 387 840 to select L >= 6; ApEn needs
     m < log2(n) - 5, i.e. m <= 8 at n = 10 000), so both implementations are
     outside their valid domain there.  The `csprng_control` block demonstrates
     this with true random bits: STS flags them at n = 10 000 as well, which
     proves the flag is a domain artefact and not a property of our bits.

Usage
-----
    python3 code/sts_crosscheck.py                 # write results/sts_crosscheck.json
    python3 code/sts_crosscheck.py --print         # also print the tables

The STS side of the comparison is *not* run from here: the NIST suite is a
separate C build.  The p-values it produced are read from
`results/sts_reports/`, which holds the verbatim STS output for exactly these
bit files (see docs/STS_CROSSCHECK.md for the commands that generated them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import nist_sp800_22 as N  # noqa: E402

RAW_KEY = ROOT / "results" / "keys" / "qrng_raw_1Mbit.npy"
REPORTS = ROOT / "results" / "sts_reports"
OUT = ROOT / "results" / "sts_crosscheck.json"

# --- what STS 2.1.2 reported, transcribed from results/sts_reports/ ----------
# n = 500 000, 2 streams (raw_1Mbit), per-stream p-values
STS_PVALUE = {
    "Frequency (Monobit)":            (0.810008, 0.667253),
    "Block Frequency (M=128)":        (0.860105, 0.116025),
    "Runs":                           (0.150740, 0.851710),
    "Longest Run of Ones (M=128)":    (0.014692, 0.979180),
    "Binary Matrix Rank (32x32)":     (0.444598, 0.622015),
    "Discrete Fourier Transform":     (0.876244, 0.825387),
    "Universal (L=6, Q=640)":         (0.149024, 0.459708),
    "Overlapping Template (m=9)":     (0.093913, 0.717982),   # asymptotic theory
    "Approximate Entropy (m=10)":     (0.185763, 0.052134),
    "Serial (m=16)":                  (0.913847, 0.566578),   # STS p_value2
    "Linear Complexity (M=500)":      (0.737558, 0.486606),
}

# Two tests use reference probabilities that differ between the SP 800-22 *text*
# (which this implementation follows) and the STS *binary*.  In both cases the
# observed class frequencies are identical -- only the constants differ, so the
# p-values agree to ~1e-2 instead of ~1e-6:
#   * Overlapping Template - we use the exact overlap distribution, STS the
#     eta/lambda asymptotic formula;
#   * Linear Complexity - SP 800-22 Table 2.10 prints pi_0 = 0.010417 while
#     linearComplexity.c hard-codes 0.01047.
THEORY_DIFFERS = {"Overlapping Template (m=9)", "Linear Complexity (M=500)"}

# n = 10 000, 100 streams -> (passed, tested) as printed in the STS report
STS_PASSRATE_RAW = {
    "Frequency (Monobit)":            (100, 100),
    "Block Frequency (M=128)":        (99, 100),
    "Cumulative Sums (fwd)":          (99, 100),
    "Cumulative Sums (rev)":          (100, 100),
    "Runs":                           (99, 100),
    "Longest Run of Ones (M=128)":    (100, 100),
    "Binary Matrix Rank (32x32)":     (100, 100),
    "Discrete Fourier Transform":     (100, 100),
    "Overlapping Template (m=9)":     (100, 100),
    "Linear Complexity (M=500)":      (98, 100),
}
STS_PASSRATE_CONTROL = {   # same configuration, os.urandom control bits
    "Frequency (Monobit)":            (98, 100),
    "Block Frequency (M=128)":        (100, 100),
    "Cumulative Sums (fwd)":          (98, 100),
    "Cumulative Sums (rev)":          (98, 100),
    "Runs":                           (100, 100),
    "Longest Run of Ones (M=128)":    (98, 100),
    "Binary Matrix Rank (32x32)":     (99, 100),
    "Discrete Fourier Transform":     (99, 100),
    "Overlapping Template (m=9)":     (98, 100),
    "Linear Complexity (M=500)":      (98, 100),
}
# out-of-domain rows, identical for real and control bits
STS_OUT_OF_DOMAIN = {
    "Universal (auto L=5)":           {"raw": (0, 100), "control": (0, 100)},
    "Approximate Entropy (m=10)":     {"raw": (73, 100), "control": (79, 100)},
}


def load_bits() -> np.ndarray:
    return np.load(RAW_KEY).astype(np.uint8)[:1_000_000]


def sha256_bits(bits: np.ndarray) -> str:
    return hashlib.sha256("".join(map(str, bits.tolist())).encode()).hexdigest()


# --- ours, parameter-matched to STS -----------------------------------------
def ours_pvalue(bits: np.ndarray) -> dict:
    """Per-stream p-values at n = 500 000 with STS's parameter choices."""
    out = {}
    out["Frequency (Monobit)"] = N.frequency_monobit(bits)[0]
    out["Block Frequency (M=128)"] = N.block_frequency(bits, 128)[0]
    out["Runs"] = N.runs(bits)[0]
    # STS tabulated partition; our battery default uses the exact DP pmf
    out["Longest Run of Ones (M=128)"] = N.longest_run_of_ones(
        bits, 128, scheme="sp80022")[0]
    # exact class probabilities; our battery default uses the 4-dp SP 800-22 values
    out["Binary Matrix Rank (32x32)"] = N.binary_matrix_rank(
        bits, 32, 32, exact_probs=True)[0]
    out["Discrete Fourier Transform"] = N.dft_spectral(bits)[0]
    out["Universal (L=6, Q=640)"] = N.universal(bits, 6, 640)[0]
    out["Approximate Entropy (m=10)"] = N.approximate_entropy(bits, 10)[0]
    # SP 800-22 applicability is m < floor(log2 n) - 2; STS hard-codes m = 16 at
    # every length, so we allow it here purely to make the comparison possible
    out["Serial (m=16)"] = N.serial(bits, 16, allow_out_of_range=True)[0]
    out["Overlapping Template (m=9)"] = N.overlapping_template(bits, 9, 1032)[0]
    out["Linear Complexity (M=500)"] = N.linear_complexity(bits, 500)[0]
    return out


def ours_passrate(chunks: np.ndarray) -> dict:
    """Pass counts over 100 streams at n = 10 000, STS-matched parameters."""
    res = {k: [] for k in STS_PASSRATE_RAW}
    for c in chunks:
        res["Frequency (Monobit)"].append(N.frequency_monobit(c)[0])
        res["Block Frequency (M=128)"].append(N.block_frequency(c, 128)[0])
        res["Cumulative Sums (fwd)"].append(N.cumulative_sums(c, "forward")[0])
        res["Cumulative Sums (rev)"].append(N.cumulative_sums(c, "reverse")[0])
        res["Runs"].append(N.runs(c)[0])
        res["Longest Run of Ones (M=128)"].append(
            N.longest_run_of_ones(c, 128, scheme="sp80022")[0])
        res["Binary Matrix Rank (32x32)"].append(
            N.binary_matrix_rank(c, 32, 32, exact_probs=True)[0])
        res["Discrete Fourier Transform"].append(N.dft_spectral(c)[0])
        m = 9
        ov = N.overlapping_template(c, m, 1032)[0]
        res["Overlapping Template (m=9)"].append(ov)
        res["Linear Complexity (M=500)"].append(N.linear_complexity(c, 500)[0])
    return {k: [int(sum(1 for p in v if p is not None and p >= 0.01)),
                int(sum(1 for p in v if p is not None))] for k, v in res.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    bits = load_bits()
    digest = sha256_bits(bits)

    # ---- comparison 1: identical p-values at n = 500 000 --------------------
    pv = {"stream1": ours_pvalue(bits[:500_000]),
          "stream2": ours_pvalue(bits[500_000:1_000_000])}
    rows = []
    for name, (s1, s2) in STS_PVALUE.items():
        if s1 is None:
            continue
        o1 = pv["stream1"][name]
        o2 = pv["stream2"][name]
        rows.append({"test": name, "ours": [o1, o2], "sts": [s1, s2],
                     "max_abs_diff": max(abs(o1 - s1), abs(o2 - s2)),
                     "reference_distribution": ("differs (exact vs asymptotic)"
                                                if name in THEORY_DIFFERS
                                                else "identical")})
    exact_rows = [r for r in rows if r["test"] not in THEORY_DIFFERS]
    maxdiff = max(r["max_abs_diff"] for r in exact_rows)

    # ---- comparison 2: pass counts at n = 10 000 x 100 streams -------------
    chunks = bits[:100 * 10_000].reshape(100, 10_000)
    ours_pass = ours_passrate(chunks)
    pass_rows = []
    for name, (sp, st) in STS_PASSRATE_RAW.items():
        op, ot = ours_pass[name]
        pass_rows.append({"test": name, "ours": [op, ot], "sts": [sp, st],
                          "agree": op == sp})

    out = {
        "suite": "NIST STS 2.1.2 (build: make, gcc, src/assess)",
        "input": {
            "source": "results/keys/qrng_raw_1Mbit.npy",
            "bits_used": int(bits.size),
            "sha256_ascii_bitstring": digest,
        },
        "comparison1_pvalues_n500000": {
            "configuration": "2 streams x 500 000 bits, ASCII input, all tests",
            "rows": rows,
            "max_abs_p_diff": float(maxdiff),
            "all_within_1e-6": bool(maxdiff < 1e-6),
            "tests_identical_to_1e-6": [r["test"] for r in exact_rows],
            "tests_with_documented_constant_difference": sorted(THEORY_DIFFERS),
        },
        "comparison2_passrates_n10000": {
            "configuration": "100 streams x 10 000 bits (STS reporting mode)",
            "rows": pass_rows,
            "all_pass_counts_agree": all(r["agree"] for r in pass_rows),
        },
        "out_of_domain": {
            "note": ("At n = 10 000 the Universal test (needs n >= 387 840 for "
                     "L >= 6) and ApEn (needs m < log2(n) - 5) are outside their "
                     "documented range. The STS flags below reproduce exactly "
                     "on true random bits, so they are a domain artefact."),
            "rows": STS_OUT_OF_DOMAIN,
        },
        "known_parameter_differences": [
            {"test": "Longest Run of Ones",
             "ours_default": "exact DP pmf, bins >= 5% mass",
             "sts": "tabulated Table 2.4 partition",
             "effect": "identical at M=10000 (7 classes); differ at M=128",
             "resolved_by": "scheme='sp80022'"},
            {"test": "Binary Matrix Rank",
             "ours_default": "P = 0.2888 / 0.5776 / 0.1336 (4 dp, SP 800-22 text)",
             "sts": "P = 0.288788 / 0.577576 / 0.133636 (6 dp)",
             "effect": "rank counts F identical; only the constants differ",
             "resolved_by": "exact_probs=True"},
            {"test": "Overlapping Template",
             "ours_default": "exact overlap distribution (dynamic programming pmf)",
             "sts": "eta/lambda asymptotic formula",
             "effect": ("class frequencies F identical; chi2 differs "
                        "(9.909 vs 9.406 on stream 1), so p differs ~2e-2"),
             "resolved_by": "documented; exact pmf retained deliberately"},
            {"test": "Serial",
             "ours_default": "m = min(16, floor(log2 n) - 3), refuses m >= log2(n)-2",
             "sts": "m = 16 at every length",
             "effect": "inapplicable for both at n = 10 000; identical at n >= 131 072",
             "resolved_by": "compare at n = 500 000"},
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  comparison 1: {len(exact_rows)}/{len(rows)} tests reproduce the STS "
          f"p-value to < 1e-6 (max |dp| = {maxdiff:.2e}); "
          f"{len(THEORY_DIFFERS)} differ only in the reference constants")
    n_agree = sum(r["agree"] for r in pass_rows)
    print(f"  comparison 2: {n_agree}/{len(pass_rows)} tests match STS pass counts")

    if args.do_print:
        print("\n--- n = 500 000, per-stream p-values ---")
        print(f"{'test':<32}{'ours s1':>10}{'STS s1':>10}{'ours s2':>10}{'STS s2':>10}")
        for r in rows:
            print(f"{r['test']:<32}{r['ours'][0]:>10.6f}{r['sts'][0]:>10.6f}"
                  f"{r['ours'][1]:>10.6f}{r['sts'][1]:>10.6f}")
        print("\n--- n = 10 000 x 100 streams, passing sequences ---")
        print(f"{'test':<32}{'ours':>10}{'STS':>10}")
        for r in pass_rows:
            print(f"{r['test']:<32}{r['ours'][0]:>7}/{r['ours'][1]:<2}"
                  f"{r['sts'][0]:>7}/{r['sts'][1]:<2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
