"""
analyze_studio_counts.py
========================
Turns the counts you download from Classiq Studio's Execute tab into the paper's
quantities.  No Classiq / Qiskit import needed -- it reads the exported files.

Put the downloads in  results/studio_counts/  with these names (any subset works):

    chsh_a0b0.json / .csv   "  a0b1, a1b0, a1b1  ->  CHSH S value
    qrng_rotation.json      ->  P(1), bias, min-entropy, monobit p-value
    qrng_ghz.json           ->  <X^8>, GHZ correlation
    bb84_batch.json         ->  demo: per-qubit 1-fraction

then:

    python3 code/analyze_studio_counts.py
    python3 code/analyze_studio_counts.py --dir results/studio_counts
    python3 code/analyze_studio_counts.py --demo      # self-test, synthetic counts

Accepted file layouts (auto-detected, because Studio has changed its export
format more than once):
    {"00": 501, "11": 499}                     plain dict
    {"counts": {"00": 501}}                    wrapped dict
    [{"bitstring": "00", "counts": 501}, ...]  list of records
    bitstring,counts / counts,bitstring        CSV columns

Outputs -> results/classiq_execution.json and a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RES = os.path.join(REPO, "results")
INDIR = os.path.join(RES, "studio_counts")
OUT = os.path.join(RES, "classiq_execution.json")

CHSH_PAIRS = [("chsh_a0b0", "a0b0"), ("chsh_a0b1", "a0b1"),
              ("chsh_a1b0", "a1b0"), ("chsh_a1b1", "a1b1")]


# ---------------------------------------------------------------------------
# tolerant readers
# ---------------------------------------------------------------------------
def read_counts(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_csv(path)
    if ext == ".json":
        with open(path) as f:
            obj = json.load(f)
        return _from_obj(obj)
    raise ValueError(f"unsupported file type: {path}")


def _from_obj(obj):
    # {"counts": {...}}
    if isinstance(obj, dict) and "counts" in obj and isinstance(obj["counts"], dict):
        return {str(k): int(v) for k, v in obj["counts"].items()}
    # list of records
    if isinstance(obj, list):
        out = {}
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            key = rec.get("bitstring") or rec.get("state") or rec.get("key")
            val = rec.get("counts") or rec.get("count") or rec.get("frequency")
            if key is not None and val is not None:
                out[str(key)] = out.get(str(key), 0) + int(val)
        if out:
            return out
    # plain {bitstring: counts}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                out[str(k)] = int(v)
            elif isinstance(v, dict):          # {"chsh_a0b0": {"00": 8509, ...}}
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        out[str(kk)] = out.get(str(kk), 0) + int(vv)
        if out:
            return out
    raise ValueError("could not find counts in this JSON")


def _read_csv(path: str) -> dict:
    out = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise ValueError(f"{path}: empty CSV")
        cols = {c.lower().strip(): c for c in rd.fieldnames}
        kcol = next((cols[c] for c in ("bitstring", "bistring", "state", "key", "outcome")
                     if c in cols), None)
        vcol = next((cols[c] for c in ("counts", "count", "frequency", "value", "occurrences")
                     if c in cols), None)
        if kcol is None or vcol is None:
            raise ValueError(f"{path}: need a bitstring column and a counts column, "
                             f"got {rd.fieldnames}")
        for row in rd:
            out[row[kcol].strip()] = out.get(row[kcol].strip(), 0) + int(float(row[vcol]))
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _clean(bitstring: str) -> str:
    s = str(bitstring).replace(" ", "").replace("_", "")
    for pre in ("0b", "b"):
        if s.lower().startswith(pre) and len(s) > 2:
            s = s[2:]
    return s


def expectation(counts: dict):
    """E = P(same) - P(different) for a 2-qubit measurement."""
    same = diff = 0
    for k, v in counts.items():
        b = _clean(k)
        if len(b) < 2:
            continue
        (same, diff) = (same + v, diff) if b[-1] == b[-2] else (same, diff + v)
    n = same + diff
    if n == 0:
        raise ValueError("no 2-bit outcomes found")
    e = (same - diff) / n
    # binomial standard error of a correlation
    p_same = same / n
    se = 2.0 * math.sqrt(max(p_same * (1 - p_same), 1e-12) / n)
    return e, se, n


def bits_from_counts(counts: dict) -> list[int]:
    """Expand {bitstring: counts} into a flat list of bits."""
    out = []
    for k, v in counts.items():
        bits = [int(c) for c in _clean(k)]
        for _ in range(int(v)):
            out.extend(bits)
    return out


def monobit_p(bits: list[int]) -> tuple[float, float]:
    n = len(bits)
    ones = sum(bits)
    p1 = ones / n
    s = abs(2 * ones - n) / math.sqrt(n)
    p = math.erfc(s / math.sqrt(2))
    return p1, p


def min_entropy_mcv(bits: list[int]) -> float:
    """SP 800-90B most-common-value estimate on the bit sequence."""
    n = len(bits)
    ones = sum(bits)
    p_max = max(ones, n - ones) / n
    if p_max >= 1.0:
        return 0.0
    pu = min(1.0, p_max + 2.576 * math.sqrt(p_max * (1 - p_max) / max(n - 1, 1)))
    return -math.log2(pu) if pu < 1 else 0.0


def ghz_correlation(counts: dict, n: int = 8):
    """<X^n> after the H layer: mass on |0..0> and |1..1>."""
    total = sum(counts.values())
    allsame = sum(v for k, v in counts.items()
                  if _clean(k) in ("0" * n, "1" * n))
    p = allsame / total
    se = math.sqrt(max(p * (1 - p), 1e-12) / total)
    return 2 * p - 1, 2 * se, total


# ---------------------------------------------------------------------------
def analyse_chsh(files: dict) -> dict:
    Es, sig, Ns = {}, {}, {}
    for stem, tag in CHSH_PAIRS:
        if stem not in files:
            return {"status": "incomplete", "missing": stem}
        e, se, n = expectation(files[stem])
        Es[tag], sig[tag], Ns[tag] = e, se, n
    # Convention check: for |Phi+> with RY(-a) on Alice and RY(-b) on Bob,
    # E(a,b) = cos(a-b), so at (0, 0, pi/2, pi/2) x (pi/4, -pi/4, pi/4, -pi/4)
    # the correlators are (+,-,+,+)/sqrt2 and the VIOLATING combination is
    #   S = E00 + E01 + E10 - E11 = 3/sqrt2 + 1/sqrt2 = 2 sqrt2.
    # (The textbook E00 - E01 + E10 + E11 belongs to a different sign convention
    #  and gives 0 for these angles.)
    S = Es["a0b0"] + Es["a0b1"] + Es["a1b0"] - Es["a1b1"]
    sS = math.sqrt(sum(s ** 2 for s in sig.values()))
    return {"status": "ok", "E": {k: round(v, 5) for k, v in Es.items()},
            "sigma_E": {k: round(v, 5) for k, v in sig.items()},
            "shots_per_setting": Ns, "S": round(S, 4),
            "sigma_S": round(sS, 4),
            "violation_sigma": round((S - 2.0) / sS, 1),
            "S_theory_ideal": round(2 * math.sqrt(2), 4)}


def analyse_qrng_rotation(counts: dict) -> dict:
    bits = bits_from_counts(counts)
    p1, p = monobit_p(bits)
    return {"n_bits": len(bits), "P_1": round(p1, 6),
            "bias": round(abs(p1 - 0.5), 6),
            "min_entropy_mcv": round(min_entropy_mcv(bits), 5),
            "monobit_p_value": round(p, 4)}


def analyse_qrng_ghz(counts: dict, n: int = 8) -> dict:
    corr, se, total = ghz_correlation(counts, n)
    return {"shots": total, "X_correlation": round(corr, 5),
            "sigma": round(se, 5)}


# ---------------------------------------------------------------------------
def demo_files() -> dict:
    """Synthetic counts that mimic ideal hardware, for a self-test."""
    import random
    random.seed(7)
    out = {}
    ideal = {"a0b0": 1 / math.sqrt(2), "a0b1": -1 / math.sqrt(2),
             "a1b0": 1 / math.sqrt(2), "a1b1": 1 / math.sqrt(2)}
    for stem, tag in CHSH_PAIRS:
        e = ideal[tag]
        p_same = (1 + e) / 2
        n = 20000
        same = int(round(n * p_same))
        out[stem] = {"00": same // 2 + 5, "11": same - same // 2 - 5,
                     "01": (n - same) // 2, "10": n - same - (n - same) // 2}
    out["qrng_rotation"] = {}
    for _ in range(2000):                    # 2000 shots x 16 bits = 32 000 bits
        b = format(random.getrandbits(16), "016b")
        out["qrng_rotation"][b] = out["qrng_rotation"].get(b, 0) + 1
    out["qrng_ghz"] = {"0" * 8: 4990, "1" * 8: 5010}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=INDIR)
    ap.add_argument("--demo", action="store_true", help="self-test on synthetic counts")
    ap.add_argument("--tag", default="classiq_studio_execution",
                    help="label stored in the JSON (e.g. the job name)")
    args = ap.parse_args()

    if args.demo:
        raw = demo_files()
        print("[demo] using synthetic counts\n")
    else:
        if not os.path.isdir(args.dir):
            sys.exit(f"{args.dir} does not exist -- download the counts from Studio "
                     f"and put them there (chsh_a0b0.json ... qrng_rotation.json ...)")
        files, seen = {}, []
        for f in sorted(os.listdir(args.dir)):
            stem = os.path.splitext(f)[0]
            if stem in ("summary",):
                continue
            try:
                files[stem] = read_counts(os.path.join(args.dir, f))
                seen.append(f"{f} -> {len(files[stem])} distinct outcomes, "
                            f"{sum(files[stem].values())} shots")
            except Exception as e:
                print(f"  ! skipped {f}: {e}")
        if not files:
            sys.exit(f"no readable count files in {args.dir}")
        print("read:")
        for s in seen:
            print("  " + s)
        raw = files

    report = {"_source": args.tag, "files": sorted(raw.keys())}

    chsh_keys = [s for s, _ in CHSH_PAIRS if s in raw]
    if chsh_keys:
        report["chsh"] = analyse_chsh(raw)
        if report["chsh"]["status"] == "ok":
            c = report["chsh"]
            print(f"\nCHSH  S = {c['S']:.4f} +- {c['sigma_S']:.4f}   "
                  f"({c['violation_sigma']} sigma above 2, ideal {c['S_theory_ideal']})")
            for tag, e in c["E"].items():
                print(f"      E({tag}) = {e:+.4f}")
        else:
            print(f"\nCHSH: incomplete, needs {report['chsh']['missing']}")

    if "qrng_rotation" in raw:
        report["qrng_rotation"] = analyse_qrng_rotation(raw["qrng_rotation"])
        r = report["qrng_rotation"]
        print(f"\nQRNG rotation: {r['n_bits']} bits  P(1) = {r['P_1']:.6f}  "
              f"bias {r['bias']:.2e}  H_min {r['min_entropy_mcv']:.4f}  "
              f"monobit p = {r['monobit_p_value']:.4f}")

    if "qrng_ghz" in raw:
        report["qrng_ghz"] = analyse_qrng_ghz(raw["qrng_ghz"])
        r = report["qrng_ghz"]
        print(f"\nQRNG GHZ: <X^8> = {r['X_correlation']:.4f} +- {r['sigma']:.4f} "
              f"({r['shots']} shots)")

    if "bb84_batch" in raw:
        c = raw["bb84_batch"]
        total = sum(c.values())
        report["bb84_batch"] = {"shots": total,
                                "distinct_outcomes": len(c),
                                "top": dict(sorted(c.items(), key=lambda kv: -kv[1])[:4])}
        print(f"\nBB84 batch: {total} shots, {len(c)} distinct 16-bit outcomes")

    os.makedirs(RES, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
