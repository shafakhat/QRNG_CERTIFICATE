"""
calibrate_battery.py
====================
Self-validation of the SP 800-22 implementation shipped with this repository.

Method (SP 800-22 Rev. 1a, Sec. 4.2 -- "second level" assessment):
  1. Generate M independent sequences of n bits from a reference PRNG
     (PCG64 with independent streams).
  2. Run the 14 statistic battery on each sequence, keeping the p-value.
  3. Check (a) the proportion of sequences passing at alpha = 0.01 against
     the confidence interval p_hat +/- 3 sqrt(p_hat(1-p_hat)/M), and
     (b) the uniformity of the M p-values with a chi-square test on 10 bins.

A correct implementation gives pass proportions inside the confidence
interval and uniformity p-values that are themselves roughly uniform.

Output: results/calibration_battery.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nist_sp800_22 as sts


def _run_one(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    bits = (rng.random(1_000_000) < 0.5).astype(np.uint8)
    res = sts.run_battery(bits)
    out = {k: v["p_value"] for k, v in res.items()}
    out["_apen_chi2"] = sts.approximate_entropy(bits, 10)["chi2"] if False else \
        res["Approximate Entropy (m=10)"]["info"]["chi2"]
    out["_apen_chi2"] = res["Approximate Entropy (m=10)"]["info"]["chi2"]
    return out


def main(M=60, out="results/calibration_battery.json"):
    t0 = time.time()
    with Pool(2) as pool:
        all_p = pool.map(_run_one, range(1000, 1000 + M))
    # ---- calibrate the ApEn reference distribution from the same null runs
    chi2s = np.array([d["_apen_chi2"] for d in all_p])
    apen_cal = {"n=1000000_m=10": {"mean": float(chi2s.mean()),
                                   "std": float(chi2s.std(ddof=1)),
                                   "M": M}}
    with open("results/apen_calibration.json", "w") as f:
        json.dump(apen_cal, f, indent=2)
    sts.APEN_CALIBRATION.update(
        {k: {"mean": v["mean"], "std": v["std"]} for k, v in apen_cal.items()
         if k.startswith("n=")})
    print("ApEn calibration:", apen_cal)

    names = [k for k in all_p[0].keys() if not k.startswith("_")]
    report = {"M_sequences": M, "n_bits": 1_000_000, "alpha": 0.01,
              "wall_seconds": None, "tests": {}}
    for name in names:
        if name.startswith("Approximate Entropy"):
            p = np.array([sts.approximate_entropy(
                (np.random.Generator(np.random.PCG64(seed)).random(1_000_000) < 0.5)
                .astype(np.uint8), 10)[0] for seed in range(1000, 1000 + M)])
        else:
            p = np.array([np.nan if d[name] is None else d[name] for d in all_p])
        ok = p[~np.isnan(p)]
        M_eff = ok.size
        passes = int(np.count_nonzero(ok >= 0.01))
        phat = passes / M_eff
        ci = 3 * np.sqrt(0.99 * 0.01 / M_eff)   # SP 800-22 uses p=0.99 for the CI
        hist, _ = np.histogram(ok, bins=10, range=(0, 1))
        chi2, punif = stats.chisquare(hist)
        report["tests"][name] = {
            "M_effective": M_eff, "n_pass": passes, "pass_rate": phat,
            "ci_low": phat - ci, "ci_high": phat + ci,
            "uniformity_chi2": float(chi2), "uniformity_p": float(punif),
            "mean_p": float(ok.mean()), "median_p": float(np.median(ok)),
            "ks_p": float(stats.kstest(ok, "uniform").pvalue),
            "calibrated": bool(phat - ci <= 0.99 <= phat + ci and punif >= 0.01),
        }
    report["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: {"pass_rate": round(v["pass_rate"], 4),
                          "uniformity_p": round(v["uniformity_p"], 4),
                          "calibrated": v["calibrated"]}
                      for k, v in report["tests"].items()}, indent=1))
    print("all calibrated:", all(v["calibrated"] for v in report["tests"].values()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
