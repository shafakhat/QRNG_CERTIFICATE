"""
run_qrng.py
===========
All QRNG experiments.  Every number is written to `results/qrng_results.json`.

    F1  Generator characterisation: 1 Mbit raw key from the rotation-based
        generator (theta = pi/2), bias, Shannon/min-entropy, full NIST battery
    F2  Miscalibration scan: bias vs theta, measured vs the Born rule
    F3  Device-noise scan: channel depolarisation + readout error
    F4  SP 800-90B min-entropy estimation (9 estimators) + IID permutation test
    F5  Toeplitz privacy amplification / extraction and validation of the key
    F6  GHZ-based generator and randomness certification (correlators)
    F7  Second-level statistical analysis over K independent sequences
    F8  Reference comparison against a classical PRNG (PCG64)

Reproduce the sampling on Classiq cloud with

    from classiq import synthesize, execute, ExecutionPreferences
    qprog = synthesize(qmod_models.model_qrng_rotation(theta=np.pi/2))
    res = execute(qprog, execution_preferences=ExecutionPreferences(
                      num_shots=65_536)).result()
    bits = [int(b, 2) for c in res[0].value.parsed_counts for b in [c.state["out"]]]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as C
import nist_sp800_22 as nist
import postprocessing as P
import sp800_90b as sb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
KEYS = os.path.join(RES, "keys")
os.makedirs(KEYS, exist_ok=True)

BATCH = C.N_BATCH


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_qrng(theta=np.pi / 2, n_bits=1_000_000, p_ch=0.0, p_ro=0.0,
                seed=None, method="stabilizer"):
    shots = int(np.ceil(n_bits / BATCH))
    qc = C.qrng_circuit(theta, BATCH, p_ch=p_ch)
    bits = C.sample_bits(qc, shots, p_ro=p_ro, method=method, seed=seed)
    return bits[:n_bits]


def bias_stats(bits):
    n = bits.size
    ones = int(bits.sum())
    p1 = ones / n
    # 99.999% Wilson interval on the bias
    z = 4.417
    lo, hi = P.wilson_interval(ones, n, z=z)
    return {"n_bits": int(n), "ones": ones, "p1": p1, "bias": p1 - 0.5,
            "bias_ci99_999": [lo - 0.5, hi - 0.5],
            "monobit_z": (2 * ones - n) / np.sqrt(n),
            "shannon_entropy": float(P.h2(p1)),
            "min_entropy": float(-np.log2(max(p1, 1 - p1))),
            "autocorr_lag1": float(np.corrcoef(bits[:-1], bits[1:])[0, 1]),
            "runs": int(1 + np.count_nonzero(np.diff(bits)))}


# ---------------------------------------------------------------------------
# F1  main characterisation
# ---------------------------------------------------------------------------
def main_characterisation(n_bits=1_000_000, seed=101):
    t0 = time.time()
    bits = sample_qrng(np.pi / 2, n_bits, seed=seed)
    st = bias_stats(bits)
    res = nist.run_battery(bits)
    st["battery"] = {k: {"p_value": v["p_value"], "pass": v["pass"],
                         "info": {kk: (vv if not isinstance(vv, (list, dict)) else "...")
                                  for kk, vv in v["info"].items()}}
                     for k, v in res.items()}
    st["battery_pass"] = int(sum(v["pass"] for v in res.values()))
    st["battery_total"] = len(res)
    st["battery_seconds"] = round(time.time() - t0, 1)
    np.save(os.path.join(KEYS, "qrng_raw_1Mbit.npy"), bits)
    if os.path.exists(os.path.join(RES, "apen_calibration.json")):
        with open(os.path.join(RES, "apen_calibration.json")) as f:
            cal = json.load(f)
        nist.APEN_CALIBRATION.update({k: {"mean": v["mean"], "std": v["std"]}
                                     for k, v in cal.items()})
    print(f"  raw 1 Mbit: p1={st['p1']:.6f} bias={st['bias']:+.2e} "
          f"H_sh={st['shannon_entropy']:.6f} H_min={st['min_entropy']:.6f} "
          f"battery {st['battery_pass']}/{st['battery_total']}")
    return st


# ---------------------------------------------------------------------------
# F2  miscalibration scan
# ---------------------------------------------------------------------------
def miscalibration_scan(deltas, n_bits=200_000, seed=200):
    rows = []
    for i, d in enumerate(deltas):
        theta = np.pi / 2 + d
        bits = sample_qrng(theta, n_bits, seed=seed + i, method="statevector")
        st = bias_stats(bits)
        p1_theory = np.sin(theta / 2) ** 2
        rows.append({"delta_rad": float(d), "theta": float(theta),
                     "p1_measured": st["p1"], "p1_theory": p1_theory,
                     "bias_measured": st["bias"],
                     "bias_theory": p1_theory - 0.5,
                     "shannon_entropy": st["shannon_entropy"],
                     "min_entropy": st["min_entropy"],
                     "monobit_z": st["monobit_z"]})
        print(f"  delta={d:+.4f} rad: p1={st['p1']:.5f} (theory {p1_theory:.5f}) "
              f"H_min={st['min_entropy']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# F3  device-noise scan
# ---------------------------------------------------------------------------
def noise_scan(p_chs, p_ros, n_bits=200_000, seed=300):
    rows = []
    for i, p_ch in enumerate(p_chs):
        for j, p_ro in enumerate(p_ros):
            bits = sample_qrng(np.pi / 2, n_bits, p_ch=p_ch, p_ro=p_ro,
                               seed=seed + 10 * i + j)
            st = bias_stats(bits)
            res = nist.run_battery(bits) if n_bits >= 200_000 else {}
            rows.append({"p_ch": p_ch, "p_ro": p_ro,
                         "p1": st["p1"], "bias": st["bias"],
                         "shannon_entropy": st["shannon_entropy"],
                         "min_entropy": st["min_entropy"],
                         "autocorr_lag1": st["autocorr_lag1"],
                         "battery_pass": int(sum(v["pass"] for v in res.values())),
                         "battery_total": len(res)})
            print(f"  p_ch={p_ch:.3f} p_ro={p_ro:.3f}: p1={st['p1']:.5f} "
                  f"H_min={st['min_entropy']:.4f} battery "
                  f"{rows[-1]['battery_pass']}/{rows[-1]['battery_total']}")
    return rows


# ---------------------------------------------------------------------------
# F4  SP 800-90B entropy estimation
# ---------------------------------------------------------------------------
def entropy_estimation(bits, n_use=200_000, seed=400, n_perms=100):
    sample = bits[:n_use]
    t0 = time.time()
    est = sb.min_entropy_90b(sample)
    iid = sb.permutation_test_iid(sample[:50_000], n_perms=n_perms, seed=seed)
    print(f"  min over 9 estimators = {est['_summary']['min']:.4f} bits/sample; "
          f"IID permutation p = {iid['p_value']:.3f} ({time.time()-t0:.0f}s)")
    return {"estimators": {k: v for k, v in est.items() if k != "_summary"},
            "summary": est["_summary"], "iid_test": iid,
            "n_used": int(sample.size), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# F5  extraction
# ---------------------------------------------------------------------------
def extraction(bits, h_min, eps_pa=1e-10, security_margin=0.02, seed=500):
    """Toeplitz strong extractor: m = n (H_min - margin) - 2 log2(1/eps)."""
    n = bits.size
    m = int(n * max(0.0, h_min - security_margin) - 2 * np.log2(1 / eps_pa))
    m = max(1, min(m, n - 1))
    t0 = time.time()
    key = P.toeplitz_pa(bits, m, seed=seed)
    st = bias_stats(key)
    res = nist.run_battery(key)
    out = {"n_in": int(n), "n_out": int(m), "h_min_used": float(h_min),
           "security_margin": security_margin, "eps_pa": eps_pa,
           "output_rate": m / n, "compression": n / m,
           "p1": st["p1"], "min_entropy": st["min_entropy"],
           "battery_pass": int(sum(v["pass"] for v in res.values())),
           "battery_total": len(res),
           "battery": {k: {"p_value": v["p_value"], "pass": v["pass"]}
                       for k, v in res.items()},
           "seconds": round(time.time() - t0, 1)}
    np.save(os.path.join(KEYS, "qrng_extracted_key.npy"), key)
    print(f"  extracted {m} bits from {n} (rate {m/n:.3f}), battery "
          f"{out['battery_pass']}/{out['battery_total']}")
    return out


# ---------------------------------------------------------------------------
# F6  GHZ certification
# ---------------------------------------------------------------------------
def ghz_certification(n_qubits=8, shots=200_000, p_chs=(0.0, 0.01, 0.02), seed=600):
    """Measure the GHZ correlators in the Z and X bases.

    Ideal values: <Z^n> = 1 and <X^n> = 1, and the individual outcomes are
    uniformly distributed while being perfectly correlated.  The pair of
    incompatible observables certifies that the randomness is not a
    pre-determined classical variable (a source with <X^n> = <Z^n> = 1 cannot
    be reproduced by a deterministic model with a fixed outcome string).
    """
    rows = []
    for i, p in enumerate(p_chs):
        vals = {}
        for basis in (0, 1):
            qc = C.ghz_circuit(n_qubits, x_basis=basis, p_ch=p)
            bits = C.sample_bits(qc, shots, seed=seed + i * 7 + basis)
            m = bits.reshape(shots, n_qubits)
            par = (m.sum(axis=1) % 2)
            corr = 1 - 2 * par.mean()
            vals["Z" if basis == 0 else "X"] = {
                "correlator": float(corr),
                "sigma": float(2 * np.sqrt(par.mean() * (1 - par.mean()) / shots)),
                "parity_uniformity_p": float(
                    2 * (1 - abs(2 * par.mean() - 1)) if True else 1.0),
                "first_qubit_p1": float(m[:, 0].mean())}
        row = {"n_qubits": n_qubits, "shots": shots, "p_ch": p,
               "Z_correlator": vals["Z"]["correlator"],
               "Z_sigma": vals["Z"]["sigma"],
               "X_correlator": vals["X"]["correlator"],
               "X_sigma": vals["X"]["sigma"],
               "Z_theory": (1 - 4 * p / 3) ** n_qubits,
               "X_theory": (1 - 4 * p / 3) ** n_qubits,
               "p1_first_qubit": vals["Z"]["first_qubit_p1"]}
        rows.append(row)
        print(f"  p_ch={p}: <Z^{n_qubits}>={row['Z_correlator']:+.4f} "
              f"<X^{n_qubits}>={row['X_correlator']:+.4f}")
    return rows


# ---------------------------------------------------------------------------
# F7  second-level analysis
# ---------------------------------------------------------------------------
def _spread_seeds(master_seed, k):
    """
    Well-spread simulator seeds.

    Consecutive small seeds are correlated for counter/linear PRNGs and the
    resulting streams show detectable structure: stage F7b measures a 25%
    failure rate of the linear-complexity statistic on streams seeded
    consecutively (700, 701, ...) versus 0% with spread seeds.  All
    multi-sequence studies derive their seeds from a master seed through an
    independent PCG64 stream.
    """
    rng = np.random.default_rng(master_seed)
    return [int(x) for x in rng.integers(1, 2 ** 31 - 1, k)]


def second_level(n_sequences=8, n_bits=1_000_000, seed=700):
    """K independent 1-Mbit sequences -> proportion and uniformity analysis."""
    from scipy import stats
    allp = []
    seeds = _spread_seeds(seed, n_sequences)
    for s in seeds:
        bits = sample_qrng(np.pi / 2, n_bits, seed=s)
        res = nist.run_battery(bits)
        allp.append({k: v["p_value"] for k, v in res.items()})
    names = list(allp[0].keys())
    out = {}
    for name in names:
        p = np.array([np.nan if d[name] is None else d[name] for d in allp])
        ok = p[~np.isnan(p)]
        M = ok.size
        passes = int(np.count_nonzero(ok >= 0.01))
        phat = passes / M
        ci = 3 * np.sqrt(0.99 * 0.01 / M)
        hist, _ = np.histogram(ok, bins=10, range=(0, 1))
        chi2, punif = stats.chisquare(hist)
        out[name] = {"M": int(M), "pass_rate": phat, "ci": ci,
                     "uniformity_p": float(punif),
                     "calibrated": bool(abs(phat - 0.99) <= ci and punif >= 0.01)}
        print(f"  {name:<34s} pass={phat:.3f} (0.99+-{ci:.3f})  "
              f"uniformity p={punif:.3f}")
    out["_seeds"] = seeds
    return out


# ---------------------------------------------------------------------------
# F7b  sampler qualification (seed hygiene of the simulator)
# ---------------------------------------------------------------------------
def sampler_qualification(n_seq=8, n_bits=1 << 19, seed=1100):
    """
    Qualify the sampling layer itself: the statistical quality of the sampler
    bounds what a QRNG characterisation can claim.  Two checks are run:
    (i) a chi-square over all 256 8-bit words of a 1-Mbit stream, which is
    sensitive to correlations within a shot; and (ii) the linear-complexity
    statistic on streams generated with consecutive versus spread simulator
    seeds, which is sensitive to PRNG seed correlation.
    """
    from scipy import stats
    bits = sample_qrng(np.pi / 2, 1 << 20, seed=seed)
    w8 = nist._window_values(bits, 8)
    cnt = np.bincount(w8, minlength=256)[:256]
    word_p = float(stats.chisquare(cnt).pvalue)
    packed = bits.reshape(-1, 16) @ (1 << np.arange(15, -1, -1))

    consec, spread = [], []
    for i in range(n_seq):
        b1 = sample_qrng(np.pi / 2, n_bits, seed=1200 + i)          # consecutive
        b2 = sample_qrng(np.pi / 2, n_bits, seed=_spread_seeds(1300 + i, 1)[0])
        consec.append(nist.linear_complexity(b1, 500)[0])
        spread.append(nist.linear_complexity(b2, 500)[0])
    out = {"word_chi2_p_8bit": word_p,
           "n_unique_16bit_words": int(np.unique(packed).size),
           "n_shots": int(packed.size),
           "lc_pass_consecutive_seeds": float(np.mean(np.array(consec) >= 0.01)),
           "lc_pass_spread_seeds": float(np.mean(np.array(spread) >= 0.01)),
           "lc_p_consecutive": [float(x) for x in consec],
           "lc_p_spread": [float(x) for x in spread],
           "n_sequences": n_seq, "n_bits_per_sequence": int(n_bits)}
    print(f"  8-bit word chi2 p={word_p:.3f}; LC pass rate: consecutive seeds "
          f"{out['lc_pass_consecutive_seeds']:.2f}, spread seeds "
          f"{out['lc_pass_spread_seeds']:.2f}")
    return out


# ---------------------------------------------------------------------------
# F10  validation of the entropy estimators
# ---------------------------------------------------------------------------
def estimator_validation(ps=(0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.97),
                         n=60_000, seed=800):
    """
    Apply the nine SP 800-90B estimators to Bernoulli(p) sources whose exact
    min-entropy is -log2(p).  A valid estimator must never *over*-estimate
    (an over-estimate would yield an unsafe extraction length); the table
    documents the measured conservativeness of each estimator.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in ps:
        x = (rng.random(n) < p).astype(np.uint8)
        est = sb.min_entropy_90b(x)
        true = float(-np.log2(max(p, 1 - p)))
        row = {"p": p, "true_hmin": true, "n_used": int(n),
               "estimates": {k: v["hmin"] for k, v in est.items() if k != "_summary"},
               "min": est["_summary"]["min"],
               "argmin": est["_summary"]["argmin"],
               "all_conservative": bool(all(v["hmin"] <= true + 1e-9
                                           for k, v in est.items() if k != "_summary"))}
        rows.append(row)
        print(f"  p={p:.2f} true H_min={true:.3f} min-est={row['min']:.3f} "
              f"({row['argmin']}) conservative={row['all_conservative']}")
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
STAGES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]


def main(quick=False, stages=None):
    """
    Run all experiments, or only the ones listed in `stages` (e.g.
    ``--stages F7,F9``).  Results are merged into results/qrng_results.json, so
    a stage can be re-run without repeating the others.
    """
    stages = STAGES if not stages else stages
    path = os.path.join(RES, "qrng_results_quick.json" if quick
                        else "qrng_results.json")
    res = {}
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
    res.setdefault("meta", {})
    res["meta"].update({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "simulator": "qiskit-aer (stabilizer / statevector)",
                        "batch_qubits_per_shot": BATCH,
                        "stages_run": stages})
    t0 = time.time()

    if "F1" in stages:
        print("\n== F1  generator characterisation (1 Mbit) ==")
        res["F1_main"] = main_characterisation(1 << (20 if not quick else 16))

    if "F2" in stages:
        print("\n== F2  miscalibration scan ==")
        deltas = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20] if not quick else [0.0, 0.05]
        res["F2_miscalibration"] = miscalibration_scan(deltas, 2 * 10 ** (5 if not quick else 4))

    if "F3" in stages:
        print("\n== F3  device-noise scan ==")
        p_chs = [0.0, 0.01, 0.02, 0.05] if not quick else [0.0, 0.01]
        p_ros = [0.0, 0.01, 0.02] if not quick else [0.0]
        res["F3_noise"] = noise_scan(p_chs, p_ros, 2 * 10 ** (5 if not quick else 4))

    raw = np.load(os.path.join(KEYS, "qrng_raw_1Mbit.npy"))
    if "F4" in stages:
        print("\n== F4  SP 800-90B min-entropy ==")
        res["F4_entropy"] = entropy_estimation(
            raw, n_use=200_000 if not quick else 50_000,
            n_perms=100 if not quick else 20)

    if "F5" in stages:
        print("\n== F5  privacy amplification / extraction ==")
        hmin = res["F4_entropy"]["summary"]["min"]
        res["F5_extraction"] = extraction(raw, hmin, seed=501)

    if "F6" in stages:
        print("\n== F6  GHZ certification ==")
        res["F6_ghz"] = ghz_certification(8, 200_000 if not quick else 20_000,
                                          (0.0, 0.01, 0.02) if not quick else (0.0,))

    if "F7" in stages:
        print("\n== F7  second-level analysis ==")
        res["F7_second_level"] = second_level(16 if not quick else 3,
                                              1 << (20 if not quick else 18))
        print("\n== F7b  sampler qualification (seed hygiene) ==")
        res["F7b_sampler"] = sampler_qualification(
            8 if not quick else 2, 1 << (19 if not quick else 15))

    if "F8" in stages:
        print("\n== F8  classical PRNG reference ==")
        rng = np.random.default_rng(42)
        prng = (rng.random(1 << 20) < 0.5).astype(np.uint8)
        prav = nist.run_battery(prng)
        res["F8_prng"] = {"battery_pass": int(sum(1 for v in prav.values() if v["pass"])),
                          "battery_applicable": int(sum(1 for v in prav.values()
                                                      if v["p_value"] is not None)),
                          "battery_total": len(prav),
                          "battery": {k: {"p_value": v["p_value"], "pass": v["pass"]}
                                      for k, v in prav.items()}}

    if "F9" in stages:
        print("\n== F9  negative controls (failure modes) ==")
        import failure_modes as fm
        res["F9_failure_modes"] = fm.run(200_000 if not quick else 20_000)

    if "F10" in stages:
        print("\n== F10  entropy-estimator validation ==")
        res["F10_estimator_validation"] = estimator_validation(
            (0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.97) if not quick else (0.50, 0.75),
            n=60_000 if not quick else 20_000)

    res["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "qrng_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nall QRNG experiments finished in {res['meta']['wall_seconds']} s")
    return res


if __name__ == "__main__":
    st = None
    for a in sys.argv[1:]:
        if a.startswith("--stages"):
            st = [x.strip() for x in a.split("=", 1)[1].split(",")]
    main(quick="--quick" in sys.argv, stages=st)
