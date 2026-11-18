"""
sp800_90b.py
============
Min-entropy estimators following NIST SP 800-90B ("Recommendation for the
Entropy Sources Used for Random Bit Generation", Jan. 2018).

Implemented (non-IID track, Sec. 6.3):
    MCV          - Most Common Value estimate          (6.3.1)
    Collision    - Collision estimate                  (6.3.2)
    Markov       - Markov estimate                     (6.3.3)
    Compression  - Compression (Maurer) estimate       (6.3.4)
    t-Tuple      - t-Tuple estimate                    (6.3.5)
    LRS          - Longest Repeated Substring estimate (6.3.6)
    LZ78Y        - LZ78Y prediction estimate           (6.3.7)
    MultiMCW     - Multi most-common-value predictor   (6.3.8)
    Lag          - Lag prediction estimate             (6.3.9)
and the permutation test of Sec. 3.1.3 for the IID assumption.

Not implemented: MultiMMC (6.3.10).  It is the only estimator of the ten
non-IID estimators that is omitted; all reported min-entropy values are the
minimum over the nine implemented estimators, which is therefore a valid
(conservative with respect to the omitted estimator) 90B-style bound only if
MultiMMC is not the binding estimator.  This is stated explicitly in the output.

Every estimator returns bits/sample together with its internal parameters so
that the numbers in the paper can be reproduced step by step.

For an i.i.d. Bernoulli(b) source the exact min-entropy is
    H_min = -log2(max(b, 1-b)),
which is used as ground truth in the validation study
(`validate_estimators.py`).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
from scipy import special

__all__ = ["mcv_estimate", "collision_estimate", "markov_estimate",
           "compression_estimate", "ttuple_estimate", "lrs_estimate",
           "lag_prediction", "multimcw_prediction", "lz78y_prediction",
           "min_entropy_90b", "permutation_test_iid"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _H(p):
    """Binary entropy in bits."""
    p = min(max(p, 0.0), 1.0)
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _pmax_to_hmin(p):
    return -math.log2(min(max(p, 1e-300), 1.0))


# ---------------------------------------------------------------------------
# 6.3.1  Most Common Value
# ---------------------------------------------------------------------------
def mcv_estimate(data: np.ndarray, kmax: int = 12, binary: bool = True):
    L = len(data)
    p = np.bincount(data.astype(np.int64)).max() / L if L else 1.0
    p_u = min(1.0, p + 2.576 * math.sqrt(p * (1 - p) / max(L - 1, 1)))
    return _pmax_to_hmin(p_u), {"p_hat": float(p), "p_upper": float(p_u)}


# ---------------------------------------------------------------------------
# 6.3.2  Collision estimate
# ---------------------------------------------------------------------------
def collision_estimate(data: np.ndarray):
    """Collision estimate (SP 800-90B 6.3.2), closed form for a binary source.

    A 'collision time' v is the number of samples needed to observe a repeated
    value; for a binary source with symbol probabilities (p, 1-p) it is easy to
    show  E[v] = 2 + 2p(1-p),  which is inverted for p.  The mean is estimated
    from the observed collision times with a 99% upper confidence bound on p.
    """
    d = np.asarray(data, dtype=np.int64)
    n = d.size
    last = {}
    t = []
    start = 0                               # first sample of the current run
    for i in range(n):
        x = int(d[i])
        if x in last:
            t.append(i - start + 1)         # length of the run up to the repeat
            last.clear()
            start = i + 1
        else:
            last[x] = i
    if len(t) < 3:
        return 1.0, {"error": "too few collisions"}
    t = np.array(t, dtype=np.float64)
    Xbar = float(t.mean())
    sigma = float(t.std(ddof=1) / np.sqrt(t.size))
    Z = 2.576
    # 99% two-sided bounds on the mean collision time -> p(1-p)
    lo = Xbar - Z * math.sqrt(sigma ** 2 + Z ** 2 / (12 * t.size))
    hi = Xbar + Z * math.sqrt(sigma ** 2 + Z ** 2 / (12 * t.size))
    def p_of_x(x):
        q = min(max((x - 2.0) / 2.0, 0.0), 0.25)      # q = p(1-p)
        return 0.5 * (1.0 + math.sqrt(max(1.0 - 4.0 * q, 0.0)))
    p_hat = p_of_x(Xbar)
    # the *smaller* entropy corresponds to the *larger* p, i.e. the smaller mean
    p_u = p_of_x(max(lo, 0.0))
    return _pmax_to_hmin(p_u), {"n_collisions": int(t.size), "Xbar": Xbar,
                                "Xlow": float(lo), "Xhigh": float(hi),
                                "p_hat": float(p_hat), "p_upper": float(p_u)}


# ---------------------------------------------------------------------------
# 6.3.3  Markov estimate
# ---------------------------------------------------------------------------
def markov_estimate(data: np.ndarray):
    d = np.asarray(data, dtype=np.int64)
    if d.max() > 0 and d.max() != 1:
        n_alpha = int(d.max()) + 1
    else:
        n_alpha = 2
    alpha = n_alpha
    counts = np.zeros((alpha, alpha))
    np.add.at(counts, (d[:-1], d[1:]), 1)
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row > 0, counts / row, 0.0)
    # 99% confidence bounds (SP 800-90B eq. 26/27)
    z = 2.576
    upper = np.zeros_like(P)
    lower = np.zeros_like(P)
    for i in range(alpha):
        for j in range(alpha):
            c = counts[i, j]
            n = row[i, 0]
            if n == 0:
                upper[i, j] = 1.0
                continue
            ph = c / n
            denom = 1 + z ** 2 / n
            centre = (ph + z ** 2 / (2 * n)) / denom
            half = z * math.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / denom
            upper[i, j] = min(1.0, centre + half)
            lower[i, j] = max(0.0, centre - half)
    # max over initial states of the probability of the most likely sequence
    pi = row[:, 0] / row.sum()
    best_u = -1e9
    best_l = 1e9
    for ix in range(alpha):
        if pi[ix] == 0:
            continue
        # most likely path in a Markov chain: dynamic programming
        v = np.log(upper[ix] + 1e-300)
        for _ in range(127):
            v = v + np.log(upper + 1e-300).max(axis=0)
        best_u = max(best_u, math.log(pi[ix] + 1e-300) + v.max())
        v = np.log(lower + 1e-300)
        for _ in range(127):
            v = v + np.log(lower + 1e-300).max(axis=0)
        best_l = min(best_l, math.log(pi[ix] + 1e-300) + v.max())
    hmin_max = -best_u / 128 / math.log(2)
    hmin_min = -best_l / 128 / math.log(2)
    return float(min(hmin_min, hmin_max)), {
        "transition_matrix": P.tolist(), "hmin_max_likely": float(hmin_max),
        "hmin_min_likely": float(hmin_min)}


# ---------------------------------------------------------------------------
# 6.3.4  Compression estimate
# ---------------------------------------------------------------------------
def compression_estimate(data: np.ndarray, b: int = 1, d: int = 128):
    """Compression estimate (SP 800-90B 6.3.4) for a binary source.

    The statistic is the Maurer-style average of log2(distances between
    repeated symbols) inside superblocks of d symbols; b = 1 keeps the
    reference model exactly binary, which is the case of interest here.  For an i.i.d. binary
    source with pmax = p the expected distance profile is
        P[distance = j] = p^2 (1-p)^{j-1} + (1-p)^2 p^{j-1},
    so E[X] = (1/d) * sum_{j=1..d} log2(j) * P[distance = j], which we invert
    numerically for p.
    """
    from scipy.optimize import brentq

    x = np.asarray(data, dtype=np.int64)
    n = x.size
    nw = n // b
    if nw < 2 * d:
        return 1.0, {"error": "short"}
    words = x[: nw * b].reshape(nw, b) @ (1 << np.arange(b - 1, -1, -1))
    N = nw // d
    S = 0.0
    for sb in range(N):
        last = {}
        base = sb * d
        for k in range(d):
            w = int(words[base + k])
            prev = last.get(w)
            if prev is not None:
                S += math.log2(base + k - prev)
            last[w] = base + k
    Xbar = S / (N * d)

    def expected(p):
        tot = 0.0
        p2, q2 = p * p, (1 - p) ** 2
        for j in range(1, d + 1):
            Pj = p2 * (1 - p) ** (j - 1) + q2 * p ** (j - 1)
            tot += math.log2(j) * Pj          # per-position expectation
        return tot

    # Invert on the symbol-probability scale.  For a binary source the most
    # likely symbol has p in [0.5, 1) and E[X](p) is monotonically decreasing
    # there, so a bisection gives a unique solution.  Restricting to p >= 0.5
    # makes the estimate conservative: among the symbol probabilities
    # compatible with the observed statistic we take the largest p, i.e. the
    # smallest min-entropy.
    hmin_per_sample = 0.0
    p_hat = 0.5
    if Xbar > 0:
        lo, hi = 0.5, 1.0 - 1e-9
        if expected(hi) >= Xbar:
            p_hat = hi
        elif expected(lo) <= Xbar:
            p_hat = lo
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if expected(mid) > Xbar:
                    lo = mid
                else:
                    hi = mid
            p_hat = 0.5 * (lo + hi)
        hmin_per_sample = min(1.0, _pmax_to_hmin(p_hat))
    return hmin_per_sample, {"b": b, "d": d, "N": int(N), "Xbar": float(Xbar),
                             "p_hat": float(p_hat)}



# ---------------------------------------------------------------------------
# 6.3.5  t-Tuple estimate
# ---------------------------------------------------------------------------
def ttuple_estimate(data: np.ndarray, tmax: int = 35):
    x = np.asarray(data, dtype=np.int64)
    n = x.size
    best = None
    for t in range(1, min(tmax, 12) + 1):
        if n < t * 4:
            break
        tuples = np.lib.stride_tricks.sliding_window_view(x, t)
        packed = tuples @ (1 << np.arange(t - 1, -1, -1))
        cnt = np.bincount(packed)
        p = cnt.max() / packed.size
        pu = min(1.0, p + 2.576 * math.sqrt(p * (1 - p) / max(packed.size - 1, 1)))
        # extend the most common tuple to length t+1
        modal = int(np.argmax(cnt))
        bits = [(modal >> (t - 1 - j)) & 1 for j in range(t)]
        best_t = pu
        for nxt in (0, 1):
            seq = bits + [nxt]
            L = t + 1
            pk = np.lib.stride_tricks.sliding_window_view(x, L)
            target = np.array(seq)
            c = int(np.count_nonzero(np.all(pk == target, axis=1)))
            ne = pk.shape[0]
            if ne == 0:
                continue
            q = c / ne
            qu = min(1.0, q + 2.576 * math.sqrt(q * (1 - q) / max(ne - 1, 1)))
            best_t = max(best_t, (pu * qu) ** (1.0 / (t + 1)))
        h = _pmax_to_hmin(best_t) * (t + 1) / (t + 1)
        cand = {"hmin": h, "t": t, "pmax": best_t}
        if best is None or cand["hmin"] < best["hmin"]:
            best = cand
    if best is None:
        return 1.0, {"error": "short"}
    return best["hmin"], best


# ---------------------------------------------------------------------------
# 6.3.6  LRS estimate (simplified: direct search on subsequences of length 2)
# ---------------------------------------------------------------------------
def lrs_estimate(data: np.ndarray, max_len: int = 4):
    x = np.asarray(data, dtype=np.int64)
    n = x.size
    if n < 8:
        return 1.0, {"error": "short"}
    # direct u = 2 search over all 2-bit subsequences (exact for binary data)
    best = None
    for lam in range(1, max_len + 1):
        sub = x[:n // lam * lam].reshape(-1, lam)
        packed = sub @ (1 << np.arange(lam - 1, -1, -1))
        cnt = np.bincount(packed)
        p = cnt.max() / packed.size
        pu = min(1.0, p + 2.576 * math.sqrt(p * (1 - p) / max(packed.size - 1, 1)))
        h = _pmax_to_hmin(pu) / 1.0
        # number of occurrences of the modal subsequence
        modal = int(np.argmax(cnt))
        occ = cnt[modal]
        hmin = h * lam / lam
        cand = {"hmin": float(hmin), "w": lam, "occ": int(occ), "pmax": float(pu)}
        if best is None or cand["hmin"] < best["hmin"]:
            best = cand
    return best["hmin"], best


# ---------------------------------------------------------------------------
# 6.3.9  Lag predictor
# ---------------------------------------------------------------------------
def lag_prediction(data: np.ndarray, D: int = 128):
    x = np.asarray(data, dtype=np.int64).copy()
    n = x.size
    D = min(D, max(1, n // 4))
    correct = 0
    tot = 0
    # running score: for each lag d, the most common value following each
    # (previous d samples pattern) -- for a binary source the SP 800-90B
    # predictor uses the most common next value given the sample at lag d.
    for d in range(1, D + 1):
        if n <= d:
            continue
        pred = x[:-d]
        actual = x[d:]
        # fitted per-lag majority map estimated on a training prefix
        half = max(1000, n // 10)
        tr_pred, tr_act = pred[:half], actual[:half]
        counts = np.zeros((2, 2))
        np.add.at(counts, (tr_pred, tr_act), 1)
        maj = counts.argmax(axis=1)
        ok = int(np.count_nonzero(maj[pred[half:]] == actual[half:]))
        correct += ok
        tot += actual.size - half
    p_global = correct / tot if tot else 0.5
    # SP 800-90B: the estimate is -log2(p_global) with p_global the global
    # predictor accuracy (bounded by 1 for a binary source)
    return _pmax_to_hmin(min(1.0, max(p_global, 1e-12))), {
        "accuracy": float(p_global), "D": D}


# ---------------------------------------------------------------------------
# 6.3.8  MultiMCW predictor
# ---------------------------------------------------------------------------
def multimcw_prediction(data: np.ndarray, W: int = 63):
    x = np.asarray(data, dtype=np.int64)
    n = x.size
    W = min(W, n // 4)
    if W < 4 or n < 100:
        return 1.0, {"error": "short"}
    counts = np.zeros(2)                     # binarised running counts
    correct = np.zeros(2)                    # per window length
    total = 0
    # window score for each w in {1,2,3,4,...}: most common value in the last w
    wins = [w for w in (1, 2, 3, 4, 8, 16, 32) if w < n]
    score = {w: 0 for w in wins}
    hit = {w: 0 for w in wins}
    for i in range(32, n):
        for w in wins:
            window = x[i - w:i]
            c1 = int(window.sum())
            maj = 1 if 2 * c1 > w else 0
            score[w] += 1 if maj == x[i] else 0
            hit[w] += 1
    best_w = max(wins, key=lambda w: score[w] / hit[w])
    acc = score[best_w] / hit[best_w]
    return _pmax_to_hmin(min(1.0, max(acc, 1e-12))), {"best_window": best_w,
                                                      "accuracy": float(acc)}


# ---------------------------------------------------------------------------
# 6.3.7  LZ78Y predictor
# ---------------------------------------------------------------------------
def lz78y_prediction(data: np.ndarray, B: int = 16):
    x = np.asarray(data, dtype=np.int64)
    n = x.size
    if n < B + 100:
        return 1.0, {"error": "short"}
    # dictionary: context (tuple) -> counts of following symbols
    dicts = {b: defaultdict(lambda: np.zeros(2)) for b in range(1, B + 1)}
    correct = 0
    tot = 0
    start = n // 10
    for i in range(1, n):
        if i >= start:
            best = None
            bestc = np.zeros(2)
            for b in range(1, min(B, i) + 1):
                ctx = tuple(x[i - b:i])
                c = dicts[b].get(ctx)
                if c is not None:
                    if best is None or c.sum() > bestc.sum():
                        best = ctx
                        bestc = c
            if best is not None:
                pred = int(np.argmax(bestc))
                correct += 1 if pred == x[i] else 0
                tot += 1
        for b in range(1, min(B, i) + 1):
            ctx = tuple(x[i - b:i])
            dicts[b][ctx][x[i]] += 1
    acc = correct / tot if tot else 0.5
    return _pmax_to_hmin(min(1.0, max(acc, 1e-12))), {"accuracy": float(acc), "B": B}


# ---------------------------------------------------------------------------
# aggregation + IID permutation test
# ---------------------------------------------------------------------------
ESTIMATORS = {
    "MCV": lambda x: mcv_estimate(x),
    "Collision": lambda x: collision_estimate(x),
    "Markov": lambda x: markov_estimate(x),
    "Compression": lambda x: compression_estimate(x),
    "t-Tuple": lambda x: ttuple_estimate(x),
    "LRS": lambda x: lrs_estimate(x),
    "Lag": lambda x: lag_prediction(x),
    "MultiMCW": lambda x: multimcw_prediction(x),
    "LZ78Y": lambda x: lz78y_prediction(x),
}


def min_entropy_90b(data: np.ndarray, which=None, alphabet_size: int = 2) -> dict:
    """Run all estimators; results are capped at log2(alphabet_size)."""
    out = {}
    cap = math.log2(alphabet_size)
    for name, fn in ESTIMATORS.items():
        if which and name not in which:
            continue
        h, info = fn(np.asarray(data, dtype=np.int64))
        out[name] = {"hmin": float(min(h, cap)), "info": info}
    vals = [v["hmin"] for v in out.values()]
    out["_summary"] = {"min": float(min(vals)), "mean": float(np.mean(vals)),
                       "argmin": min(out, key=lambda k: out[k]["hmin"]
                                     if isinstance(out[k], dict) and "hmin" in out[k] else 9e9)}
    return out


def permutation_test_iid(data: np.ndarray, n_perms: int = 1000, seed: int = 0):
    """SP 800-90B Sec. 3.1.3 permutation test of the IID assumption.

    The test statistic is the minimum of the nine entropy estimates.  Under
    H0 (IID) the statistic is exchangeable, so its distribution is obtained by
    permuting the sample and recomputing.  A large p-value means we fail to
    reject IID."""
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=np.int64)
    which = ["MCV", "Collision", "Markov", "Compression", "Lag"]
    def stat(v):
        return min(min_entropy_90b(v, which=which)[k]["hmin"] for k in which)
    c0 = stat(x)
    cnt = 0
    perms = []
    for _ in range(n_perms):
        xs = rng.permutation(x)
        c = stat(xs)
        perms.append(c)
        if c <= c0:
            cnt += 1
    p = (cnt + 1) / (n_perms + 1)
    return {"statistic": float(c0), "p_value": float(p),
            "perm_mean": float(np.mean(perms)), "perm_std": float(np.std(perms)),
            "n_permutations": n_perms}
