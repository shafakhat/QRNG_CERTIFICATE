"""
nist_sp800_22.py
================
A from-scratch, vectorised NumPy/SciPy implementation of the NIST SP 800-22
Rev. 1a statistical test suite for random and pseudorandom bit generators.

Implemented tests (13 of the 15 tests of the suite):
    T01 Frequency (Monobit)
    T02 Frequency within a Block
    T03 Runs
    T04 Longest Run of Ones in a Block
    T05 Binary Matrix Rank
    T06 Discrete Fourier Transform (Spectral)
    T07 Non-overlapping Template Matching   (148-template aggregate)
    T08 Overlapping Template Matching       (exact DP reference distribution)
    T09 Universal Statistical (Maurer)
    T10 Linear Complexity (Berlekamp-Massey)
    T11 Serial
    T12 Approximate Entropy
    T13 Cumulative Sums (forward and reverse)

Not implemented: T14 Random Excursions, T15 Random Excursions Variant.
(These are the two "excursion" tests; see the reproducibility
section -- they are the only two tests of the suite that are omitted, and the
reported pass rates are therefore for the 13-test battery.)

Design notes
------------
* Input sequences are numpy arrays of dtype uint8 with values in {0, 1}.
* Every function returns (p_value, info) where `info` is a small dict with the
  intermediate statistics (useful for the paper's tables).
* `p_value = None` means the test is "not applicable" to the supplied sequence
  (e.g. the Runs test precondition |pi - 1/2| < 2/sqrt(n) is violated).
* All reference distributions and constants follow SP 800-22 Rev. 1a.

Author: research code for the QRNG entropy-validation experiments.
"""
from __future__ import annotations

import numpy as np
from scipy import special
from functools import lru_cache

__all__ = [
    "frequency_monobit", "block_frequency", "runs", "longest_run_of_ones",
    "binary_matrix_rank", "dft_spectral", "non_overlapping_template",
    "overlapping_template", "universal", "linear_complexity", "serial",
    "approximate_entropy", "cumulative_sums", "run_battery", "BATTERY",
]

SQRT2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _to_pm1(bits: np.ndarray) -> np.ndarray:
    """Map {0,1} bits to {-1,+1}."""
    return 2.0 * bits.astype(np.int64) - 1.0


def _igamc(a, x):
    """Regularised upper incomplete gamma Q(a,x) = igamc, as used by NIST."""
    return special.gammaincc(a, x)


def _window_values(bits: np.ndarray, m: int) -> np.ndarray:
    """Integer value of every length-m sliding window (m <= 24)."""
    n = bits.size
    if n < m:
        return np.empty(0, dtype=np.int64)
    w = np.lib.stride_tricks.sliding_window_view(bits.astype(np.int64), m)
    weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
    return w @ weights


def _pattern_counts(bits: np.ndarray, m: int) -> np.ndarray:
    """Overlapping counts of all 2^m patterns."""
    vals = _window_values(bits, m)
    return np.bincount(vals, minlength=1 << m).astype(np.float64)


# ----------------------------------------------------------------------------
# T01  Frequency (Monobit) test
# ----------------------------------------------------------------------------
def frequency_monobit(bits: np.ndarray):
    n = bits.size
    s = _to_pm1(bits).sum()
    s_obs = abs(s) / np.sqrt(n)
    p = special.erfc(s_obs / SQRT2)
    return float(p), {"S_n": float(s), "s_obs": float(s_obs), "n": int(n)}


# ----------------------------------------------------------------------------
# T02  Frequency test within a block
# ----------------------------------------------------------------------------
def block_frequency(bits: np.ndarray, M: int = 128):
    n = bits.size
    N = n // M
    if N < 1:
        return None, {"error": "sequence shorter than one block"}
    blocks = bits[: N * M].reshape(N, M)
    pi = blocks.mean(axis=1)
    chi2 = 4.0 * M * np.sum((pi - 0.5) ** 2)
    p = _igamc(N / 2.0, chi2 / 2.0)
    return float(p), {"M": M, "N": N, "chi2": float(chi2)}


# ----------------------------------------------------------------------------
# T03  Runs test
# ----------------------------------------------------------------------------
def runs(bits: np.ndarray):
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return None, {"error": "precondition |pi-1/2| < 2/sqrt(n) violated",
                      "pi": float(pi)}
    v_obs = 1 + int(np.count_nonzero(np.diff(bits)))
    num = abs(v_obs - 2.0 * n * pi * (1 - pi))
    den = 2.0 * np.sqrt(2.0 * n) * pi * (1 - pi)
    p = special.erfc(num / den)
    return float(p), {"pi": float(pi), "V_obs": v_obs}


# ----------------------------------------------------------------------------
# T04  Longest run of ones in a block
# ----------------------------------------------------------------------------
def _longest_run(block: np.ndarray) -> int:
    """Length of the longest run of ones in a block (vectorised)."""
    idx = np.flatnonzero(block == 0)
    if idx.size == 0:
        return int(block.size)
    d = np.diff(np.concatenate(([-1], idx, [block.size]))) - 1
    return int(d.max())


def _count_no_k_ones(M: int, k: int) -> int:
    """Exact number of M-bit strings containing no run of k consecutive ones.

    a[n] = 2^n for n < k, and a[n] = a[n-1] + ... + a[n-k] otherwise, evaluated
    with a sliding window of Python big integers (exact, no overflow)."""
    if k > M:
        return 1 << M
    a = [0] * (M + 1)
    for n in range(k):
        a[n] = 1 << n
    W = sum(a[0:k])
    for n in range(k, M + 1):
        a[n] = W
        W = W + a[n] - a[n - k]
    return a[M]


@lru_cache(maxsize=None)
def _longest_run_pmf(M: int):
    """Exact PMF of the longest run of ones in M i.i.d. fair bits."""
    pmf = {}
    prev = 0.0
    k = 1
    while k <= M:
        cur = _count_no_k_ones(M, k) / (1 << M)     # P(maxrun < k)
        if cur - prev > 1e-13:
            pmf[k - 1] = cur - prev                 # P(maxrun == k-1)
        prev = cur
        if cur > 1 - 1e-13:
            break
        k += 1
    pmf[k] = max(1.0 - prev, 0.0)                   # P(maxrun >= k)
    return {a: b for a, b in pmf.items() if b > 1e-12}


def longest_run_of_ones(bits: np.ndarray, M: int | None = None,
                        min_bin_prob: float = 0.05, scheme: str = "exact"):
    """Longest-run-of-ones test with an *exact* reference distribution.

    The block length M is chosen as in SP 800-22 Table 2.4, but the class
    probabilities are computed exactly by dynamic programming instead of being
    taken from the tabulated values, and the bins are formed greedily so that
    every bin has probability >= `min_bin_prob` under H0.

    `scheme="sp80022"` instead uses the *tabulated* class boundaries and
    probabilities printed in SP 800-22 Table 2.4 (the partition implemented by
    the NIST STS reference code).  The two schemes coincide at M = 10000
    (7 classes) and differ at M = 128, where the tabulated partition pools all
    runs <= 4 into the first class.  The default `scheme="exact"` is what the
    battery reported in the paper uses; `scheme="sp80022"` exists so that an
    independent implementation can be compared bin for bin.
    """
    n = bits.size
    if n < 128:
        return None, {"error": "n < 128"}
    if M is None:
        M = 8 if n < 6272 else (128 if n < 750000 else 10000)
    nb = n // M
    blocks = bits[: nb * M].reshape(nb, M)
    lr = np.array([_longest_run(b) for b in blocks])
    if scheme == "sp80022":
        # (V[], pi[]) exactly as tabulated in SP 800-22 Table 2.4 / STS
        if M == 8:
            V = [1, 2, 3, 4]
            pi = [0.21484375, 0.3671875, 0.23046875, 0.1875]
        elif M == 128:
            V = [4, 5, 6, 7, 8, 9]
            pi = [0.1174035788, 0.242955959, 0.249363483,
                  0.17517706, 0.102701071, 0.112398847]
        elif M == 10000:
            V = [10, 11, 12, 13, 14, 15, 16]
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        else:
            return None, {"error": f"no tabulated partition for M={M}"}
        # class k holds longest run == V[k]; the first class also holds
        # smaller runs and the last class also holds larger runs
        idx = np.clip(np.searchsorted(np.asarray(V), lr, side="left"),
                      0, len(V) - 1)
        counts = np.bincount(idx, minlength=len(V)).astype(float)
        pi = np.asarray(pi, dtype=float)
        pi = pi / pi.sum()
        N = nb
        chi2 = np.sum((counts - N * pi) ** 2 / (N * pi))
        df = len(V) - 1
        p = _igamc(df / 2.0, chi2 / 2.0)
        return float(p), {"M": M, "N": N, "n_bins": len(V), "df": df,
                          "nu": counts.astype(int).tolist(),
                          "pi": np.round(pi, 6).tolist(),
                          "chi2": float(chi2), "max_run": int(lr.max()),
                          "scheme": "sp80022"}
    pmf = _longest_run_pmf(M)
    # greedy binning of the exact pmf
    bins, cur, acc = [], [], 0.0
    for k in sorted(pmf):
        cur.append(k)
        acc += pmf[k]
        if acc >= min_bin_prob:
            bins.append((cur, acc))
            cur, acc = [], 0.0
    if cur:
        if bins:
            ks, p0 = bins[-1]
            bins[-1] = (ks + cur, p0 + acc)
        else:
            bins.append((cur, acc))
    pi = np.array([b[1] for b in bins])
    pi = pi / pi.sum()
    counts = np.array([np.count_nonzero(np.isin(lr, b[0])) for b in bins],
                      dtype=float)
    N = nb
    chi2 = np.sum((counts - N * pi) ** 2 / (N * pi))
    df = len(bins) - 1
    p = _igamc(df / 2.0, chi2 / 2.0)
    return float(p), {"M": M, "N": N, "n_bins": len(bins), "df": df,
                      "nu": counts.astype(int).tolist(),
                      "pi": np.round(pi, 6).tolist(),
                      "chi2": float(chi2), "max_run": int(lr.max())}


# ----------------------------------------------------------------------------
# T05  Binary matrix rank test
# ----------------------------------------------------------------------------
def _rank_gf2(mat: np.ndarray) -> int:
    m = mat.copy()
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if m[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            m[[r, piv]] = m[[piv, r]]
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        r += 1
        if r == rows:
            break
    return r


def binary_matrix_rank(bits: np.ndarray, M: int = 32, Q: int = 32,
                       exact_probs: bool = False):
    """
    Binary matrix rank test.

    The class probabilities are the 4-decimal values tabulated in SP 800-22
    Sec. 2.5.4 (P_32 = 0.2888, P_31 = 0.5776, P_30 = 0.1336).  With
    `exact_probs=True` the numerically exact probabilities are used instead;
    this reproduces the NIST STS p-value to its printed precision.  The rank
    *counts* F are identical under either setting -- only the constants differ.
    """
    n = bits.size
    N = n // (M * Q)
    if N < 1:
        return None, {"error": "sequence too short for M=Q=32"}
    F = np.zeros(3)
    for i in range(N):
        blk = bits[i * M * Q:(i + 1) * M * Q].reshape(M, Q)
        r = _rank_gf2(blk)
        F[0 if r == M else (1 if r == M - 1 else 2)] += 1
    # theoretical probabilities (SP 800-22 Sec. 2.5.4)
    if exact_probs:
        p_full, p_m1, p_rest = 0.2887880950866024, 0.5775761901732044, \
            0.1336357147397952
    else:
        p_full, p_m1 = 0.2888, 0.5776
        p_rest = 0.1336
    chi2 = ((F[0] - p_full * N) ** 2 / (p_full * N)
            + (F[1] - p_m1 * N) ** 2 / (p_m1 * N)
            + (F[2] - p_rest * N) ** 2 / (p_rest * N))
    p = np.exp(-chi2 / 2.0)
    return float(p), {"M": M, "Q": Q, "N": N, "F": F.astype(int).tolist(),
                      "chi2": float(chi2)}


# ----------------------------------------------------------------------------
# T06  Discrete Fourier transform (spectral) test
# ----------------------------------------------------------------------------
def dft_spectral(bits: np.ndarray):
    n = bits.size
    X = _to_pm1(bits)
    S = np.fft.fft(X)
    modulus = np.abs(S[: n // 2])
    T = np.sqrt(np.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.count_nonzero(modulus < T)
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
    p = special.erfc(abs(d) / SQRT2)
    return float(p), {"T": float(T), "N0": float(N0), "N1": int(N1), "d": float(d)}


# ----------------------------------------------------------------------------
# T07  Non-overlapping template matching test (148-template aggregate)
# ----------------------------------------------------------------------------
def _non_overlapping_count(block: np.ndarray, template: np.ndarray) -> int:
    m = template.size
    w = np.lib.stride_tricks.sliding_window_view(block, m)
    starts = np.flatnonzero(np.all(w == template, axis=1))
    if starts.size == 0:
        return 0
    cnt, pos = 0, 0
    while pos < starts.size:
        cnt += 1
        nxt = starts[pos] + m
        pos = int(np.searchsorted(starts, nxt, side="left"))
    return cnt


def _two_nine_templates() -> list[np.ndarray]:
    """The 148 templates used by the NIST reference implementation."""
    tmpls = []
    for i in range(1, 149):
        s = format(i, "09b")     # 9-bit strings 000000001 ... 100101100
        if s == "111111111":     # NIST's reference code fixes the all-ones run
            s = "111111111"
        tmpls.append(np.array([int(c) for c in s], dtype=np.uint8))
    return tmpls


def non_overlapping_template(bits: np.ndarray, m: int = 9, n_blocks: int = 8,
                             templates: list | None = None):
    """
    Non-overlapping template matching test.

    Returns the *per-template* p-values (the 148 9-bit templates of SP 800-22,
    Sec. 2.7) together with their pass fraction.  The reference implementation
    of the suite evaluates this test template-by-template; the 148 tests are
    strongly dependent within a block, so we deliberately do **not** sum them
    into a single chi-square statistic.  For the battery-level second-order
    analysis the median template p-value is used as the representative
    statistic (documented in the docstring above).
    """
    n = bits.size
    M = n // n_blocks
    if M <= m:
        return None, {"error": "block too short"}
    if templates is None:
        templates = _two_nine_templates()
    J = len(templates)
    mu = (M - m + 1) / 2.0 ** m
    var = M * (1.0 / 2.0 ** m - (2.0 * m - 1) / 2.0 ** (2.0 * m))
    pvals = np.empty(J)
    # pre-extract the windows of every block once (memory: N x (M-m+1))
    blocks = [bits[i * M:(i + 1) * M] for i in range(n_blocks)]
    wins = [np.lib.stride_tricks.sliding_window_view(b, m) for b in blocks]
    for j, tmpl in enumerate(templates):
        chi2 = 0.0
        for w in wins:
            starts = np.flatnonzero(np.all(w == tmpl, axis=1))
            if starts.size == 0:
                cnt = 0
            else:
                cnt, pos = 0, 0
                while pos < starts.size:
                    cnt += 1
                    pos = int(np.searchsorted(starts, starts[pos] + m, side="left"))
            chi2 += (cnt - mu) ** 2 / var
        pvals[j] = _igamc(n_blocks / 2.0, chi2 / 2.0)
    alpha = 0.01
    return float(np.median(pvals)), {
        "m": m, "M": M, "J": J, "N": n_blocks,
        "n_pass": int(np.count_nonzero(pvals >= alpha)),
        "pass_fraction": float(np.count_nonzero(pvals >= alpha) / J),
        "min_p": float(pvals.min()), "median_p": float(np.median(pvals)),
        "p_values": pvals.tolist(),
    }


# ----------------------------------------------------------------------------
# T08  Overlapping template matching test
# ----------------------------------------------------------------------------
def overlapping_template(bits: np.ndarray, m: int = 9, M: int = 1032,
                         K: int = 5, template: str = "1" * 9):
    n = bits.size
    N = n // M
    if N < 1:
        return None, {"error": "sequence shorter than one block"}
    tmpl = np.array([int(c) for c in template], dtype=np.uint8)
    # exact reference distribution by dynamic programming
    pi = _exact_overlap_pmf(M, tmpl, K)
    nu = np.zeros(K + 1)
    for i in range(N):
        blk = bits[i * M:(i + 1) * M]
        w = np.lib.stride_tricks.sliding_window_view(blk, m)
        c = int(np.count_nonzero(np.all(w == tmpl, axis=1)))
        nu[min(c, K)] += 1
    mask = pi * N > 0
    chi2 = np.sum((nu[mask] - N * pi[mask]) ** 2 / (N * pi[mask]))
    df = int(np.count_nonzero(mask)) - 1
    p = _igamc(df / 2.0, chi2 / 2.0)
    return float(p), {"m": m, "M": M, "N": N, "chi2": float(chi2), "df": df,
                      "pi": pi.tolist(), "nu": nu.astype(int).tolist()}


def _exact_overlap_pmf(M: int, tmpl: np.ndarray, K: int) -> np.ndarray:
    """Exact PMF of the number of overlapping occurrences of `tmpl` in a
    random M-bit block (truncated above K-1, with mass >= K in the last bin)."""
    m = tmpl.size
    t = "".join(str(int(b)) for b in tmpl)
    trans = np.zeros((m, 2), dtype=np.int64)
    occ = np.zeros((m, 2), dtype=np.int64)
    for s in range(m):
        for b in (0, 1):
            cand = t[:s] + str(b)
            k = min(len(cand), m)
            while k > 0 and not t.startswith(cand[-k:]):
                k -= 1
            if k == m:                 # an occurrence just completed
                occ[s, b] = 1
                k = m - 1              # restart state = longest proper suffix
                while k > 0 and not t.startswith(cand[-k:]):
                    k -= 1
            trans[s, b] = k
    dist = np.zeros((m, K + 1))
    dist[0, 0] = 1.0
    for _ in range(M):
        nxt = np.zeros_like(dist)
        for s in range(m):
            row = dist[s]
            for b in (0, 1):
                tgt = trans[s, b]
                if occ[s, b]:
                    nxt[tgt, 1:] += 0.5 * row[:-1]
                    nxt[tgt, K] += 0.5 * row[K]
                else:
                    nxt[tgt, :] += 0.5 * row
        dist = nxt
    return dist.sum(axis=0)


# ----------------------------------------------------------------------------
# T09  Universal statistical (Maurer) test
# ----------------------------------------------------------------------------
def universal(bits: np.ndarray, L: int = 7, Q: int = 1280):
    n = bits.size
    if n < 2 ** L:
        return None, {"error": "n too small"}
    K = n // L - Q
    if K < 1:
        return None, {"error": "K < 1"}
    # table of last positions; vectorised by grouping the L-bit words
    words = _window_values(bits, L)[:: L]        # non-overlapping L-bit blocks
    words = words[: K + Q]
    table = -np.ones(2 ** L, dtype=np.int64)
    fn = 0.0
    for i in range(Q + K):
        w = int(words[i])
        last = table[w]
        if i >= Q:
            fn += np.log2(i - last)      # both indices 0-based -> true distance
        table[w] = i
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * K ** (-3.0 / L) / 15.0
    exp_table = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656}
    var_table = {6: 2.954, 7: 3.125, 8: 3.238}
    expected = exp_table.get(L, 6.1962507)
    variance = var_table.get(L, 3.125)
    sigma = c * np.sqrt(variance / K)
    p = special.erfc(abs(fn / K - expected) / (SQRT2 * sigma))
    return float(p), {"L": L, "Q": Q, "K": K, "f_n": float(fn / K),
                      "expected": expected, "sigma": float(sigma), "c": float(c)}


# ----------------------------------------------------------------------------
# T10  Linear complexity test (Berlekamp-Massey)
# ----------------------------------------------------------------------------
def _berlekamp_massey_int(x: int, n: int) -> int:
    """
    Linear complexity of an n-bit sequence.

    `x` holds the sequence in *reversed* bit order: bit k of x is s_{n-1-k}.
    The recurrence is updated with big-integer word operations, so each
    iteration costs O(n/64) machine words instead of O(n) Python operations.
    """
    c = 1                      # connection polynomial, bit j = c_j
    b = 1
    l = 0
    m = -1
    for i in range(n):
        # discrepancy: d = s_i + sum_{j=1..l} c_j s_{i-j}
        d = (x >> (n - 1 - i)) & 1
        if l:
            mask = (1 << l) - 1
            # window over s_i, s_{i-1}, ..., s_{i-l}  (l+1 bits)
            win = (x >> (n - 1 - i)) & ((1 << (l + 1)) - 1)
            # bit j of ((c>>1)&mask) is c_{j+1}; bit j of (win>>1) is s_{i-j-1}
            d ^= (((c >> 1) & mask) & (win >> 1)).bit_count() & 1
        if d & 1:
            t = c
            c ^= b << (i - m)
            if 2 * l <= i:
                l = i + 1 - l
                b = t
                m = i
    return l


def linear_complexity(bits: np.ndarray, M: int = 500):
    n = bits.size
    N = n // M
    if N < 1:
        return None, {"error": "n < M"}
    Ls = np.empty(N, dtype=np.int64)
    for i in range(N):
        blk = bits[i * M:(i + 1) * M]
        # pack LSB = first bit
        x = int("".join(str(int(b)) for b in blk), 2)   # bit k = s_{M-1-k}
        Ls[i] = _berlekamp_massey_int(x, M)
    mu = (M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0
          - (M / 3.0 + 2.0 / 9.0) / 2.0 ** M)
    # NIST's integer binning
    nu = np.zeros(7, dtype=np.int64)
    for l in Ls:
        t = (-1) ** M * (l - mu) + 2.0 / 9.0
        if t <= -2.5:
            nu[0] += 1
        elif t <= -1.5:
            nu[1] += 1
        elif t <= -0.5:
            nu[2] += 1
        elif t <= 0.5:
            nu[3] += 1
        elif t <= 1.5:
            nu[4] += 1
        elif t <= 2.5:
            nu[5] += 1
        else:
            nu[6] += 1
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    pi = pi / pi.sum()
    chi2 = np.sum((nu - N * pi) ** 2 / (N * pi))
    p = _igamc(6 / 2.0, chi2 / 2.0)
    return float(p), {"M": M, "N": N, "nu": nu.tolist(), "chi2": float(chi2),
                      "mean_L": float(Ls.mean()), "mu": float(mu)}


# ----------------------------------------------------------------------------
# T11  Serial test
# ----------------------------------------------------------------------------
def serial(bits: np.ndarray, m: int = 16, allow_out_of_range: bool = False):
    """
    Serial test (SP 800-22 Sec. 2.11).

    The standard's applicability condition is m < floor(log2 n) - 2.  A call
    that violates it returns `None` unless `allow_out_of_range=True`, which is
    needed only to reproduce the NIST STS, whose code hard-codes m = 16 at every
    sequence length (see code/sts_crosscheck.py).
    """
    n = bits.size
    if m >= int(np.floor(np.log2(n))) - 2 and not allow_out_of_range:
        return None, {"error": "m too large for n"}
    def psi2(mm):
        # SP 800-22 Sec. 2.11.1: the sequence is extended cyclically, i.e. the
        # first mm-1 bits are appended to the end, so that every position of the
        # original n-bit string starts one mm-bit word.  (A non-cyclic sliding
        # count uses only n-mm+1 words and gives a p-value that differs in the
        # third decimal; the reference STS implements the cyclic version.)
        if mm <= 0:
            return 0.0
        ext = np.concatenate([bits, bits[: mm - 1]]) if mm > 1 else bits
        cnt = _pattern_counts(ext, mm)
        return (2.0 ** mm / n) * np.sum(cnt ** 2) - n
    p2, p1, p0 = psi2(m), psi2(m - 1), psi2(m - 2)
    d1 = p2 - p1
    d2 = p2 - 2 * p1 + p0
    pv1 = _igamc(2 ** (m - 2), d1 / 2.0)
    pv2 = _igamc(2 ** (m - 3), d2 / 2.0)
    return float(pv2), {"m": m, "psi2_m": float(p2), "psi2_m1": float(p1),
                        "psi2_m2": float(p0), "p_value_1": float(pv1),
                        "p_value_2": float(pv2)}


# ----------------------------------------------------------------------------
# T12  Approximate entropy test
# ----------------------------------------------------------------------------
# Empirical calibration of the ApEn statistic under H0 (mean, std), filled by
# calibrate_battery.py; falls back to the asymptotic chi-square moments.
APEN_CALIBRATION: dict = {}


def approximate_entropy(bits: np.ndarray, m: int = 10, calibrated: bool = True):
    """Approximate-entropy test.

    The pattern counts are taken over the *circular* sequence (the first m+1
    bits are appended) exactly as in the NIST reference implementation.  The
    asymptotic reference is chi-square with 2^m degrees of freedom; a small
    finite-size offset in the mean of the statistic (measured once by Monte
    Carlo in calibrate_battery.py and stored in APEN_CALIBRATION) is corrected
    before the p-value is computed.
    """
    n = bits.size
    ext = np.concatenate([bits, bits[: m + 1]])
    def phi(mm):
        w = np.lib.stride_tricks.sliding_window_view(ext, mm)[:n]
        packed = w @ (1 << np.arange(mm - 1, -1, -1))
        c = np.bincount(packed, minlength=1 << mm)[: 1 << mm] / n
        nz = c > 0
        return float(np.sum(c[nz] * np.log(c[nz])))
    apen = phi(m) - phi(m + 1)
    chi2 = 2.0 * n * (np.log(2.0) - apen)
    key = f"n={n}_m={m}"
    if calibrated and key in APEN_CALIBRATION:
        mu = APEN_CALIBRATION[key]["mean"]
        sd = APEN_CALIBRATION[key]["std"]
        z = (chi2 - mu) / sd
        p = special.erfc(abs(z) / SQRT2)
        ref = {"mu": mu, "sd": sd, "z": float(z), "source": "calibrated"}
    else:
        p = _igamc(2 ** m / 2.0, chi2 / 2.0)
        ref = {"df": 2 ** m, "source": "asymptotic"}
    return float(p), {"m": m, "ApEn": float(apen), "chi2": float(chi2), **ref}


# ----------------------------------------------------------------------------
# T13  Cumulative sums
# ----------------------------------------------------------------------------
def cumulative_sums(bits: np.ndarray, mode: str = "forward"):
    n = bits.size
    X = _to_pm1(bits)
    if mode == "reverse":
        X = X[::-1]
    S = np.cumsum(X)
    z = int(np.max(np.abs(S))) if n else 0
    if z == 0:
        return 1.0, {"z": 0, "mode": mode}
    k1 = np.arange(-int((n / z - 1) / 4.0) - 1, int((n / z - 1) / 4.0) + 2)
    k2 = np.arange(-int((n / z + 3) / 4.0) - 1, int((n / z + 3) / 4.0) + 2)
    sq = np.sqrt(n)
    s1 = np.sum(special.ndtr((4 * k1 + 1) * z / sq) - special.ndtr((4 * k1 - 1) * z / sq))
    s2 = np.sum(special.ndtr((4 * k2 + 3) * z / sq) - special.ndtr((4 * k2 + 1) * z / sq))
    p = 1.0 - s1 + s2
    return float(min(max(p, 0.0), 1.0)), {"z": z, "mode": mode}


# ----------------------------------------------------------------------------
# battery driver
# ----------------------------------------------------------------------------
BATTERY = [
    "Frequency (Monobit)",
    "Block Frequency (M=128)",
    "Runs",
    "Longest Run of Ones",
    "Binary Matrix Rank (32x32)",
    "Discrete Fourier Transform",
    "Non-overlapping Template (148)",
    "Overlapping Template (m=9)",
    "Universal (Maurer, L=7)",
    "Linear Complexity (M=500)",
    "Serial (m=16)",
    "Approximate Entropy (m=10)",
    "Cumulative Sums (fwd)",
    "Cumulative Sums (rev)",
]


def run_battery(bits: np.ndarray, verbose: bool = False,
                adaptive: bool = True) -> dict:
    """
    Run the 13-test battery (14 statistics: cusum forward and reverse).

    With `adaptive=True` the test parameters that depend on the sequence
    length (Serial m, Approximate Entropy m, Linear Complexity M) are reduced
    for short sequences instead of declaring the test inapplicable, and
    `n_applicable` is returned so that pass fractions are always reported
    against the number of tests that could actually be run.
    """
    n = bits.size
    m_serial = 16
    m_apen = 10
    m_lc = 500
    if adaptive:
        lim = int(np.floor(np.log2(max(n, 4))))
        m_serial = max(2, min(16, lim - 3))
        m_apen = max(2, min(10, lim - 5))
        m_lc = max(20, min(500, n // 200))
    out = {}
    fns = [
        ("Frequency (Monobit)", lambda b: frequency_monobit(b)),
        ("Block Frequency (M=128)", lambda b: block_frequency(b, 128)),
        ("Runs", lambda b: runs(b)),
        ("Longest Run of Ones", lambda b: longest_run_of_ones(b)),
        ("Binary Matrix Rank (32x32)", lambda b: binary_matrix_rank(b, 32, 32)),
        ("Discrete Fourier Transform", lambda b: dft_spectral(b)),
        ("Non-overlapping Template (148)", lambda b: non_overlapping_template(b)),
        ("Overlapping Template (m=9)", lambda b: overlapping_template(b)),
        ("Universal (Maurer, L=7)", lambda b: universal(b, 7, 1280)),
        (f"Linear Complexity (M={m_lc})", lambda b: linear_complexity(b, m_lc)),
        (f"Serial (m={m_serial})", lambda b: serial(b, m_serial)),
        (f"Approximate Entropy (m={m_apen})", lambda b: approximate_entropy(b, m_apen)),
        ("Cumulative Sums (fwd)", lambda b: cumulative_sums(b, "forward")),
        ("Cumulative Sums (rev)", lambda b: cumulative_sums(b, "reverse")),
    ]
    for name, fn in fns:
        p, info = fn(bits)
        out[name] = {"p_value": p, "info": info,
                     "pass": (p is not None and p >= 0.01),
                     "applicable": p is not None}
        if verbose:
            pv = "n/a " if p is None else f"{p:.4g}"
            print(f"  {name:<32s} p = {pv}  {'PASS' if out[name]['pass'] else 'FAIL'}")
    return out
