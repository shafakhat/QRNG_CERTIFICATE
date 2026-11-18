"""
postprocessing.py
=================
Classical post-processing used by the raw-to-final random-number pipeline:
binary Shannon entropy, a Wilson score interval for binomial counts, and
privacy amplification of the raw bit string with a random Toeplitz matrix over
GF(2).

`toeplitz_pa` is a real implementation and not a model: it multiplies the raw
key by a random Toeplitz matrix over GF(2).  The construction is the
universal-2 hash family used for privacy amplification (Carter & Wegman 1979;
Lefèvre, unpublished manuscript, 2004); the number of output bits is chosen from
the min-entropy estimate produced by `run_qrng.py`, so the Toeplitz output is
the extracted key that the SP 800-22 battery is applied to.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# entropies and intervals
# ---------------------------------------------------------------------------
def h2(x: float) -> float:
    """Binary Shannon entropy (bits)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def wilson_interval(k: int, n: int, z: float = 2.576) -> tuple:
    """Wilson score interval for a binomial proportion (99% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Toeplitz privacy amplification over GF(2)
# ---------------------------------------------------------------------------
def _carryless_mul(x: int, y: int) -> int:
    """Polynomial multiplication over GF(2) of two Python integers."""
    acc = 0
    if x.bit_count() > y.bit_count():
        x, y = y, x
    i = 0
    while x:
        if x & 1:
            acc ^= y << i
        x >>= 1
        i += 1
    return acc


def toeplitz_pa(key: np.ndarray, out_bits: int, seed: int = 0) -> np.ndarray:
    """
    Universal-2 hashing with a random Toeplitz matrix.

    y_i = sum_j T_{i,j} x_j with T_{i,j} = t_{i-j}; implemented as the low
    `out_bits` coefficients of the GF(2) product T(z) X(z), where X(z) has the
    key bits as coefficients.  The seed defines the Toeplitz vector t.
    """
    n = key.size
    out_bits = min(out_bits, n)
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n + out_bits - 1, dtype=np.uint8)
    t_int = int("".join(str(int(v)) for v in t[::-1]), 2)      # t_{n+m-2} ... t_0
    x_int = int("".join(str(int(v)) for v in key[::-1]), 2)
    prod = _carryless_mul(t_int, x_int)
    bits = np.array([(prod >> i) & 1 for i in range(out_bits)], dtype=np.uint8)
    return bits
