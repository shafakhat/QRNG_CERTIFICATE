"""
failure_modes.py
================
F9: negative controls for the QRNG validation pipeline.

A statistical test battery is only credible if it is shown to *reject* flawed
sources.  We therefore simulate four physically motivated imperfections of a
rotation-based generator and record which statistics detect them:

  (a) asymmetric readout      - p(1|0) != p(0|1)  (detector imbalance)
  (b) amplitude damping (T1)  - relaxation towards |0> after preparation
  (c) slow pulse drift        - theta(t) = pi/2 + eps sin(2 pi f t) with
                                f small enough that consecutive samples are
                                correlated (classical common-mode drift)
  (d) phase damping (T2)      - pure dephasing of the prepared state

For each case we report the measured bias, the min-entropy, and the list of
NIST SP 800-22 statistics that fail at alpha = 0.01.
"""
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, ReadoutError, amplitude_damping_error,
                              phase_damping_error)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as C
import nist_sp800_22 as nist
import postprocessing as P

BATCH = C.N_BATCH
DM_BATCH = 6        # density-matrix registers must stay small (memory)


# ---------------------------------------------------------------------------
# samplers for the four failure modes
# ---------------------------------------------------------------------------
def sample_asymmetric_readout(n_bits, p10=0.02, p01=0.06, seed=None):
    """p10 = P(read 1 | true 0), p01 = P(read 0 | true 1)."""
    nm = NoiseModel(basis_gates=C.BASIS_GATES)
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p10, p10], [p01, 1 - p01]]))
    sim = AerSimulator(method="stabilizer", noise_model=nm,
                       seed_simulator=seed, max_parallel_threads=2)
    qc = QuantumCircuit(BATCH, BATCH)
    for q in range(BATCH):
        qc.ry(np.pi / 2, q)
    qc.measure(range(BATCH), range(BATCH))
    shots = int(np.ceil(n_bits / BATCH))
    mem = sim.run(transpile(qc, sim), shots=shots, memory=True).result().get_memory()
    bits = (np.frombuffer("".join(mem).encode(), dtype=np.uint8) - ord("0"))
    return bits[:n_bits], 0.5 * (1 - p01 + p10)      # exact P(1) (measured convention)


def sample_amplitude_damping(n_bits, gamma=0.08, seed=None):
    """T1 relaxation with amplitude gamma applied after state preparation."""
    sim = AerSimulator(method="density_matrix", seed_simulator=seed,
                       max_parallel_threads=2)
    nq = DM_BATCH
    qc = QuantumCircuit(nq, nq)
    for q in range(nq):
        qc.ry(np.pi / 2, q)
        qc.append(amplitude_damping_error(gamma).to_instruction(), [q])
    qc.measure(range(nq), range(nq))
    shots = int(np.ceil(n_bits / nq))
    mem = sim.run(qc, shots=shots, memory=True).result().get_memory()
    bits = (np.frombuffer("".join(mem).encode(), dtype=np.uint8) - ord("0"))
    # after RY(pi/2), rho11 = 1/2, amplitude damping maps it to (1-gamma)/2
    return bits[:n_bits], 0.5 * (1 - gamma)


def sample_phase_damping(n_bits, lam=0.3, seed=None):
    """T2 dephasing: no bias in the Z basis, but purity is lost."""
    sim = AerSimulator(method="density_matrix", seed_simulator=seed,
                       max_parallel_threads=2)
    nq = DM_BATCH
    qc = QuantumCircuit(nq, nq)
    for q in range(nq):
        qc.ry(np.pi / 2, q)
        qc.append(phase_damping_error(lam).to_instruction(), [q])
    qc.measure(range(nq), range(nq))
    shots = int(np.ceil(n_bits / nq))
    mem = sim.run(qc, shots=shots, memory=True).result().get_memory()
    bits = (np.frombuffer("".join(mem).encode(), dtype=np.uint8) - ord("0"))
    return bits[:n_bits], 0.5


def sample_drift(n_bits, eps=0.05, f=0.02, n_blocks=200, seed=None):
    """
    Classical common-mode drift of the pulse: the rotation angle executes a
    slow sinusoidal excursion, so that neighbouring samples share a bias.
    The drift is a *slow classical parameter* -- it changes on a timescale
    much longer than the sampling interval -- so it is modelled by grouping
    the samples into blocks of constant angle within one drift period.
    """
    sim = AerSimulator(method="statevector", seed_simulator=seed,
                       max_parallel_threads=2)
    per_block = max(BATCH, int(np.ceil(n_bits / n_blocks / BATCH) * BATCH))
    out = []
    for k in range(n_blocks):
        theta = np.pi / 2 + eps * np.sin(2 * np.pi * f * k)
        qc = QuantumCircuit(BATCH, BATCH)
        for q in range(BATCH):
            qc.ry(theta, q)
        qc.measure(range(BATCH), range(BATCH))
        shots = per_block // BATCH
        mem = sim.run(transpile(qc, sim), shots=shots,
                      memory=True).result().get_memory()
        out.append("".join(mem))
    bits = (np.frombuffer("".join(out).encode(), dtype=np.uint8) - ord("0"))
    return bits[:n_bits], 0.5


# ---------------------------------------------------------------------------
def analyse(name, bits, p1_theory, description):
    res = nist.run_battery(bits)
    fails = {k: v["p_value"] for k, v in res.items() if not v["pass"]}
    n_ones = int(bits.sum())
    p1 = n_ones / bits.size
    hmin = float(-np.log2(max(p1, 1 - p1)))
    ac1 = float(np.corrcoef(bits[:-1], bits[1:])[0, 1])
    row = {
        "case": name, "description": description,
        "p1": float(p1), "p1_theory": float(p1_theory),
        "bias": float(p1 - 0.5), "min_entropy": hmin,
        "shannon_entropy": float(P.h2(p1)),
        "autocorr_lag1": ac1,
        "n_fail": len(fails),
        "n_tests": len(res),
        "failed_tests": {k: (None if v is None else float(v)) for k, v in fails.items()},
    }
    print(f"  {name:<22s} p1={p1:.5f} (th {p1_theory:.5f}) H_min={hmin:.4f} "
          f"ac1={ac1:+.4f} failed {len(fails)}/{len(res)}: "
          f"{', '.join(list(fails)[:4])}{' ...' if len(fails) > 4 else ''}")
    return row


def run(n_bits=200_000, seed=900):
    rows = []
    b, p1t = sample_asymmetric_readout(n_bits, seed=seed)
    rows.append(analyse("asymmetric readout", b, p1t, r"$p_{1|0}=0.02$, $p_{0|1}=0.06$"))
    b, p1t = sample_amplitude_damping(n_bits, 0.08, seed=seed + 1)
    rows.append(analyse("amplitude damping", b, p1t, r"$T_1$ relaxation, $\gamma=0.08$"))
    b, p1t = sample_drift(n_bits, 0.05, 0.02, seed=seed + 2)
    rows.append(analyse("slow pulse drift", b, p1t,
                        r"$\theta(t)=\pi/2+0.05\sin(2\pi\cdot0.02\,t)$"))
    b, p1t = sample_phase_damping(n_bits, 0.3, seed=seed + 3)
    rows.append(analyse("phase damping", b, p1t, r"$T_2$ dephasing, $\lambda=0.3$"))
    return rows


if __name__ == "__main__":
    import json
    rep = run()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "qrng_failure_modes.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print("wrote", path)
