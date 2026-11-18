"""
circuits.py
===========
Execution layer: Qiskit/Aer circuits that are the semantic equivalent of the
QMOD models in `qmod_models.py`, plus the noise model and sampling helpers.

Device / channel model
----------------------
Three physically distinct error sources are modelled separately so that each
can be calibrated independently on hardware:

  * `p_ch`  : depolarising *quantum channel* applied to every qubit between
              state preparation and measurement (fibre/free-space transmission
              plus memory decoherence).  Bloch vector shrinks by
              lambda = 1 - 4 p_ch / 3, so a two-qubit correlator shrinks by
              lambda^2 and the sifted-key QBER is (1 - lambda^2)/2.
  * `p1`,`p2`: depolarising gate errors on 1- and 2-qubit gates (p2 = 5 p1
              by default, the typical superconducting ratio).
  * `p_ro`  : symmetric readout (detector) flip error.

The depolarising channel is inserted as an explicit Pauli-channel instruction
so that it is applied even to circuits that contain no gates (a BB84 round in
the Z basis is otherwise gate-free and would be noiseless, which would be
unphysical).

Classiq cloud equivalent of every run (`synthesize` + `execute` on the QMOD
model of `qmod_models.py`) is quoted in the corresponding runner script.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

N_BATCH = 16          # qubits per shot in batched sampling (matches QMOD)

BASIS_GATES = ["id", "h", "x", "y", "z", "s", "sdg", "t", "tdg",
               "rx", "ry", "rz", "cx", "cz", "swap"]


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def depol_instruction(p: float):
    """Identity / X / Y / Z with probabilities (1-p, p/3, p/3, p/3)."""
    return pauli_error([("I", 1 - p), ("X", p / 3), ("Y", p / 3), ("Z", p / 3)])


def add_channel(qc: QuantumCircuit, qubits, p_ch: float) -> QuantumCircuit:
    """Apply the depolarising quantum channel to `qubits` (in place)."""
    if p_ch > 0:
        instr = depol_instruction(p_ch).to_instruction()
        for q in qubits:
            qc.append(instr, [q])
    return qc


def make_noise(p1: float = 0.0, p2: float | None = None,
               p_ro: float = 0.0) -> NoiseModel | None:
    """Gate + readout noise model (the channel is added inside the circuits)."""
    if p1 == 0.0 and p_ro == 0.0 and p2 in (None, 0.0):
        return None
    p2 = 5 * p1 if p2 is None else p2
    nm = NoiseModel(basis_gates=BASIS_GATES + ["kraus"])
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(min(p1, 1.0), 1),
                                       ["h", "x", "y", "z", "s", "sdg",
                                        "rx", "ry", "rz"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(min(p2, 1.0), 2),
                                       ["cx", "cz", "swap"])
    if p_ro > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def simulator(p1=0.0, p2=None, p_ro=0.0, method="stabilizer", seed=None):
    return AerSimulator(method=method, noise_model=make_noise(p1, p2, p_ro),
                        seed_simulator=seed, max_parallel_threads=2)


# ---------------------------------------------------------------------------
# CHSH
# ---------------------------------------------------------------------------
CHSH_SETTINGS = [("A0B0", 0.0, np.pi / 4), ("A0B1", 0.0, -np.pi / 4),
                 ("A1B0", np.pi / 2, np.pi / 4), ("A1B1", np.pi / 2, -np.pi / 4)]


def chsh_circuit(a: float, b: float, p_ch: float = 0.0) -> QuantumCircuit:
    """
    |Phi+> = (|00>+|11>)/sqrt(2); Alice measures along
    (cos a)Z + (sin a)X via RY(-a), Bob along (cos b)Z + (sin b)X via RY(-b),
    both followed by a Z-basis measurement.  The depolarising channel is
    applied to both qubits after state preparation.
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    add_channel(qc, [0, 1], p_ch)
    qc.ry(-a, 0)
    qc.ry(-b, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def chsh_correlator(counts: dict, shots: int) -> tuple:
    """E = P(same) - P(different), with the binomial standard error."""
    same = 0
    for key, c in counts.items():
        bits = key.replace(" ", "")          # qiskit: leftmost char = highest qubit
        if bits[-1] == bits[-2]:
            same += c
    p_same = same / shots
    return 2 * p_same - 1, 2 * np.sqrt(p_same * (1 - p_same) / shots), p_same


def chsh_S(Es):
    E00, E01, E10, E11 = Es
    return E00 + E01 + E10 - E11


# ---------------------------------------------------------------------------
# BB84 (batched)
# ---------------------------------------------------------------------------
def bb84_mask_circuit(n: int, alice_basis: int, bob_basis: int,
                      p_ch: float = 0.0) -> QuantumCircuit:
    """
    One shot returns `n` independent BB84 rounds executed in parallel.
    Alice encodes bit 0 in `alice_basis`; a final outcome of 1 therefore means
    "Bob's bit differs from Alice's bit".  Because the noise is
    bit-flip symmetric the same mask applies to Alice's bit 1.
    """
    qc = QuantumCircuit(n, n)
    if alice_basis == 1:
        for q in range(n):
            qc.h(q)
    add_channel(qc, range(n), p_ch)
    if bob_basis == 1:
        for q in range(n):
            qc.h(q)
    qc.measure(range(n), range(n))
    return qc


def sample_error_mask(n_rounds: int, alice_basis: int, bob_basis: int,
                      p_ch=0.0, p1=0.0, p_ro=0.0, seed=None) -> np.ndarray:
    """Error mask (1 = bit error) of `n_rounds` protocol rounds."""
    sim = simulator(p1=p1, p_ro=p_ro, method="stabilizer", seed=seed)
    shots = int(np.ceil(n_rounds / N_BATCH))
    qc = bb84_mask_circuit(N_BATCH, alice_basis, bob_basis, p_ch)
    res = sim.run(transpile(qc, sim), shots=shots, memory=True).result()
    mem = res.get_memory()
    mask = np.frombuffer("".join(mem).encode(), dtype=np.uint8) - ord("0")
    return mask[:n_rounds].astype(np.uint8)


# ---------------------------------------------------------------------------
# QRNG
# ---------------------------------------------------------------------------
def qrng_circuit(theta: float, n: int = N_BATCH, p_ch: float = 0.0) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.ry(theta, q)
    add_channel(qc, range(n), p_ch)
    qc.measure(range(n), range(n))
    return qc


def ghz_circuit(n: int, x_basis: int = 1, p_ch: float = 0.0) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    add_channel(qc, range(n), p_ch)
    if x_basis:
        for q in range(n):
            qc.h(q)
    qc.measure(range(n), range(n))
    return qc


def sample_bits(circuit: QuantumCircuit, shots: int, p1=0.0, p_ro=0.0,
                method="stabilizer", seed=None) -> np.ndarray:
    """Flat uint8 array of measured bits, in shot order."""
    sim = simulator(p1=p1, p_ro=p_ro, method=method, seed=seed)
    res = sim.run(transpile(circuit, sim), shots=shots, memory=True).result()
    mem = res.get_memory()
    return (np.frombuffer("".join(mem).encode(), dtype=np.uint8) - ord("0")).astype(np.uint8)
