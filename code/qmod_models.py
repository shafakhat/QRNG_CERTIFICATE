"""
qmod_models.py
==============
Classiq QMOD models for the two experiments of this repository.

    1. QKD : entangling (CHSH / E91) and prepare-and-measure (BB84) circuits.
    2. QRNG: rotation-based and GHZ-correlation-based generators.

Every model is *built and validated locally* by the Classiq SDK: `create_model`
runs the QMOD parser, type checker and semantic expander, and `write_qmod`
serialises the model.  Running `python qmod_models.py` writes the `.qmod`
files to `<repo>/qmod/` and prints a validation report, so the models can be
opened unchanged in Classiq Studio (https://platform.classiq.io) and taken
through `synthesize()` and `execute()` on Classiq's simulator or hardware.

Because Classiq's synthesis and execution stages are cloud services that
require an account token, the numerical results in this repository are
produced by `circuits.py`, which constructs the semantic equivalent of every
QMOD model in Qiskit and executes it on the local Aer simulator.  The two
layers are in one-to-one correspondence (identical state preparation,
rotation conventions and measurement order); `circuits.py` records the exact
Classiq calls needed to reproduce each run in the cloud.

QMOD conventions used here
--------------------------
* `main` cannot have quantum parameters, so every model allocates its own
  qubits and exposes only classical parameters and `Output` registers.
* Helper `@qfunc`s act on already-allocated qubits (no `allocate` inside), so
  they can be composed and batched with `repeat`.
* Classical parameters exposed to the Studio GUI are `CReal`, `CInt` and
  fixed-length `CArray[CInt, N]`.
"""

import json
import os

import numpy as np

from classiq import (CArray, CInt, CReal, CX, H, Output, QArray, QBit, RY, X,
                     allocate, create_model, if_, qfunc, repeat, write_qmod)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMOD_DIR = os.path.join(REPO, "qmod")
N_BATCH = 16          # qubits executed in parallel per shot


# =============================================================================
# state-preparation and measurement helpers (act on allocated qubits)
# =============================================================================
@qfunc
def prep_ry(theta: CReal, q: QBit):
    """RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>."""
    RY(theta, q)


@qfunc
def prep_bell_phi_plus(q0: QBit, q1: QBit):
    """|Phi+> = (|00> + |11>)/sqrt(2)."""
    H(q0)
    CX(q0, q1)


@qfunc
def bb84_encode(alice_bit: CInt, alice_basis: CInt, q: QBit):
    """BB84 encoding:  Z basis |0>/|1>,  X basis |+>/|-|>."""
    if_(alice_bit == 1, lambda: X(q))
    if_(alice_basis == 1, lambda: H(q))


@qfunc
def measure_in_basis(basis: CInt, q: QBit):
    """Rotate the X-basis measurement into the Z basis."""
    if_(basis == 1, lambda: H(q))


# =============================================================================
# QRNG models
# =============================================================================
def model_qrng_rotation(n: int = N_BATCH, theta: float = np.pi / 2):
    """
    Rotation-based QRNG: n independent qubits prepared in RY(theta)|0> and
    measured.  theta = pi/2 gives the unbiased Born distribution
    P(0) = P(1) = 1/2; theta != pi/2 models pulse miscalibration, which is the
    bias mechanism studied in these experiments.
    """
    @qfunc
    def main(theta: CReal, out: Output[QArray[QBit, n]]):
        allocate(out)
        repeat(n, lambda i: prep_ry(theta, out[i]))

    return create_model(main)


def model_qrng_ghz(n_qubits: int = 8):
    """
    Entanglement-based QRNG.  The GHZ state (|0..0> + |1..1>)/sqrt(2) is
    measured in the X basis; the parity of the outcome string is checked
    against the Z-basis correlators, which certifies that the output cannot be
    pre-determined by a classical variable.
    """
    @qfunc
    def main(x_basis: CInt, out: Output[QArray[QBit, n_qubits]]):
        allocate(out)
        H(out[0])
        repeat(n_qubits - 1, lambda i: CX(out[i], out[i + 1]))
        if_(x_basis == 1, lambda: repeat(n_qubits, lambda i: H(out[i])))

    return create_model(main)


# =============================================================================
# QKD models
# =============================================================================
def model_chsh(a_angle: float, b_angle: float):
    """
    One CHSH trial: both parties measure the equatorial observable
    cos(angle) Z + sin(angle) X, implemented as RY(-angle) then a Z-basis
    measurement.  Optimal settings are Alice in {0, pi/2}, Bob in {pi/4, -pi/4}.
    """
    @qfunc
    def main(a: CReal, b: CReal, out: Output[QArray[QBit, 2]]):
        allocate(out)
        prep_bell_phi_plus(out[0], out[1])
        RY(-a, out[0])
        RY(-b, out[1])

    return create_model(main)


def model_bb84_round(alice_bit: int, alice_basis: int, bob_basis: int):
    """A single prepare-and-measure BB84 round (one qubit, one measurement)."""
    @qfunc
    def main(bit: CInt, abasis: CInt, bbasis: CInt, out: Output[QBit]):
        allocate(out)
        bb84_encode(bit, abasis, out)
        measure_in_basis(bbasis, out)

    return create_model(main)


def model_bb84_batch(n: int = N_BATCH):
    """
    Batched BB84 preparation: n qubits are encoded in parallel per shot, so a
    single simulator shot returns n protocol rounds.  Alice's bits and bases
    are classical inputs; Bob's basis is applied downstream (basis
    reconciliation is classical).
    """
    @qfunc
    def main(bits: CArray[CInt, n], bases: CArray[CInt, n],
             out: Output[QArray[QBit, n]]):
        allocate(out)
        repeat(n, lambda i: bb84_encode(bits[i], bases[i], out[i]))

    return create_model(main)


MODELS = {
    "qrng_rotation": lambda: model_qrng_rotation(),
    "qrng_ghz": lambda: model_qrng_ghz(),
    "chsh_pair": lambda: model_chsh(0.0, np.pi / 4),
    "bb84_round": lambda: model_bb84_round(0, 0, 0),
    "bb84_batch": lambda: model_bb84_batch(),
}


def validate_all(write: bool = True) -> dict:
    """Build, validate and serialise every QMOD model with the local SDK."""
    report = {}
    os.makedirs(QMOD_DIR, exist_ok=True)
    for name, build in MODELS.items():
        try:
            model = build()
            if write:
                write_qmod(model, name, directory=QMOD_DIR, decimal_precision=6)
            report[name] = {"status": "ok"}
        except Exception as exc:                     # pragma: no cover
            report[name] = {"status": "error", "error": str(exc).strip()}
    return report


if __name__ == "__main__":
    rep = validate_all()
    print(json.dumps(rep, indent=2))
    with open(os.path.join(QMOD_DIR, "validation_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    ok = all(v["status"] == "ok" for v in rep.values())
    print("\nall models valid:", ok)
