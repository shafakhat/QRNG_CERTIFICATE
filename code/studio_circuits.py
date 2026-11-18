"""
studio_circuits.py
==================
Writes the five experiment circuits as OpenQASM files you can upload to
https://platform.classiq.io/circuit  (that page accepts `.qasm` and `.qprog`).

ZERO DEPENDENCIES on purpose: no qiskit, no classiq. The QASM text is built
directly from a gate list, so this runs in any interpreter -- including a bare
`python3`, the Classiq venv, or a machine with nothing installed.

Usage
-----
    python3 code/studio_circuits.py                  # write every .qasm
    python3 code/studio_circuits.py --print qrng_ghz # dump one to stdout
    python3 code/studio_circuits.py --print-qmod     # also print the QMOD

Outputs
-------
    qasm/<name>.qasm          OpenQASM 2.0   -> upload / paste
    qasm/qasm3/<name>.qasm    OpenQASM 3.0   -> use if 2.0 is refused

Circuit <-> model correspondence
--------------------------------
    chsh_pair      qmod/chsh_pair.qmod      H, CX, RY(-a), RY(-b), measure 2
    bb84_round     qmod/bb84_round.qmod     X,H (Alice), no rotation (Bob), measure 1
    bb84_batch     qmod/bb84_batch.qmod     16 parallel BB84 encodings
    qrng_rotation  qmod/qrng_rotation.qmod  16 x RY(theta), measure 16
    qrng_ghz       qmod/qrng_ghz.qmod       GHZ-8, H layer, measure 8
"""
from __future__ import annotations

import argparse
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
QASM2_DIR = os.path.join(REPO, "qasm")
QASM3_DIR = os.path.join(QASM2_DIR, "qasm3")
QMOD_DIR = os.path.join(REPO, "qmod")

PI = 3.141592653589793
Q, C = "q", "c"


# ---------------------------------------------------------------------------
# circuit definitions: (name, n_qubits, [(gate, [args...]), ...])
# args are ints (qubit indices) or floats (rotation angles)
# ---------------------------------------------------------------------------
def circuits():
    a_alice, a_bob = PI / 4, PI / 8          # CHSH settings used in fig. 1
    theta = PI / 2                           # unbiased rotation for the QRNG

    # fixed example pattern for the batched BB84 encoding (same as the paper's
    # seed-7 draw), written out so the file is self-contained
    bits = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    bases = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]

    out = []

    # 1. CHSH trial -------------------------------------------------------
    g = [("h", [0]), ("cx", [0, 1]),
         ("ry", [-a_alice, 0]), ("ry", [-a_bob, 1])]
    out.append(("chsh_pair", 2, g))

    # 2. BB84 rounds, one per (bit, basis) combination ----------------------
    #    matching bases   -> deterministic outcome (the round survives sifting)
    #    mismatched bases -> equiprobable outcome (the round is discarded)
    #    encode: X if bit=1, then H if basis=1;  measurement: H if bob_basis=1
    out.append(("bb84_round", 1, [("x", [0]), ("h", [0])]))        # bit 1, X vs Z (mismatch)
    out.append(("bb84_round_z0", 1, []))                           # bit 0, Z vs Z -> always 0
    out.append(("bb84_round_z1", 1, [("x", [0])]))                 # bit 1, Z vs Z -> always 1
    out.append(("bb84_round_x0", 1, [("h", [0])]))                 # bit 0, X vs X -> always 0
    out.append(("bb84_round_x1", 1, [("x", [0]), ("h", [0]), ("h", [0])]))  # bit 1, X vs X -> always 1

    # 3. 16 BB84 encodings in parallel ------------------------------------
    g = []
    for i in range(16):
        if bits[i]:
            g.append(("x", [i]))
        if bases[i]:
            g.append(("h", [i]))
    out.append(("bb84_batch", 16, g))

    # 4. rotation-based QRNG: 16 x RY(theta) ------------------------------
    g = [("ry", [theta, i]) for i in range(16)]
    out.append(("qrng_rotation", 16, g))

    # 4b. small versions, for a results table you can actually read.
    #     16 qubits -> 65 536 possible outcomes, so 20 000 shots yields ~19 900
    #     rows of count 1-2.  4 qubits -> 16 rows of ~1 250 each, and
    #     4 x 20 000 = 80 000 bits, only twice the error of the 16-qubit run.
    for nq in (1, 2, 4, 8):
        g = [("ry", [theta, i]) for i in range(nq)]
        out.append((f"qrng_rotation{nq}q", nq, g))

    # 5. GHZ-8, measured in the X basis -----------------------------------
    g = [("h", [0])]
    g += [("cx", [i, i + 1]) for i in range(7)]
    g += [("h", [i]) for i in range(8)]
    out.append(("qrng_ghz", 8, g))

    return out


# ---------------------------------------------------------------------------
# the four CHSH settings, so S can be computed from four Studio executions
#   a in {0, pi/2}  (Alice),  b in {pi/4, -pi/4}  (Bob)
# ---------------------------------------------------------------------------
CHSH_SETTINGS = [
    ("chsh_a0b0", 0.0, PI / 4),
    ("chsh_a0b1", 0.0, -PI / 4),
    ("chsh_a1b0", PI / 2, PI / 4),
    ("chsh_a1b1", PI / 2, -PI / 4),
]


def chsh_circuit(a, b):
    """Bell pair + equatorial settings: RY(-a) on Alice, RY(-b) on Bob."""
    return [("h", [0]), ("cx", [0, 1]), ("ry", [-a, 0]), ("ry", [-b, 1])]


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------
def to_qasm2(name, n, gates):
    L = [f"// {name}  --  the paper Classiq model, OpenQASM 2.0",
         "OPENQASM 2.0;", 'include "qelib1.inc";',
         f"qreg {Q}[{n}];", f"creg {C}[{n}];"]
    for gate, args in gates:
        if gate in ("x", "h"):
            L.append(f"{gate} {Q}[{args[0]}];")
        elif gate in ("cx", "cz"):
            L.append(f"{gate} {Q}[{args[0]}],{Q}[{args[1]}];")
        elif gate in ("ry", "rx", "rz"):
            L.append(f"{gate}({args[0]!r}) {Q}[{args[1]}];")
        elif gate == "u":
            L.append(f"u({args[0]!r},{args[1]!r},{args[2]!r}) {Q}[{args[3]}];")
        else:
            raise ValueError(gate)
    for i in range(n):
        L.append(f"measure {Q}[{i}] -> {C}[{i}];")
    return "\n".join(L) + "\n"


def to_qasm3(name, n, gates):
    L = [f"// {name}  --  the paper Classiq model, OpenQASM 3.0",
         "OPENQASM 3.0;", 'include "stdgates.inc";',
         f"qubit[{n}] {Q};", f"bit[{n}] {C};"]
    for gate, args in gates:
        if gate in ("x", "h"):
            L.append(f"{gate} {Q}[{args[0]}];")
        elif gate in ("cx", "cz"):
            L.append(f"{gate} {Q}[{args[0]}], {Q}[{args[1]}];")
        elif gate in ("ry", "rx", "rz"):
            L.append(f"{gate}({args[0]!r}) {Q}[{args[1]}];")
        else:
            raise ValueError(gate)
    for i in range(n):
        L.append(f"{C}[{i}] = measure {Q}[{i}];")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def print_qmod(only=None):
    if not os.path.isdir(QMOD_DIR):
        print(f"(no {QMOD_DIR} here)")
        return
    for f in sorted(os.listdir(QMOD_DIR)):
        if not f.endswith(".qmod"):
            continue
        n = f[:-5]
        if only and n not in only:
            continue
        print(f"\n{'=' * 68}\n# Studio -> Model tab -> paste -> Synthesize\n# {n}\n{'=' * 68}")
        print(open(os.path.join(QMOD_DIR, f)).read().rstrip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--print", dest="show", nargs="*", default=None,
                    help="print the QASM of these circuits instead of writing files")
    ap.add_argument("--print-qmod", action="store_true")
    args = ap.parse_args()

    allc = {n: (nn, g) for n, nn, g in circuits()}

    if args.show is not None:
        names = args.show or list(allc)
        for n in names:
            if n not in allc:
                sys.exit(f"unknown: {n}; choose from {sorted(allc)}")
            nn, g = allc[n]
            print(f"\n# ----- {n}.qasm  ({nn} qubits) -----")
            print(to_qasm2(n, nn, g), end="")
        if args.print_qmod:
            print_qmod(args.show or None)
        return 0

    os.makedirs(QASM2_DIR, exist_ok=True)
    os.makedirs(QASM3_DIR, exist_ok=True)
    chsh_dir = os.path.join(QASM2_DIR, "chsh")
    os.makedirs(chsh_dir, exist_ok=True)
    for name, a, b in CHSH_SETTINGS:
        g = chsh_circuit(a, b)
        with open(os.path.join(chsh_dir, f"{name}.qasm"), "w") as f:
            f.write(to_qasm2(f"{name}  (a={a:.4f}, b={b:.4f})", 2, g))
    print(f"4 CHSH setting files -> {os.path.relpath(chsh_dir, REPO)}/\n")
    rows = []
    for n, nn, g in circuits():
        if args.only and n not in args.only:
            continue
        p2 = os.path.join(QASM2_DIR, f"{n}.qasm")
        p3 = os.path.join(QASM3_DIR, f"{n}.qasm")
        with open(p2, "w") as f:
            f.write(to_qasm2(n, nn, g))
        with open(p3, "w") as f:
            f.write(to_qasm3(n, nn, g))
        rows.append((n, nn, len(g) + nn, p2))

    print(f"{'circuit':16s} {'qubits':>7s} {'ops':>5s}   file")
    for n, nn, ops, p in rows:
        print(f"{n:16s} {nn:7d} {ops:5d}   {os.path.relpath(p, REPO)}")
    print(f"\nOpenQASM 3.0 copies -> {os.path.relpath(QASM3_DIR, REPO)}/")
    print("""
Next:
  1. open  https://platform.classiq.io/circuit
  2. drag one of the .qasm files into "Upload Quantum Program"
  3. the visualizer renders it (download gives the .qprog form)
If the upload is refused as an unsupported dialect, use qasm/qasm3/ instead.""")
    if args.print_qmod:
        print_qmod(args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
