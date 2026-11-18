#!/usr/bin/env python3
"""
run_ibm_hardware.py -- execute the paper's circuits on real IBM Quantum hardware.

Why this script exists
----------------------
Every reported number currently comes from classical simulation
(Aer, or the Classiq hosted simulator).  The single highest-value upgrade is to
run the same six circuits on a real QPU: four CHSH settings and the 8-qubit GHZ
in both the X and Z bases.  This script does that and writes the counts in
exactly the layout that `code/analyze_studio_counts.py` already understands, so
no analysis code has to change.

Circuits (six jobs, ~120 000 shots total)
    chsh_a0b0, chsh_a0b1, chsh_a1b0, chsh_a1b1   -> CHSH S
    qrng_ghz      (X basis, H layer before readout) -> <X^8> = +1 signature
    qrng_ghz_z    (Z basis, no H layer)             -> 00000000 / 11111111 only

Usage
    # 1. validate everything without touching the cloud (no account needed)
    python3 code/run_ibm_hardware.py --dry-run

    # 2. once you have an IBM Quantum Open Plan account
    export QISKIT_IBM_TOKEN="<your IBM Cloud API key>"
    export QISKIT_IBM_CHANNEL="ibm_quantum_platform"
    export QISKIT_IBM_INSTANCE="<your instance CRN>"     # if your account has one
    python3 code/run_ibm_hardware.py --shots 20000

    # 3. feed the results to the existing analysis
    python3 code/analyze_studio_counts.py --dir results/ibm_hw

Design notes
------------
* Qiskit returns little-endian bitstrings ("c1c0"); Classiq Studio returns
  big-endian ("q0q1...").  Everything written here is normalised to BIG-ENDIAN
  so the first character is qubit 0 / Alice, matching the Studio exports.
* All six circuits are submitted in ONE session so the per-job cloud overhead is
  paid once.  The quota is 10 QPU-minutes per 28-day rolling window; six
  shallow circuits use seconds of it.  The real cost is queue time.
* Nothing is submitted unless --yes is passed after the printed estimate, so an
  accidental run cannot burn quota.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QASM = os.path.join(REPO, "qasm")
OUT = os.path.join(REPO, "results", "ibm_hw")

# (stem, qasm path relative to qasm/, n_qubits, human label)
CIRCUITS = [
    ("chsh_a0b0", "chsh/chsh_a0b0.qasm", 2, "CHSH a=0,    b=pi/4"),
    ("chsh_a0b1", "chsh/chsh_a0b1.qasm", 2, "CHSH a=0,    b=-pi/4"),
    ("chsh_a1b0", "chsh/chsh_a1b0.qasm", 2, "CHSH a=pi/2, b=pi/4"),
    ("chsh_a1b1", "chsh/chsh_a1b1.qasm", 2, "CHSH a=pi/2, b=-pi/4"),
    ("qrng_ghz", "qrng_ghz.qasm", 8, "GHZ-8, X basis (H layer kept)"),
]

# The Z-basis GHZ is the X-basis circuit without the final H layer.  Generated
# here rather than stored, so the two variants cannot drift apart.
GHZ_Z_QASM = """// qrng_ghz_z  --  GHZ-8 measured in the Z basis (no analysis H layer)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
"""


def load_circuits():
    """Read the QASM files; return list of (stem, qasm_text, n_qubits, label)."""
    out = []
    for stem, rel, n, label in CIRCUITS:
        path = os.path.join(QASM, rel)
        if not os.path.isfile(path):
            sys.exit("missing circuit file: %s" % path)
        with open(path) as f:
            out.append((stem, f.read(), n, label))
    out.append(("qrng_ghz_z", GHZ_Z_QASM, 8, "GHZ-8, Z basis (no H layer)"))
    return out


def qiskit_available():
    try:
        import qiskit  # noqa: F401
        import qiskit_ibm_runtime  # noqa: F401
        return True
    except Exception:
        return False


def dry_run(circuits, shots):
    """Validate the QASM and print the submission plan.  Needs no account."""
    print("=" * 72)
    print("DRY RUN -- nothing is submitted")
    print("=" * 72)
    total = 0
    for stem, text, n, label in circuits:
        gates = [ln.strip() for ln in text.splitlines()
                 if ln.strip() and not ln.strip().startswith("//")]
        n_cx = sum(1 for g in gates if g.startswith("cx"))
        n_1q = sum(1 for g in gates if g.split()[0] in ("h", "ry", "rx", "rz"))
        print("  %-12s %d qubits | %d 1q | %d cx | %s"
              % (stem, n, n_1q, n_cx, label))
        total += shots
    print()
    print("  jobs            : %d" % len(circuits))
    print("  shots per job   : %d" % shots)
    print("  shots total     : %d" % total)
    print()
    print("  quota check: the Open Plan allows 10 QPU-minutes per rolling")
    print("  28-day window.  These are 2- and 8-qubit circuits whose transpiled")
    print("  depth is O(10), so six jobs at %d shots consume on the order of" % shots)
    print("  SECONDS of QPU time.  The binding cost is queue time, not quota.")
    print()
    if qiskit_available():
        print("  qiskit: installed")
    else:
        print("  qiskit: NOT installed in this environment.  On the machine that")
        print("          will submit, run:")
        print("              pip install 'qiskit>=2.1' 'qiskit-ibm-runtime>=0.40'")
    print()
    print("  when you have access:")
    print("      export QISKIT_IBM_CHANNEL=ibm_quantum_platform")
    print("      export QISKIT_IBM_TOKEN=<IBM Cloud API key>")
    print("      python3 code/run_ibm_hardware.py --shots %d" % shots)
    return 0


def normalise_counts(raw):
    """Qiskit 'c1c0' (little-endian) -> 'q0q1' big-endian, matching Studio."""
    out = {}
    for key, val in raw.items():
        bits = key.replace(" ", "")[::-1]      # reverse once, keep spaces out
        out[bits] = out.get(bits, 0) + int(val)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def submit(circuits, shots, backend_name, yes, optimization):
    from qiskit import qasm2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    service = QiskitRuntimeService()
    if backend_name in (None, "", "least_busy"):
        backend = service.least_busy(operational=True, simulator=False,
                                     min_num_qubits=8)
    else:
        backend = service.backend(backend_name)
    print("backend: %s (%d qubits)" % (backend.name, backend.num_qubits))

    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization)
    pubs, meta = [], []
    for stem, text, n, label in circuits:
        qc = qasm2.loads(text)
        isa = pm.run(qc)
        width = max(1, isa.depth())
        two_q = sum(isa.count_ops().get(g, 0) for g in ("cx", "cz", "ecr"))
        est_s = shots * (width * 0.6e-6 + 3.0e-6)     # crude: depth + readout
        print("  %-12s transpiled depth %3d | %2d two-qubit gates | "
              "~%.2f s QPU" % (stem, width, two_q, est_s))
        pubs.append((isa, None, shots))
        meta.append({"stem": stem, "label": label, "n_qubits": n,
                     "transpiled_depth": width, "two_qubit_gates": two_q})

    approx = sum(shots * (m["transpiled_depth"] * 0.6e-6 + 3.0e-6)
                 for m in meta)
    print("\nestimated QPU time: ~%.1f s (quota: 600 s per 28 days)" % approx)
    if not yes:
        reply = input("submit %d jobs x %d shots on %s? [y/N] "
                      % (len(circuits), shots, backend.name))
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted; nothing submitted")
            return 1

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    job = sampler.run([p[0] for p in pubs])
    print("job id: %s" % job.job_id())
    print("queued; status: %s" % job.status())
    result = job.result()

    os.makedirs(OUT, exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    for idx, m in enumerate(meta):
        raw = result[idx].data.meas.get_counts() if hasattr(result[idx].data, "meas") \
            else list(result[idx].data.__dict__.values())[0].get_counts()
        counts = normalise_counts(raw)
        payload = {"counts": counts,
                   "metadata": {"backend": backend.name, "job_id": job.job_id(),
                                "shots": shots, "submitted_utc": stamp,
                                "convention": "big-endian: first char = qubit 0",
                                "circuit": m["label"],
                                "transpiled_depth": m["transpiled_depth"],
                                "two_qubit_gates": m["two_qubit_gates"]}}
        path = os.path.join(OUT, m["stem"] + ".json")
        with open(path, "w") as f:
            json.dump({m["stem"]: payload}, f, indent=2)
        print("wrote %s (%d distinct outcomes)" % (path, len(counts)))
    print("\nnext:  python3 code/analyze_studio_counts.py --dir results/ibm_hw")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=20000,
                    help="shots per circuit (default 20000, same as the paper)")
    ap.add_argument("--backend", default="least_busy",
                    help="backend name, or 'least_busy' (default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate circuits and print the plan; submit nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--optimization", type=int, default=1,
                    help="transpiler optimisation level (default 1; keep low so "
                         "the circuit stays recognisable as the one in the paper)")
    args = ap.parse_args()

    circuits = load_circuits()
    if args.dry_run or not qiskit_available():
        if not args.dry_run:
            print("note: qiskit not importable here, falling back to --dry-run\n")
        return dry_run(circuits, args.shots)
    return submit(circuits, args.shots, args.backend, args.yes, args.optimization)


if __name__ == "__main__":
    sys.exit(main())
