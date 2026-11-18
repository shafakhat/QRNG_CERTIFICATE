"""
run_classiq_synthesis.py
========================
Run the five QMOD models of this repository through the **real Classiq platform**
and record what the synthesiser produces -- the missing piece behind the
"the synthesis runs on a high-level quantum software stack".

What it does
------------
1.  ``--check``   local sanity: SDK version, model validation (offline), auth state.
2.  default       for every model and for each of two synthesis profiles
                  (``width``-optimised and ``depth``-optimised):

                    create_model  ->  synthesize  ->  metrics  ->  execute

                  and writes ``results/classiq_synthesis.json``.
3.  ``--dry-run`` no account needed: writes the same JSON with the *hand-written
                  Qiskit baseline* column (if qiskit is installed) and leaves the
                  synthesised columns empty, ready to be filled in.

Metrics recorded per (model, profile): qubits, depth, CX count, total gates, and
optionally the shot histogram from a real execution.

Usage
-----
    python3 code/run_classiq_synthesis.py --check        # first, no account
    python3 code/run_classiq_synthesis.py --shots 20000  # after authenticate()
    python3 code/run_classiq_synthesis.py --dry-run      # baseline only

Authentication (one time, needs a browser):
    python3 -c "import classiq; classiq.authenticate()"
Classiq is free for academic use -- register with your institutional e-mail at
https://platform.classiq.io and then run the command above.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "results", "classiq_synthesis.json")


# ---------------------------------------------------------------------------
# model catalogue -- mirrors code/qmod_models.py
# ---------------------------------------------------------------------------
def build_catalogue():
    """Import the builders; the module writes the .qmod files as a side effect."""
    sys.path.insert(0, os.path.join(REPO, "code"))
    import qmod_models as qm

    cat = []
    for name, fn in (
        ("qrng_rotation", lambda: qm.model_qrng_rotation()),
        ("qrng_ghz", lambda: qm.model_qrng_ghz()),
        # CHSH setting (a, b) = (0, pi/4): the pair used for the E00 correlator
        ("chsh_pair", lambda: qm.model_chsh(0.0, np.pi / 4)),
        ("bb84_round", lambda: qm.model_bb84_round(0, 0, 0)),
        ("bb84_batch", lambda: qm.model_bb84_batch()),
    ):
        cat.append((name, fn))
    return cat


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sdk_state():
    try:
        import classiq
    except ImportError:
        return {"installed": False}
    st = {"installed": True, "version": getattr(classiq, "__version__", "?")}
    try:                                     # cheap authenticated call
        from classiq import ExecutionJob
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            ExecutionJob.from_id("00000000-0000-0000-0000-000000000000")
    except Exception as e:                   # noqa: BLE001
        msg = str(e)
        st["authenticated"] = not ("401" in msg or "Not authenticated" in msg)
        if not st["authenticated"]:
            st["auth_hint"] = msg.splitlines()[0][:120]
    return st


def circuit_of(qprog):
    """Return the transpiled circuit of a QuantumProgram across SDK versions."""
    for getter in ("transpiled_circuit",):
        try:
            c = getattr(qprog, getter, None)
            if c is not None:
                return c
        except Exception:                    # noqa: BLE001
            pass
    for path in (("data", "transpiled_circuit"), ("program_circuit",), ("data", "circuit")):
        obj = qprog
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if obj is not None:
                return obj
        except Exception:                    # noqa: BLE001
            continue
    return None


def metrics(qprog):
    """qubits / depth / CX / total gates, from the transpiled circuit if we can."""
    circ = circuit_of(qprog)
    out = {"qubits": None, "depth": None, "cx": None, "gates": None, "source": None}
    if circ is None:
        return out
    try:                                     # qiskit-style Circuit
        ops = circ.count_ops()
        out.update(qubits=int(circ.num_qubits), depth=int(circ.depth()),
                   cx=int(sum(v for k, v in ops.items() if k in ("cx", "CX", "cnot"))),
                   gates=int(sum(ops.values())), source="transpiled_circuit")
        return out
    except Exception:                        # noqa: BLE001
        pass
    try:                                     # QASM fallback: count gate lines
        qasm = circ if isinstance(circ, str) else str(circ)
        lines = [ln.strip() for ln in qasm.splitlines()]
        ops = [ln for ln in lines if ln and not ln.startswith(("OPENQASM", "include",
                                                              "qreg", "creg", "//", "barrier"))]
        names = [ln.split()[0] for ln in ops if ln.split()]
        out.update(qubits=None, depth=len(ops),
                   cx=sum(1 for n in names if n in ("cx", "CX", "cnot")),
                   gates=len(ops), source="qasm")
    except Exception:                        # noqa: BLE001
        pass
    return out


def synthesize_profiles(model, profiles):
    """Synthesise one model under several constraint profiles."""
    from classiq import Constraints, OptimizationParameter, Preferences, synthesize

    results = {}
    for label, param in profiles:
        prefs = Preferences(
            optimization_level=1,
            timeout_seconds=600,
            optimization_timeout_seconds=300,
        )
        cons = Constraints(optimization_parameter=param)
        qp = synthesize(model, constraints=cons, preferences=prefs)
        results[label] = {"metrics": metrics(qp), "qprog": qp}
    return results


def execute_shots(qprog, shots, seed):
    """Execute on Classiq's simulator; return the top of the histogram."""
    import inspect

    from classiq import ExecutionPreferences, execute
    try:
        job = execute(qprog, execution_preferences=ExecutionPreferences(
            num_shots=shots, random_seed=seed))
    except TypeError:                        # 1.3x: preferences live on the model
        job = execute(qprog)
    res = job.result()
    counts = None
    for attr in ("counts", "parsed_counts", "data"):
        v = getattr(res, attr, None)
        if v:
            counts = v
            break
    if isinstance(counts, dict):
        items = sorted(((str(k), int(v)) for k, v in counts.items()),
                       key=lambda kv: -kv[1])[:10]
        return dict(items)
    return None


# ---------------------------------------------------------------------------
# main flows
# ---------------------------------------------------------------------------
def do_check():
    st = sdk_state()
    print("Classiq SDK:", json.dumps(st, indent=2))
    if not st.get("installed"):
        print("\nInstall it with:  pip install classiq")
        return 1
    cat = build_catalogue()
    print("\nLocal model validation (no account needed):")
    ok = True
    for name, fn in cat:
        try:
            fn()
            print(f"  {name:16s} valid")
        except Exception as e:               # noqa: BLE001
            ok = False
            print(f"  {name:16s} FAILED: {e}")
    if not st.get("authenticated"):
        print("\nNext step (needs a browser, one time):")
        print('  python3 -c "import classiq; classiq.authenticate()"')
    else:
        print("\nAuthenticated -- you can run:  python3 code/run_classiq_synthesis.py")
    return 0 if ok else 1


def baseline_metrics():
    """Hand-written Qiskit equivalent of each model (needs qiskit installed)."""
    try:
        from qiskit import transpile
        import circuits
    except Exception:                        # noqa: BLE001
        return {}
    out = {}
    probes = {
        "qrng_rotation": lambda: circuits.build_qrng_rotation(1)[0]
        if hasattr(circuits, "build_qrng_rotation") else None,
    }
    for name, fn in probes.items():
        try:
            circ = fn()
            if circ is None:
                continue
            t = transpile(circ, basis_gates=["u", "cx"], optimization_level=1)
            out[name] = {"qubits": t.num_qubits, "depth": t.depth(),
                         "cx": t.count_ops().get("cx", 0),
                         "gates": sum(t.count_ops().values()),
                         "source": "qiskit-baseline"}
        except Exception:                    # noqa: BLE001
            continue
    return out


def do_run(profiles, shots, seed, dry_run):
    from classiq import OptimizationParameter
    cat = build_catalogue()
    payload = {
        "meta": {
            "sdk": sdk_state(),
            "shots": shots,
            "profiles": [p[0] for p in profiles],
            "note": "synthesised metrics are recorded verbatim from the platform",
        },
        "models": {},
    }
    base = baseline_metrics()
    for name, fn in cat:
        entry = {"baseline": base.get(name), "profiles": {}}
        if not dry_run:
            try:
                model = fn()
                syn = synthesize_profiles(model, profiles)
                for label, info in syn.items():
                    entry["profiles"][label] = dict(info["metrics"])
                    if shots:
                        try:
                            entry["profiles"][label]["counts"] = execute_shots(
                                info["qprog"], shots, seed)
                        except Exception as e:   # noqa: BLE001
                            entry["profiles"][label]["execute_error"] = str(e)[:200]
            except Exception as e:           # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                if "401" in str(e) or "Not authenticated" in str(e):
                    entry["error"] += "  -> run classiq.authenticate() first"
        payload["models"][name] = entry
        status = "error" if entry.get("error") else (
            "ok" if entry["profiles"] else "baseline only")
        print(f"  {name:16s} {status}")
        if entry.get("error"):
            print(f"      {entry['error']}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT_JSON, REPO)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify SDK + offline model validation + auth state")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the JSON skeleton with the Qiskit baseline only")
    ap.add_argument("--shots", type=int, default=0,
                    help="shots per model on the Classiq simulator (0 = synthesise only)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.check:
        return do_check()

    from classiq import OptimizationParameter
    profiles = [("width", OptimizationParameter.WIDTH),
                ("depth", OptimizationParameter.DEPTH)]
    print("Synthesising with the Classiq platform ..." if not args.dry_run
          else "Dry run: baseline only ...")
    return do_run(profiles, args.shots, args.seed, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
