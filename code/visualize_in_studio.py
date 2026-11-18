"""
visualize_in_studio.py
======================
Renders, for each QMOD model in this project, the three things Classiq Studio can
show you:

  1. the INTERACTIVE CIRCUIT   -- the hierarchical block / quantum-data-flow view
                                 of the *synthesized* circuit  (show(qprog))
  2. the MEASUREMENT RESULTS   -- a histogram of sampled bitstrings (sample(...))
  3. the STATE VECTOR          -- exact amplitudes / probabilities (optional,
                                 small circuits only)

It also writes a saved quantum program you can drag-and-drop into
https://platform.classiq.io/circuit/ and an offline PNG of every histogram, so you
have the pictures even outside Studio.

Run it INSIDE your Classiq Studio workspace (needs a logged-in session):

    /home/openvscode-server/classiq-venv/bin/python code/visualize_in_studio.py
    /home/openvscode-server/classiq-venv/bin/python code/visualize_in_studio.py --only qrng_ghz
    /home/openvscode-server/classiq-venv/bin/python code/visualize_in_studio.py --shots 20000 --statevector

Outputs -> results/visuals/  (PNG histograms, .qprog files, summary.json)

Nothing here changes any number in the papers: it is inspection only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
OUT = os.path.join(REPO, "results", "visuals")
sys.path.insert(0, CODE)


# ----------------------------------------------------------------------------
# what to render
# ----------------------------------------------------------------------------
#  name              builder (from qmod_models)                  needs parameters?
def _models():
    import qmod_models as qm

    return {
        # Bell pair + RY(-a), RY(-b) measurement settings  --> CHSH S value
        "chsh_pair": (lambda: qm.model_chsh(np.pi / 4, np.pi / 8), None),
        # one BB84 round: encode -> measure in Bob's basis
        "bb84_round": (lambda: qm.model_bb84_round(1, 0, 1), None),
        # 16 BB84 rounds in parallel
        "bb84_batch": (lambda: qm.model_bb84_batch(16), None),
        # 16 independent RY(theta)  --> raw QRNG entropy
        "qrng_rotation": (lambda: qm.model_qrng_rotation(16, np.pi / 2), None),
        # 8-qubit GHZ, optional X-basis rotation
        "qrng_ghz": (lambda: qm.model_qrng_ghz(8), None),
    }


# ----------------------------------------------------------------------------
def metrics_from_qasm(qasm: str) -> dict:
    """Crude but useful: gate counts and a sequential depth estimate."""
    gates, qubits, depth = 0, set(), 0
    two_q = 0
    for line in qasm.splitlines():
        s = line.strip().rstrip(";")
        if not s or s.startswith(("//", "OPENQASM", "include", "qreg", "creg",
                                  "measure", "barrier", "if")):
            continue
        if " " not in s and "(" not in s:
            continue
        name = s.split("(")[0].split(" ")[0]
        args = s[s.find("(") + 1: s.find(")")].split(",") if "(" in s else []
        gates += 1
        depth += 1
        for a in args:
            a = a.strip()
            if a.startswith("q"):
                qubits.add(a)
        if name.lower() in ("cx", "cz", "cy", "swap", "ccx"):
            two_q += 1
    return {"qubits": len(qubits) or None, "gates": gates, "cx": two_q,
            "depth_sequential": depth}


def plot_histogram(df, title, path):
    """Bar chart of the sampled bitstrings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = df.sort_values("counts", ascending=False).head(32)
    labels = [str(b) for b in top["bitstring"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(len(top)), top["counts"], color="#2b6cb0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6, family="monospace")
    ax.set_ylabel("counts")
    ax.set_title(title, fontsize=9)
    n = int(df["counts"].sum())
    ax.text(0.99, 0.95, f"{n} shots, {len(df)} distinct outcomes",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def rand_metrics(df, n_outcomes_expected):
    """Empirical uniformity check on the sampled distribution."""
    p = df["probability"].to_numpy()
    exp = 1.0 / n_outcomes_expected
    chi2 = float(np.sum((p - exp) ** 2 / exp))
    return {"emp_mean_prob": float(p.mean()), "emp_max_dev": float(np.max(np.abs(p - exp))),
            "chi2_vs_uniform": chi2}


# ----------------------------------------------------------------------------
def run_one(name, builder, shots, seed, want_sv, open_analyzer):
    from classiq import calculate_state_vector, sample, show, synthesize

    model = builder()
    print(f"\n=== {name} : synthesising ...", flush=True)
    qprog = synthesize(model)

    qasm = qprog.qasm
    m = metrics_from_qasm(qasm)
    print(f"    synthesized: {m['qubits']} qubits, {m['gates']} gates, "
          f"{m['cx']} two-qubit, depth~{m['depth_sequential']}")

    # (1) save + optionally open the interactive circuit view
    qfile = os.path.join(OUT, f"{name}.qprog")
    try:
        qprog.save_result(qfile)
        print(f"    saved quantum program -> {qfile}")
    except Exception as e:                                    # API drift guard
        print(f"    (could not save quantum program: {type(e).__name__}: {e})")

    if open_analyzer:
        try:
            show(qprog)      # opens the analyzer; renders inline in a notebook
            print("    analyzer opened (hierarchical blocks + quantum data flow)")
        except Exception as e:
            print(f"    (show() failed: {type(e).__name__}: {e})")

    # (1b) the QASM the synthesis engine actually produced
    try:
        qasm_path = os.path.join(OUT, f"{name}_synth.qasm")
        with open(qasm_path, "w") as f:
            f.write(qasm) if isinstance(qasm, str) and qasm else f.write("")
        print(f"    synthesised QASM -> {qasm_path}")
        info_qasm = qasm_path
    except Exception as e:
        print(f"    (QASM not written: {type(e).__name__}: {e})")
        info_qasm = None

    # (2) sample
    df = sample(qprog, backend="simulator", num_shots=shots, random_seed=seed)
    csv = os.path.join(OUT, f"{name}_counts.csv")
    df.to_csv(csv, index=False)
    png = os.path.join(OUT, f"{name}_hist.png")
    plot_histogram(df, f"{name}  (Classiq simulator, {shots} shots)", png)
    print(f"    {len(df)} distinct outcomes -> {png}")

    info = {"qubits": m["qubits"], "gates": m["gates"], "cx": m["cx"],
            "depth_sequential": m["depth_sequential"],
            "n_outcomes_observed": int(len(df)),
            "shots": int(df["counts"].sum()),
            "files": [os.path.basename(csv), os.path.basename(png)],
            "qasm": os.path.basename(info_qasm) if info_qasm else None}

    # (3) state vector (small circuits only)
    if want_sv and (m["qubits"] or 99) <= 12:
        try:
            sv = calculate_state_vector(qprog)
            amps = np.asarray(sv, dtype=complex).ravel()
            probs = np.abs(amps) ** 2
            keep = np.argsort(probs)[::-1][:16]
            state = [{"bitstring": format(int(i), f"0{m['qubits']}b"),
                      "probability": float(probs[i])} for i in keep if probs[i] > 1e-12]
            svfile = os.path.join(OUT, f"{name}_statevector.json")
            with open(svfile, "w") as f:
                json.dump({"amplitudes_kept": len(state), "top": state}, f, indent=2)
            info["statevector_top"] = state[:6]
            print(f"    state vector ({len(state)} non-negligible amplitudes) -> "
                  f"{os.path.basename(svfile)}")
        except Exception as e:
            print(f"    (state vector skipped: {type(e).__name__}: {e})")

    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="render only these models (default: all five)")
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--statevector", action="store_true",
                    help="also compute exact state vectors for circuits <= 12 qubits")
    ap.add_argument("--no-analyzer", action="store_true",
                    help="do not call show() (use when running headless)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    try:
        import qmod_models  # noqa: F401
        import classiq  # noqa: F401
    except ModuleNotFoundError as e:
        sys.exit(f"missing package: {e}\nrun this inside the Classiq venv "
                 f"(/home/openvscode-server/classiq-venv/bin/python).")

    all_models = _models()
    todo = {k: v for k, v in all_models.items() if not args.only or k in args.only}
    unknown = set(args.only or []) - set(all_models)
    if unknown:
        sys.exit(f"unknown model(s): {sorted(unknown)}; choose from {sorted(all_models)}")

    summary, failed = {}, {}
    for name, (builder, _params) in todo.items():
        try:
            summary[name] = run_one(name, builder, args.shots, args.seed,
                                    args.statevector, not args.no_analyzer)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failed[name] = msg
            print(f"    FAILED: {msg}")
            if "401" in msg or "authenticat" in msg.lower():
                print("    -> not logged in. Run:  python -c \"import classiq; classiq.authenticate()\"")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"shots": args.shots, "seed": args.seed,
                   "rendered": summary, "failed": failed}, f, indent=2)

    print(f"\n{len(summary)}/{len(todo)} models rendered -> {OUT}")
    if failed:
        print("failed:", ", ".join(failed))
    return 0 if len(summary) == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
