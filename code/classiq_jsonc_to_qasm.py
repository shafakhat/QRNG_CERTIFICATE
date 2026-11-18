"""
classiq_jsonc_to_qasm.py
========================
Take a `.jsonc` file exported from Classiq Studio and produce the two things the
the analysis needs:

    * the synthesised circuit as OpenQASM  ->  print it, and
    * its resource counts (qubits / depth / CX / gates)
      -> merged into results/classiq_synthesis.json, which
         code/make_classiq_table.py turns into Table~\\ref{tab:synthesis}.

It does not matter whether the .jsonc you exported is a **model** (what Studio
shows in the QMOD editor) or an already **synthesised program**: the script
detects which, and acts accordingly.

    .jsonc is a model      -> synthesize it (width + depth profiles)  [needs login]
    .jsonc is a program    -> read the circuit straight out of it     [offline]

Usage
-----
    python3 code/classiq_jsonc_to_qasm.py path/to/model.jsonc
    python3 code/classiq_jsonc_to_qasm.py *.jsonc --shots 20000
    python3 code/classiq_jsonc_to_qasm.py x.jsonc --name chsh_pair --offline

`--offline` skips synthesis and only reports what the file already contains,
which works without a Classiq login.

Notes
-----
* `jsonc` allows `//` and `/* */` comments, which `json.loads` rejects -- they
  are stripped here, with strings preserved (so `https://...` survives).
* Model files from Studio look like  {"kind": "user", "functions": [...], ...}
* Program files contain the transpiled circuit; the QASM is taken from the
  first key of ('qasm', 'qasm3', 'transpiled_circuit', 'program_circuit').
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "results", "classiq_synthesis.json")


# ---------------------------------------------------------------------------
# jsonc -> python object
# ---------------------------------------------------------------------------
def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments that are *outside* string literals."""
    out, i, n = [], 0, len(text)
    in_str = esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    body = "".join(out)
    body = re.sub(r",(\s*[}\]])", r"\1", body)      # trailing commas
    return body


def load_jsonc(path: str):
    with open(path, "r", errors="ignore") as f:
        raw = f.read()
    for attempt in (raw, strip_jsonc(raw)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as e:
            last = e
    raise SystemExit(f"could not parse {path}: {last}")


# ---------------------------------------------------------------------------
# what did we get?
# ---------------------------------------------------------------------------
PROGRAM_KEYS = ("qasm", "qasm3", "transpiled_circuit", "program_circuit", "circuit")


def find_qasm(obj):
    """Depth-first search for a QASM string or a circuit object."""
    if isinstance(obj, str):
        return obj if obj.lstrip().upper().startswith(("OPENQASM", "QASM")) else None
    if isinstance(obj, dict):
        for k in PROGRAM_KEYS:
            if k in obj:
                got = find_qasm(obj[k])
                if got:
                    return got
        for v in obj.values():
            got = find_qasm(v)
            if got:
                return got
    if isinstance(obj, list):
        for v in obj:
            got = find_qasm(v)
            if got:
                return got
    return None


def is_model(obj) -> bool:
    return isinstance(obj, dict) and "functions" in obj and "kind" in obj


def qasm_metrics(qasm: str) -> dict:
    """qubits / depth / cx / gates counted from OpenQASM text."""
    lines = [ln.strip() for ln in qasm.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("//", "/*", "*"))]
    qubits = 0
    for ln in lines:
        m = re.match(r"qreg\s+\w+\[(\d+)\]", ln)
        if m:
            qubits += int(m.group(1))
    body = [ln for ln in lines if not re.match(
        r"(OPENQASM|include|qreg|creg|barrier|measure|if\s*\()", ln)]
    names = [ln.split()[0] for ln in body if ln.split()]
    return {"qubits": qubits or None,
            "depth": len(body),          # sequential estimate; exact depth needs
            "cx": sum(1 for x in names if x in ("cx", "CX", "cnot")),   # a parser
            "gates": len(body),
            "source": "qasm-text"}


# ---------------------------------------------------------------------------
# synthesis path (needs a Classiq login)
# ---------------------------------------------------------------------------
def synthesize_model(obj, name, shots, seed):
    from classiq import (Constraints, ExecutionPreferences, OptimizationParameter,
                         Preferences, SerializedModel, execute, synthesize)

    model = SerializedModel.parse_obj(obj) if hasattr(SerializedModel, "parse_obj") \
        else SerializedModel.model_validate(obj)
    out = {}
    for label, param in (("width", OptimizationParameter.WIDTH),
                         ("depth", OptimizationParameter.DEPTH)):
        prefs = Preferences(optimization_level=1, timeout_seconds=600,
                            optimization_timeout_seconds=300)
        qp = synthesize(model, constraints=Constraints(optimization_parameter=param),
                        preferences=prefs)
        qasm = getattr(qp, "qasm", None) or find_qasm(qp.model_dump())
        entry = qasm_metrics(qasm) if qasm else {"source": None}
        entry["qasm"] = qasm
        if shots:
            try:
                job = execute(qp, execution_preferences=ExecutionPreferences(
                    num_shots=shots, random_seed=seed))
                c = getattr(job.result(), "counts", None)
                if isinstance(c, dict):
                    entry["counts"] = dict(sorted(c.items(), key=lambda kv: -kv[1])[:10])
            except Exception as e:                       # noqa: BLE001
                entry["execute_error"] = str(e)[:200]
        out[label] = entry
    return out


# ---------------------------------------------------------------------------
def merge_result(name, profiles, baseline=None):
    payload = {"meta": {}, "models": {}}
    if os.path.exists(OUT_JSON):
        try:
            payload = json.load(open(OUT_JSON))
        except json.JSONDecodeError:
            pass
    payload.setdefault("meta", {})["note"] = "synthesised metrics recorded from the platform"
    payload.setdefault("models", {})[name] = {"baseline": baseline, "profiles": profiles}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    print(f"  merged into {os.path.relpath(OUT_JSON, REPO)}")


def handle(path, name, shots, seed, offline):
    print(f"\n=== {os.path.basename(path)}")
    obj = load_jsonc(path)
    if is_model(obj):
        fns = [f.get("name", "?") for f in obj.get("functions", [])]
        print(f"  type: MODEL   (kind={obj.get('kind')!r}, functions={fns})")
        if offline:
            print("  --offline: nothing to extract from a model without synthesising.")
            print("  Re-run without --offline to synthesise it (needs a Classiq login).")
            return
        profiles = synthesize_model(obj, name, shots, seed)
        for label, e in profiles.items():
            print(f"  {label:5s}: qubits={e.get('qubits')} depth~{e.get('depth')} "
                  f"cx={e.get('cx')} gates={e.get('gates')}")
            if e.get("qasm"):
                print("  --- qasm (first 400 chars) ---")
                print("  " + e["qasm"][:400].replace("\n", "\n  "))
            if e.get("counts"):
                print(f"  counts: {list(e['counts'].items())[:6]}")
        merge_result(name, {k: {kk: vv for kk, vv in v.items() if kk != "qasm"}
                            for k, v in profiles.items()})
        qasm = next((v.get("qasm") for v in profiles.values() if v.get("qasm")), None)
        if qasm:
            qpath = os.path.join(REPO, "results", f"{name}.qasm")
            open(qpath, "w").write(qasm)
            print(f"  wrote {os.path.relpath(qpath, REPO)}")
        return

    qasm = find_qasm(obj)
    if qasm:
        m = qasm_metrics(qasm)
        print(f"  type: PROGRAM (QASM found)  qubits={m['qubits']} cx={m['cx']} "
              f"gates={m['gates']} depth~{m['depth']}")
        merge_result(name, {"width": m})
        qpath = os.path.join(REPO, "results", f"{name}.qasm")
        open(qpath, "w").write(qasm)
        print(f"  wrote {os.path.relpath(qpath, REPO)}")
        print("  --- qasm (first 400 chars) ---")
        print("  " + qasm[:400].replace("\n", "\n  "))
        return

    print("  type: UNRECOGNISED -- top-level keys:", list(obj)[:12] if isinstance(obj, dict) else type(obj))
    print("  Send me this file as-is; the format will be added to the parser.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more .jsonc files (globs allowed)")
    ap.add_argument("--name", help="model name (default: from filename)")
    ap.add_argument("--shots", type=int, default=0, help="shots to execute (0 = skip)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--offline", action="store_true",
                    help="do not call the platform; only read what the file contains")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    for f in files:
        if not os.path.exists(f):
            print(f"!! not found: {f}")
            continue
        name = args.name or re.sub(r"\.(jsonc|json)$", "", os.path.basename(f))
        handle(f, name, args.shots, args.seed, args.offline)
    print("\nnext: python3 code/make_classiq_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
