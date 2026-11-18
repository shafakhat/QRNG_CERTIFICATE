# QRNG_CERTIFICATE

Experimental code and measured results for a study of **quantum random number
generation from gate-model circuits**: two generators (a rotation-based one and a
GHZ-entangled one) are simulated, their raw bit streams are measured, an entropy
estimate is certified with the NIST SP 800-90B estimators, the stream is
compressed with a randomness extractor, and the result is validated with the
14-statistic NIST SP 800-22 battery. The study also measures what the battery does
*not* catch, using negative controls.

Everything here is CPU-only classical simulation. No quantum hardware is required
and no hardware results are claimed.

---

## 1. What the experiment does

`code/run_qrng.py` runs ten experiments and writes every number it measures to
`results/qrng_results.json`:

| experiment | question it answers |
|---|---|
| **F1** generator characterisation | 1 Mbit of raw bits from `RY(π/2)`: bias, Shannon and min-entropy, full NIST battery |
| **F2** miscalibration scan | How does the measured bias follow the Born rule as the pulse angle is detuned? |
| **F3** device-noise scan | What does channel depolarisation and readout error do to the entropy? |
| **F4** SP 800-90B entropy | The nine estimators on 200 000 samples, plus the IID permutation test |
| **F5** extraction and re-validation | Compress with a Toeplitz extractor, then test the extractor output |
| **F6** GHZ generator | Do the GHZ correlators certify randomness from entanglement? |
| **F7** second-level analysis | Do 16 independent 1-Mbit sequences each pass every statistic, and are the p-values uniform? |
| **F8** classical reference | How does the quantum stream compare with PCG64? |
| **F9** negative controls | Which injected faults (detector asymmetry, T1, pulse drift, T2) does the battery actually detect? |
| **F10** estimator validation | Are the nine estimators conservative on Bernoulli sources of known min-entropy? |

---

## 2. What you need

| software | version | why it is needed |
|---|---|---|
| **Python** | 3.10 or newer | everything runs on Python |
| **NumPy** | ≥ 1.26 | array maths, the bit streams |
| **SciPy** | ≥ 1.11 | special functions and distributions inside the statistical tests |
| **Qiskit** | ≥ 1.0 | circuit construction |
| **Qiskit Aer** | ≥ 0.14 | the local simulator that actually executes the circuits |
| **Matplotlib** | ≥ 3.8 | only for `visualize_in_studio.py` |
| **Classiq SDK** | ≥ 1.30, *optional* | only for `qmod_models.py`, `run_classiq_synthesis.py` and the cloud route (needs a free Classiq account) |

Install:

```bash
git clone https://github.com/shafakhat/qrng-certification.git
cd qrng-certification
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

`requirements.txt` also installs the optional Classiq SDK. If you do not want it,
`pip install numpy scipy matplotlib qiskit qiskit-aer` is enough to run every
experiment in this repository.

---

## 3. How to run it

### Smoke test (about 1 minute)

```bash
python3 code/run_qrng.py --quick
```

Small shot counts. It proves the pipeline works and writes
`results/qrng_results.json` with reduced statistics. Use it to check the
installation; the shipped numbers come from the full run.

### Full run (about 7 minutes)

```bash
python3 code/run_qrng.py
```

This reproduces the shipped `results/qrng_results.json` (the reference run took
404 s). It ends with:

```
all QRNG experiments finished in <N> s
```

### Run only some stages

Any subset can be re-run; results are merged into the same JSON:

```bash
python3 code/run_qrng.py --stages=F7,F9        # second-level tests and negative controls only
```

Stage names are `F1 … F10`.

### Companion scripts

```bash
python3 code/calibrate_battery.py      # calibrate the battery on 60 reference sequences -> results/calibration_battery.json
python3 code/sts_crosscheck.py         # compare against the NIST STS 2.1.2 reference code  -> results/sts_crosscheck.json
```

### Optional: Classiq cloud route

The QMOD models in `qmod/` are the same circuits in Classiq's QMOD language:

```bash
pip install classiq
classiq authenticate                     # paste your free account token
python3 code/run_classiq_synthesis.py    # synthesise and report depth/width
python3 code/visualize_in_studio.py      # printable circuit diagrams
```

Without an account everything else still runs: the QMOD files are validated
locally and the local Aer sampler is used instead.

---

## 4. What you get

| file | contents |
|---|---|
| `results/qrng_results.json` | **all measured numbers**: F1–F10, F7b sampler qualification |
| `results/calibration_battery.json`, `results/calibration.log` | battery self-calibration on reference sequences |
| `results/sts_crosscheck.json`, `results/sts_reports/` | agreement with the NIST STS 2.1.2 reference implementation |
| `results/apen_calibration.json` | empirical distribution of the approximate-entropy statistic |
|  `results/seed_sensitivity.json`| how stable the statistics are across seeds |
| `results/keys/qrng_extracted_key.npy` | the extracted key as a NumPy array |
| `results/classiq_*.json`, `results/studio_counts/` | raw counts exported from Classiq Studio and the execution log |
| `results/qrng_run.log` | console log of the reference run |

---

## 5. Check that your run matches

After the full run, these values in `results/qrng_results.json` should match:

| quantity | key in the JSON | value |
|---|---|---|
| raw bias of the generator | `F1_main.p1` | 0.500032 (bias 3.24 × 10⁻⁵) |
| min-entropy per sample, single run | `F1_main.min_entropy` | 0.999906 bits |
| certified entropy rate (SP 800-90B minimum) | `F5_extraction.h_min_used` | 0.7294 bits/sample |
| extractor output length | `F5_extraction.n_out` | 743 803 bits from 1 048 576 |
| statistics passed by the extractor output | `F5_extraction.battery_pass` | 13 of 14 |
| statistics passed by the raw stream | `F7_second_level` | 14 of 14 |

Seeds are fixed, so a full run is deterministic on the same Qiskit/Aer version.
Different Aer releases change the random stream, so exact last digits can move;
the physics does not.

---

## 6. Repository layout

```
code/     the experiments
  run_qrng.py           F1–F10, the main driver
  circuits.py           circuit builders and the device/noise model
  nist_sp800_22.py      14-statistic NIST SP 800-22 battery (written from scratch)
  sp800_90b.py          the nine SP 800-90B estimators and the IID test
  postprocessing.py     Toeplitz randomness extractor
  failure_modes.py      the negative controls used by F9
  calibrate_battery.py  battery self-calibration on reference sequences
  sts_crosscheck.py     comparison against the NIST STS 2.1.2 reference code
  analyze_studio_counts.py, parse_studio_table.py   Classiq Studio exports
  qmod_models.py        the same circuits as Classiq QMOD models
  run_classiq_synthesis.py, studio_circuits.py, classiq_jsonc_to_qasm.py,
  visualize_in_studio.py, run_ibm_hardware.py        platform and hardware routes
qmod/     the QMOD models as files (openable in Classiq Studio)
qasm/     the same circuits exported as OpenQASM 2/3
results/  every measured number, the raw counts and the extracted key
```

---

## 7. Scope, honestly

* All numbers come from **classical simulation** of the circuits (Qiskit Aer) with
  an explicit, stated device model — depolarising channel, readout error, and the
  injected faults used as negative controls — plus execution on the Classiq hosted
  simulator. Nothing was run on quantum hardware.
* The device model is memoryless; real detectors have drift and afterpulsing that
  this model does not represent.
* `code/run_ibm_hardware.py` contains the route for running the same circuits on
  IBM hardware; it needs credentials and was not used for any reported number.

## Licence

MIT — see `LICENSE`. Use it, cite it, build on it.
