# Classiq Studio execution log — QRNG experiments

Every job below was executed in Classiq Studio and the counts were exported
to `results/studio_counts/`. 6 jobs in this repository,
83000 shots total, 40000 shots feeding the reported results.

| program | shots | completed | purpose | used in results |
|---|---|---|---|---|
| `bb84_batch.qasm` | 1000 | 2026-11-05 19:42 | diagnostic: fixed-pattern demo circuit | no (diagnostic) |
| `qrng_rotation.qasm` | 20000 | 2026-11-05 19:54 | diagnostic: 16 qubits, 65 536 bins | no (diagnostic) |
| `qrng_rotation8q.qasm` | 20000 | 2026-11-05 20:05 | QRNG: rotation generator, 8 qubits | yes |
| `bb84_round.qasm` | 20000 | 2026-11-05 20:17 | diagnostic: single-round unit cell | no (diagnostic) |
| `qrng_ghz.qasm` | 20000 | 2026-11-05 20:53 | QRNG: GHZ-8 X-basis certificate | yes |
| `bb84_round.qasm` | 2000 | 2026-11-05 20:56 | diagnostic: basis-mismatch round (bit 1, Alice X, Bob Z) | no (diagnostic) |
