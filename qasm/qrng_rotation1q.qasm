// qrng_rotation1q  --  QRNG Classiq model, OpenQASM 2.0
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
ry(1.5707963267948966) q[0];
measure q[0] -> c[0];
