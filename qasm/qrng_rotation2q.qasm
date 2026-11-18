// qrng_rotation2q  --  QRNG Classiq model, OpenQASM 2.0
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
