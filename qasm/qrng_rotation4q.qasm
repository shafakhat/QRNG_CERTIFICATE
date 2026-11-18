// qrng_rotation4q  --  QRNG Classiq model, OpenQASM 2.0
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
