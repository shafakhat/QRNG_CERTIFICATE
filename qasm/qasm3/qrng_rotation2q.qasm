// qrng_rotation2q  --  QRNG Classiq model, OpenQASM 3.0
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
c[0] = measure q[0];
c[1] = measure q[1];
