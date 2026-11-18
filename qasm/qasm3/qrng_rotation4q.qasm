// qrng_rotation4q  --  QRNG Classiq model, OpenQASM 3.0
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
