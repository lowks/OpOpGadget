from OpOp.snapio import *
import numpy as np

#p=_load_single('provauno.ics')

#p=_load_multi('snapshot_000')

p=load_snap('galaxy_littleendian.dat')
p=load_snap('snapshot_000')
print(p[340])


write_snap(p,'prova.dat',safe_write=False, enable_mass=True)

p=load_snap('prova.dat')
print(p[340])

