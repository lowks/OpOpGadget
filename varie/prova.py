import OpOp.grid
import OpOp.df
import OpOp.Model
from OpOp.Model import GeneralModel
from OpOp.Model import NbodyModel
import OpOp.particle
import OpOp.snap
from scipy.interpolate import UnivariateSpline
from OpOp.snap import Analysis
from OpOp.snap import Profile
import numpy as np
import matplotlib.pyplot as plt


R,mass_t,dens_t,pot_t=np.loadtxt('stellarcomp.txt',unpack=True,comments='#')
mms=GeneralModel(R,dens_t,rc=0.6,G='(kpc km2)/(M_sun s2)',Mmax=1e7)
mmdm=GeneralModel(R,dens_t,rc=5,G='(kpc km2)/(M_sun s2)',Mmax=1e8)



s={'type':2,'model':mms, 'npart':int(1e5)}
dm={'type':1, 'model':mmdm,'npart':int(5e7)}
N=int(1e6)

a=NbodyModel([dm])


p=a.generate()
a=Analysis(p, safe=False, auto_centre=True)

prof=Profile(a.p,Ngrid=512,xmin=0.01,xmax=200,kind='log',type=1)
dspline_prof=UnivariateSpline(prof.grid.gx,prof.dens,k=1,s=0)
dspline_true=UnivariateSpline(R,dens_t,k=1,s=0)
plt.plot(prof.grid.gx,dspline_prof(prof.grid.gx)/dspline_prof(5))
plt.plot(R*5,dspline_true(R)/dspline_true(1))
plt.xlim(0,10)
plt.ylim(-5,5)
plt.show()

'''
prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin',type=1)
plt.plot(prof.grid.gx,np.log10(prof.dens))
prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin',type=2)
plt.plot(prof.grid.gx,np.log10(prof.dens))
prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin')
plt.plot(prof.grid.gx,np.log10(prof.dens))
plt.show()

prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin',type=1)
plt.plot(prof.grid.gx,prof.masscum/(1e8+1e7))
prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin',type=2)
plt.plot(prof.grid.gx,prof.masscum/(1e8+1e7))
prof=Profile(p,Ngrid=512,xmin=0.2,xmax=20,kind='lin')
plt.plot(prof.grid.gx,prof.masscum/(1e8+1e7))
plt.show()
'''