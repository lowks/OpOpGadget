from OpOp.particle_src.particle import Header,Particles
from OpOp.model_src.GeneralModel import GeneralModel
from OpOp.model_src.Model import  Model
from OpOp.grid_src.grid import grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from random import uniform as ru

R,mass_t,dens_t,pot_t=np.loadtxt('stellarcomp.txt',unpack=True,comments='#')


mm=GeneralModel(R,dens_t)


g=grid(galaxy_model=mm, data=R)






'''
#prova per settare i raggi

npart=1e6
mpart=1/npart

mnorm=g.mgrid/np.max(g.mgrid)

max_m=np.max(g.mgrid)
min_m=np.min(g.mgrid)

#u random tra 0 e 1
#random tra [min_m, max_m) : (max-min)*u + min


mass_idx=np.random.uniform(size=int(npart))
mass_idx=mass_idx/np.max(mass_idx) * (max_m-min_m) +min_m

pos=g.eval_rad(mass_idx)

print(np.max(mass_idx),g.eval_rad(np.max(mass_idx)))
print(np.min(mass_idx),g.eval_rad(np.min(mass_idx)))

masshist,bine=np.histogram(pos,bins=g.gedge)
masscum=np.cumsum(masshist)

#plt.step(g.gx,masshist*mpart)
#plt.xlim(0,7)
#plt.show()

masscum_teo=mm.mass(g.gx)/np.max(mm.mass(g.gx))
#masscum_teo=mass_t
masscum_mod_1=masscum*mpart

mass_idx=np.random.uniform(low=min_m,high=max_m,size=int(npart))
masshist,bine=np.histogram(pos,bins=g.gedge)
masscum=np.cumsum(masshist)

masscum_mod_2=masscum*mpart

#plt.plot(g.gx,masscum*mpart)
#plt.plot(g.gx,masscum_teo)
plt.scatter(g.gx,np.log10(np.abs((masscum_teo-masscum_mod_1)/masscum_teo)),c='b',label='Mod-1 Npart 1e6')
plt.plot(g.gx,np.log10(np.abs((masscum_teo-masscum_mod_2)/masscum_teo)),c='r',label='Mod-2 Npart 1e6')


npart=1e3
mpart=1/npart

mnorm=g.mgrid/np.max(g.mgrid)

max_m=np.max(g.mgrid)
min_m=np.min(g.mgrid)

#u random tra 0 e 1
#random tra [min_m, max_m) : (max-min)*u + min


mass_idx=np.random.uniform(size=int(npart))
mass_idx=mass_idx/np.max(mass_idx) * (max_m-min_m) +min_m

pos=g.eval_rad(mass_idx)

print(np.max(mass_idx),g.eval_rad(np.max(mass_idx)))
print(np.min(mass_idx),g.eval_rad(np.min(mass_idx)))

masshist,bine=np.histogram(pos,bins=g.gedge)
masscum=np.cumsum(masshist)

#plt.step(g.gx,masshist*mpart)
#plt.xlim(0,7)
#plt.show()

masscum_teo=mm.mass(g.gx)/np.max(mm.mass(g.gx))
#masscum_teo=mass_t
masscum_mod_1=masscum*mpart

mass_idx=np.random.uniform(low=min_m,high=max_m,size=int(npart))
masshist,bine=np.histogram(pos,bins=g.gedge)
masscum=np.cumsum(masshist)

masscum_mod_2=masscum*mpart

#plt.plot(g.gx,masscum*mpart)
#plt.plot(g.gx,masscum_teo)
plt.scatter(g.gx,np.log10(np.abs((masscum_teo-masscum_mod_1)/masscum_teo)),marker='*',c='b',label='Mod-1 Npart 1e3')
plt.plot(g.gx,np.log10(np.abs((masscum_teo-masscum_mod_2)/masscum_teo)),c='r', ls='dashed',label='Mod-2 Npart 1e3')


plt.xlim(0.2,20)
plt.xlabel('R/Rc',fontsize=18)
plt.ylabel('$\log (\epsilon_M)$',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.suptitle('Cumulative mass',fontsize=20)
plt.legend()
#plt.savefig('cumulative_mass_test.pdf')
plt.show()


npart=1e3
mpart=1/npart
mnorm=g.mgrid/np.max(g.mgrid)
max_m=np.max(g.mgrid)
min_m=np.min(g.mgrid)
mass_idx=np.random.uniform(size=int(npart))
mass_idx=mass_idx/np.max(mass_idx) * (max_m-min_m) +min_m
pos=g.eval_rad(mass_idx)
masshist,bine=np.histogram(pos,bins=g.gedge)
dentm=masshist/g.g_vol


mass_idx=np.random.uniform(low=min_m,high=max_m,size=int(npart))
pos=g.eval_rad(mass_idx)
masshist,bine=np.histogram(pos,bins=g.gedge)
dentm2=masshist/g.g_vol

dentm_f=UnivariateSpline(g.gx,dentm,s=0,k=1)
dentm2_f=UnivariateSpline(g.gx,dentm2,s=0,k=1)
dentt_f=UnivariateSpline(R,dens_t,s=0,k=1)

#plt.plot(g.gx,np.log10(dentm/np.max(dentm)))
#plt.plot(R,np.log10(dens_t/np.max(dens_t)))
dens_mod1=dentm_f(g.gx)/dentm_f(1)
dens_mod2=dentm2_f(g.gx)/dentm2_f(1)
dens_teo=dentt_f(g.gx)/dentt_f(1)
diff=np.abs(dens_mod1-dens_teo)/dens_teo
diff2=np.abs(dens_mod2-dens_teo)/dens_teo
#plt.plot(g.gx,np.log10(dens_mod1))
#plt.plot(g.gx,np.log10(dens_teo))
plt.plot(g.gx, np.log10(diff),c='b', ls='dashed', label='Mod-1 Npart 1e3' )
plt.plot(g.gx, np.log10(diff2),c='r', ls='dashed', label='Mod-2 Npart 1e3' )


npart=1e6
mpart=1/npart
mnorm=g.mgrid/np.max(g.mgrid)
max_m=np.max(g.mgrid)
min_m=np.min(g.mgrid)
mass_idx=np.random.uniform(size=int(npart))
mass_idx=mass_idx/np.max(mass_idx) * (max_m-min_m) +min_m
pos=g.eval_rad(mass_idx)
masshist,bine=np.histogram(pos,bins=g.gedge)
dentm=masshist/g.g_vol


mass_idx=np.random.uniform(low=min_m,high=max_m,size=int(npart))
pos=g.eval_rad(mass_idx)
masshist,bine=np.histogram(pos,bins=g.gedge)
dentm2=masshist/g.g_vol

dentm_f=UnivariateSpline(g.gx,dentm,s=0,k=1)
dentm2_f=UnivariateSpline(g.gx,dentm2,s=0,k=1)
dentt_f=UnivariateSpline(R,dens_t,s=0,k=1)

#plt.plot(g.gx,np.log10(dentm/np.max(dentm)))
#plt.plot(R,np.log10(dens_t/np.max(dens_t)))
dens_mod1=dentm_f(g.gx)/dentm_f(1)
dens_mod2=dentm2_f(g.gx)/dentm2_f(1)
dens_teo=dentt_f(g.gx)/dentt_f(1)
diff=np.abs(dens_mod1-dens_teo)/dens_teo
diff2=np.abs(dens_mod2-dens_teo)/dens_teo
#plt.plot(g.gx,np.log10(dens_mod1))
#plt.plot(g.gx,np.log10(dens_teo))
plt.plot(g.gx, np.log10(diff),c='b', label='Mod-1 Npart 1e6' )
plt.plot(g.gx, np.log10(diff2),c='r', label='Mod-2 Npart 1e6' )




plt.xlabel('R/Rc',fontsize=18)
plt.ylabel(r'$\log (\epsilon_\rho) $',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.suptitle('Density profile',fontsize=20)
plt.xlim(0,10)
plt.ylim(-3,0)
plt.legend(loc='lower right')
plt.savefig('density_profile_test.pdf')
'''
