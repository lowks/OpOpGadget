__author__ = 'Giuliano'
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import romb
from OpOp.df.spherical import df_isotropic
from scipy import interpolate


#Plummer pot
def pot_pl(r,b=1,m=1):
    G=4.5168846503287735e-39
    return (G*m)/np.sqrt(b*b+r*r)


#Plummer dens
def dens_pl(r,b=1,m=1,ra=1e20):
    cost=(3*m)/(4*np.pi*b*b*b)
    return cost*(1/np.sqrt((1+(r/b)*(r/b))**5))


#Henrquist
def pot_h(r,b=1,m=1):
    G=4.5168846503287735e-39
    return ((G*m)/(r+b))

def dens_h(r,b=1,m=1,ra=1e6):
    cost=m/(2*np.pi)
    cost2=b/r
    return (cost*cost2/(r+b)**3)*(1+(r*r)/(ra*ra))








R=np.logspace(np.log10(3E-3),np.log10(300),512)
Rplot=np.linspace(0,10,1000)
e=np.logspace(np.log10(0.001),np.log10(1),512)
#plt.plot(R,dens(R)/dens(0))
#plt.plot(R,pot(R)/pot(0))
phigrid=np.linspace(0.001,1,int(2**9+1))


pgrid=pot_pl(R,1,1)#+pot_pl(R,10,5)
dgrid=dens_pl(R,1,1)


pgrid=pgrid/np.max(pgrid)
dgrid=dgrid/np.max(dgrid)

print(R)
print(dgrid)


e,fe=df_isotropic(dgrid,pgrid)
#_,_,dff=nip(dgrid,pgrid)




fig=plt.figure()
ax1=fig.add_axes((0.1,0.5,0.8,0.45))
ax2=fig.add_axes((0.1,0.3,0.8,0.2))
ax3=fig.add_axes((0.1,0.1,0.8,0.20))

ax1.set_xticklabels([])
ax1.set_ylabel(r'$\tilde{f}(\tilde{\epsilon})$',fontsize=18)
ax1.yaxis.set_tick_params(labelsize=15)

ax2.set_xticklabels([])
ax2.yaxis.set_ticks([0,0.001,0.002])
ax2.yaxis.tick_right()
ax2.yaxis.set_ticks_position('both')
ax2.yaxis.set_tick_params(labelsize=15)


ax3.set_xlabel(r'$\tilde{\epsilon}$',fontsize=25)
ax3.xaxis.set_tick_params(labelsize=15)
ax3.yaxis.set_ticks([0.2,0.6,1,1.4])
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_ylabel('%')

R=np.logspace(np.log10(0.001),np.log10(10),512)

R=np.linspace(0.1,1,1000)


ax1.plot(R,f(R),'r',lw=2,label='Numerical df')
ax1.plot(pgrid,dff/np.max(dff),'g',lw=2,label='Nip df')
ax1.plot(R,R**3.5,'b',lw=2,label='Analytical df')






ax2.plot(R,f(R)-R**3.5,'black',label='Absolute error')

ax3.plot(R,100*(np.abs(f(R)-R**3.5)/R**3.5),'black',label='Relative error')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper right')
plt.grid()
plt.savefig('plummer_test_2.pdf')
plt.show()

