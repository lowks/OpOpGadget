import numpy as np
import matplotlib.pyplot as plt
from OpOp.df.spherical import df_isotropic, nip
from scipy.interpolate import UnivariateSpline

def moving_average(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plummer(x):

    dens=(1+x*x)**(-2.5)
    mass=(x*x*x)/(1+x*x)**(1.5)
    pot=1/np.sqrt(1+x*x)

    return dens,mass,pot

def henrquist(x):

    dens=1/(x*(1+x)**3)
    mass=(x*x)/(1+x)**2
    pot=1/(1+x)

    return dens,mass,pot

def jaffe(x):

    dens=1/(x*x*(1+x)**2)
    mass=(x)/(1+x)
    pot=np.log((1+x)/x)

    return dens,mass,pot

def nip(dens,pot):
    drpsi=np.zeros(len(dens),dtype=float)
    inte=np.zeros(len(dens),dtype=float)
    df=np.zeros(len(dens),dtype=float)
    for i in range(len(dens)-1):
        drpsi[i]=(dens[i+1]-dens[i])/(pot[i+1]-pot[i])


    for i in range(len(dens)-1):
        dqi=0
        for j in np.arange(i,len(dens)-1):
            dqs=np.sqrt(pot[i]-pot[j+1])
            inte[i]+=drpsi[j]*(dqs-dqi)
            dqi=dqs

    for i in range(len(dens)-1):
        df[i]=(inte[i+1]-inte[i])/(pot[i+1]-pot[i])


    return drpsi,inte,df

def df_h(e):
    c=np.sqrt(e)/( (1-e)*(1-e) )
    a=(1-2*e)*(8*e*e -8*e -3)
    b=(3* (np.arcsin(np.sqrt(e))) )/(np.sqrt(e*(1-e)))
    return c*(a+b)

R=np.logspace(np.log10(3E-3),np.log10(300),512)

dens,mass,pot=henrquist(R)
dens=dens/np.max(dens)
mass=mass/np.max(mass)
pot=pot/np.max(pot)

'''
plt.plot(R,dens)
plt.plot(R,pot)
plt.xlim(0,1)
'''

e,df=df_isotropic(dens,pot)
dff=UnivariateSpline(e[::-1],df[::-1],k=1,s=0)
dft=df_h(pot)




plt.plot(e,np.log(dff(e)/dff(0.3)),label='P')
plt.plot(e,np.log(dft/df_h(0.3)),label='T')
#plt.plot(pot,np.log(dff),label='F')

'''
ee=np.logspace(np.log10(0.0001),np.log10(0.997),512)
dft=df_h(ee)
dft=dft/np.max(dft)
plt.plot(ee,np.log(dft),label='T-e=0.997')


ee=np.logspace(np.log10(0.0001),np.log10(0.9997),512)
dft=df_h(ee)
dft=dft/np.max(dft)
plt.plot(ee,np.log(dft),label='T-e=0.9997')
'''

#plt.plot(pot,np.log(pot**3.5),label='True')
plt.legend(loc='best')
plt.ylabel('log(f)',fontsize=15)
plt.xlabel('e',fontsize=15)
plt.suptitle('Jaffe')

plt.show()
