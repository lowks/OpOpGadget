from OpOp.Model import GeneralModel
from OpOp.df import df_isotropic
import  numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import time


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
    #pot=np.log((1+x)/x)
    pot=np.log(1+1/x)

    return dens,mass,pot

def df_h(e):
    c=np.sqrt(e)/( (1-e)*(1-e) )
    a=(1-2*e)*(8*e*e -8*e -3)
    b=(3* (np.arcsin(np.sqrt(e))) )/(np.sqrt(e*(1-e)))
    return c*(a+b)

#Plummer pot
def pot_pl(r,b=1,m=1):
    G=4.5168846503287735e-39
    return (G*m)/np.sqrt(b*b+r*r)


#Plummer dens
def dens_pl(r,b=1,m=1,ra=1e20):
    cost=(3*m)/(4*np.pi*b*b*b)
    return cost*(1/np.sqrt((1+(r/b)*(r/b))**5))


R,mass_t,dens_t,pot_t=np.loadtxt('stellarcomp.txt',unpack=True,comments='#')







#dens_t,mass_t,pot_t=plummer(R)
#dens_t,mass_t,pot_t=henrquist(R)
dens_t,mass_t,pot_t=jaffe(R)
dens_t=dens_t/np.max(dens_t)
mass_t=mass_t/np.max(mass_t)


pot_t=pot_t/np.max(pot_t)




smodel=GeneralModel(R,dens_t)
mass=smodel.mass(R)
mass=mass/np.max(mass)
pot=smodel.pot(R,R[-1])
print(pot[0])
pot=pot/np.max(pot)

smodelc=GeneralModel(R,dens_t,use_c=True)
potc=smodelc.pot(R,R[-1])
print(potc[0])
potc=potc/np.max(potc)
massc=smodelc.mass(R)
massc=massc/np.max(massc)

e,df,dff=df_isotropic(dens_t,pot,use_c=True)
df=dff(e)/dff(0.4)
ec,dfc,dffc=df_isotropic(dens_t,potc,use_c=True)
dfc=dffc(e)/dffc(0.4)


#dft=e**3.5/0.4**3.5
#dft=df_h(e)/df_h(0.4)


fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

#Dens
ax1.plot(R,np.log10(dens_t),c='black')
ax1.set_xlim(0,10)
ax1.set_ylim(-15,0)
ax1.set_xlabel('R/Rc',fontsize=14)
ax1.set_ylabel(r'$\rho / \rho_{max}$',fontsize=18)

#Test Mass

errmass=np.where( mass<0.0001, 0, 100*np.abs(mass-mass_t)/(mass_t) )
errmassc=np.where( mass<0.0001, 0, 100*np.abs(massc-mass_t)/(mass_t) )


ax2.plot(R,np.log10(errmass))
ax2.plot(R,np.log10(errmassc))
ax2.set_xlim(0,100)
#ax2.set_ylim(-3,-1)
ax2.set_xlabel('R/Rc',fontsize=14)
ax2.set_ylabel(r' Log ($err_{mass}$) [%]',fontsize=16)




#Test Pot

errpot=np.where( pot<0.0001, 0, 100*np.abs(pot-pot_t)/(pot_t) )
errpotc=np.where( pot<0.0001, 0, 100*np.abs(potc-pot_t)/(pot_t) )

ax3.plot(R,np.log10(errpot))
ax3.plot(R,np.log10(errpot))

#ax3.plot(R,errpot)
#ax3.plot(R,errpotc)



#ax3.plot(R,np.log10(pot))
#ax3.plot(R,np.log10(potc))

ax3.set_xlim(0,100)
#ax3.set_ylim(6,10)
ax3.set_xlabel('R/Rc',fontsize=14)
ax3.set_ylabel(r' $err_{pot}$ [%]',fontsize=16)
#ax3.set_ylabel(r'$Pot/Pot_{max}$',fontsize=16)


#Test DF


ax4.plot(e,np.log10(df),label='Method-1')
ax4.plot(e,np.log10(dfc),label='Method-2')
#ax4.plot(e,np.log10(dft),label=True)

ax4.set_ylim(-10,10)
ax4.set_xlabel(r'$\epsilon/\epsilon_{max}$',fontsize=16)
ax4.set_ylabel('Log dfnorm',fontsize=16)

plt.legend(loc='upper left')
plt.suptitle('Jaffe',fontsize=16)
plt.show()
#plt.savefig('testjaffe.pdf')








def old():
    '''
    output=np.zeros(shape=(len(dens_t),2))
    output[:,0]=dens_t[::-1]
    output[:,1]=pot_t[::-1]
    np.savetxt('dp.txt',output)






    smodelc=GeneralModelc(Rf,dens_t)
    potc=smodelc.pot(Rf,1)
    potc=potc/np.max(potc)
    massc=smodelc.mass(Rf)
    massc=massc/np.max(massc)

    smodel=GeneralModel(Rf,dens_t)
    pot=smodel.pot(Rf,Rf[-1])
    pot=pot/np.max(pot)
    mass=smodel.mass(Rf)
    mass=mass/np.max(mass)

    plt.plot(Rf,massc,label='C')
    plt.plot(Rf,mass,label='Python')
    plt.plot(R,mass_t,label='True')
    plt.xlim(0,10)
    plt.legend(loc='best')
    plt.savefig('stellar_mass.pdf')
    plt.show()

    diffmassc=np.where(mass_t<0.001,0,100*np.abs(massc-mass_t)/(mass_t))
    diffmass=np.where(mass_t<0.001,0,100*np.abs(mass-mass_t)/(mass_t))
    plt.plot(Rf,diffmassc,label='C')
    plt.plot(Rf,diffmass,label='Python')
    plt.xlim(0,10)
    plt.legend(loc='best')
    plt.show()



    plt.plot(Rf,potc,label='C')
    plt.plot(Rf,pot,label='Python')
    plt.plot(R,pot_t,label='True')
    plt.xlim(0,10)
    plt.legend(loc='best')
    plt.savefig('stellar_pot.pdf')
    plt.show()

    diffpotc=np.where(mass_t<0.0001,0,100*np.abs(potc-pot_t)/(pot_t))
    diffpot=np.where(mass_t<0.0001,0,100*np.abs(pot-pot_t)/(pot_t))
    plt.plot(Rf,diffpotc,label='C')
    plt.plot(Rf,diffpot,label='Python')
    plt.xlim(0,20)
    plt.legend(loc='best')
    plt.show()

    e,df=df_isotropic(dens_t,pot)
    ec,dfc=nip(dens_t,pot)

    dff=UnivariateSpline(e[::-1],df[::-1],k=1,s=0,ext=2)
    dfcf=UnivariateSpline(ec[::-1],dfc[::-1],k=1,s=0,ext=2)

    #ec,dfc,dfuncc=df_isotropic(dens_t,potc,normalize=True, use_c=True, check_negative=False)

    plt.plot(e,dff(e)/dff(0.97),label='P')
    plt.plot(ec,dfcf(ec)/dfcf(0.8),label='C')

    plt.ylim(0,1)
    plt.legend(loc='best')
    #plt.plot(e,e**3.5)
    plt.savefig('stellar_df.pdf')
    plt.show()
    '''

    '''
    R=np.logspace(np.log10(3E-5),np.log10(300),1000)
    #dens_t,mass_t,pot_t=plummer(R)
    #dens_t,mass_t,pot_t=henrquist(R)
    dens_t,mass_t,pot_t=jaffe(R)
    dens_t=dens_t/np.max(dens_t)
    mass_t=mass_t/np.max(mass_t)
    pot_t=pot_t/np.max(pot_t)
    e,df,dfunc=df_isotropic(dens_t,pot_t,normalize=True, use_c=False, check_negative=False)
    ec,dfc=nip(dens_t,pot_t)
    dfc=dfc/np.max(dfc)
    plt.plot(e,df,label='uno',c='blue')
    plt.plot(ec,dfc,'--',label='unoc',c='blue')
    plt.ylim(0,1)
    plt.legend()


    R=np.logspace(np.log10(3E-4),np.log10(300),1000)
    #dens_t,mass_t,pot_t=plummer(R)
    #dens_t,mass_t,pot_t=henrquist(R)
    dens_t,mass_t,pot_t=jaffe(R)
    dens_t=dens_t/np.max(dens_t)
    mass_t=mass_t/np.max(mass_t)
    pot_t=pot_t/np.max(pot_t)
    e,df,dfunc=df_isotropic(dens_t,pot_t,normalize=True, use_c=False, check_negative=False)
    ec,dfc=nip(dens_t,pot_t)
    dfc=dfc/np.max(dfc)
    plt.plot(e,df,label='due',c='black')
    plt.plot(ec,dfc,'--',label='duec',c='black')


    R=np.logspace(np.log10(3E-3),np.log10(300),1000)
    #dens_t,mass_t,pot_t=plummer(R)
    #dens_t,mass_t,pot_t=henrquist(R)
    dens_t,mass_t,pot_t=jaffe(R)
    dens_t=dens_t/np.max(dens_t)
    mass_t=mass_t/np.max(mass_t)
    pot_t=pot_t/np.max(pot_t)
    e,df,dfunc=df_isotropic(dens_t,pot_t,normalize=True, use_c=False, check_negative=False)
    ec,dfc=nip(dens_t,pot_t)
    dfc=dfc/np.max(dfc)
    plt.plot(e,df,label='tre',c='red')
    plt.plot(ec,dfc,'--',label='trec',c='red')




    plt.legend()
    plt.show()
    '''

    '''
    '''
    '''
    #plt.plot(R,pot)
    #plt.plot(R,densc)
    #plt.plot(R,pot/np.max(pot))
    #plt.scatter(R,smodel.pot_arr/np.max(smodel.pot_arr))
    plt.plot(R,pot)
    plt.plot(R,pot_t)
    plt.xlim(0,5)
    plt.show()
    plt.plot(R/np.max(R), 100*np.abs(pot_t-pot)/pot_t, label='Potential', lw=2 )



    diffmass=np.where((mass<0.000001) ,0,(mass-mass_t)/mass)
    plt.plot(R/np.max(R), 100*diffmass , label='Mass',lw=2)


    e,df,dfunc=df_isotropic(dens_t,pot,normalize=True, )

    #plt.plot(e,100*np.abs(df-e**3.5)/e**3.5,label='DF',lw=2)
    plt.ylim(-1,10)
    plt.legend(loc='best')
    plt.show()

    #plt.scatter(e,df)
    #pp=np.polyfit(e,df,10)
    #dff=np.poly1d(pp)
    #plt.plot(e,dff(e),c='orange',lw=2)

    window_size, poly_order =31, 3
    dff = savgol_filter(df, window_size, poly_order)
    plt.plot(e,dff,c='red')
    plt.show()
    '''
    return 0