import numpy as np
import matplotlib.pyplot as plt


vx1,vy1,vz1,t1,vx2,vy2,vz2,t2=np.loadtxt('out_gen_1e3.txt',unpack=True)

v=vx1
v2=vx2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(2.7,vxmean,vxerr,fmt='o',c='b',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(2.8,vxmean,vxerr,fmt='^',c='b',markersize=7)


v=vy1
v2=vy2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(2.9,vxmean,vxerr,fmt='o',c='r',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(3.0,vxmean,vxerr,fmt='^',c='r',markersize=7)


v=vz1
v2=vz2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(3.1,vxmean,vxerr,fmt='o',c='g',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(3.2,vxmean,vxerr,fmt='^',c='g',markersize=7)


vx1,vy1,vz1,t1,vx2,vy2,vz2,t2=np.loadtxt('out_gen_1e4.txt',unpack=True)

v=vx1
v2=vx2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(3.7,vxmean,vxerr,fmt='o',c='b',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(3.8,vxmean,vxerr,fmt='^',c='b',markersize=7)


v=vy1
v2=vy2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(3.9,vxmean,vxerr,fmt='o',c='r',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(4.0,vxmean,vxerr,fmt='^',c='r',markersize=7)


v=vz1
v2=vz2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(4.1,vxmean,vxerr,fmt='o',c='g',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(4.2,vxmean,vxerr,fmt='^',c='g',markersize=7)

vx1,vy1,vz1,t1,vx2,vy2,vz2,t2=np.loadtxt('out_gen_1e5.txt',unpack=True)

v=vx1
v2=vx2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(4.7,vxmean,vxerr,fmt='o',c='b',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(4.8,vxmean,vxerr,fmt='^',c='b',markersize=7)


v=vy1
v2=vy2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(4.9,vxmean,vxerr,fmt='o',c='r',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(5.0,vxmean,vxerr,fmt='^',c='r',markersize=7)


v=vz1
v2=vz2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(5.1,vxmean,vxerr,fmt='o',c='g',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(5.2,vxmean,vxerr,fmt='^',c='g',markersize=7)

vx1,vy1,vz1,t1,vx2,vy2,vz2,t2=np.loadtxt('out_gen_1e6.txt',unpack=True)

v=vx1
v2=vx2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(5.7,vxmean,vxerr,fmt='o',c='b',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(5.8,vxmean,vxerr,fmt='^',c='b',markersize=7)


v=vy1
v2=vy2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(5.9,vxmean,vxerr,fmt='o',c='r',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(6.0,vxmean,vxerr,fmt='^',c='r',markersize=7)


v=vz1
v2=vz2
vxmean=np.mean(v)
vxerr=np.std(v)
plt.errorbar(6.1,vxmean,vxerr,fmt='o',c='g',markersize=7)
vxmean=np.mean(v2)
vxerr=np.std(v2)
plt.errorbar(6.2,vxmean,vxerr,fmt='^',c='g',markersize=7)



plt.axvspan(2.6, 3.3, color='grey', alpha=0.3, lw=0)
plt.axvspan(3.6, 4.3, color='grey', alpha=0.3, lw=0)
plt.axvspan(4.6, 5.3, color='grey', alpha=0.3, lw=0)
plt.axvspan(5.6, 6.3, color='grey', alpha=0.3, lw=0)

plt.xlim(2,7)
plt.ylim(6.2,6.8)
plt.xlabel('Log(Npart)',fontsize=16)
plt.ylabel('Velocity dispersion [km/s]',fontsize=16)


plt.plot(-100,-100,'o',c='white',label='C extension')
plt.plot(-100,-100,'^',c='white',label='Pure Python')
plt.plot(-100,-100,'s',c='blue',label='X-axis')
plt.plot(-100,-100,'s',c='red',label='Y-axis')
plt.plot(-100,-100,'s',c='green',label='Z-axis')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(bbox_to_anchor=(1.125, 1.12),ncol=5,prop={'size':13},numpoints=1)
plt.savefig('Results_gen.pdf')
#plt.show()

'''
vx1,vy1,vz1,t1a,vx2,vy2,vz2,t2a=np.loadtxt('out_gen_1e3.txt',unpack=True)
vx1,vy1,vz1,t1b,vx2,vy2,vz2,t2b=np.loadtxt('out_gen_1e4.txt',unpack=True)
vx1,vy1,vz1,t1c,vx2,vy2,vz2,t2c=np.loadtxt('out_gen_1e5.txt',unpack=True)
vx1,vy1,vz1,t1d,vx2,vy2,vz2,t2d=np.loadtxt('out_gen_1e6.txt',unpack=True)

t1am=np.mean(t1a)
t1as=np.std(t1a)
t1bm=np.mean(t1b)
t1bs=np.std(t1b)
t1cm=np.mean(t1c)
t1cs=np.std(t1c)
t1dm=np.mean(t1d)
t1ds=np.std(t1d)

t2am=np.mean(t2a)
t2as=np.std(t2a)
t2bm=np.mean(t2b)
t2bs=np.std(t2b)
t2cm=np.mean(t2c)
t2cs=np.std(t2c)
t2dm=np.mean(t2d)
t2ds=np.std(t2d)



tcm=np.array([t1am,t1bm,t1cm,t1dm])
tcs=np.array([t1as,t1bs,t1cs,t1ds])
tpym=np.array([t2am,t2bm,t2cm,t2dm])
tpys=np.array([t2as,t2bs,t2cs,t2ds])
n=[3,4,5,6]



plt.errorbar(n,np.log10(tpym),tpys/(tpym*np.log(10)),fmt='-o',markersize=8,c='blue',lw=1.2,label='Pure Python')
plt.errorbar(n,np.log10(tcm),tcs/(tcm*np.log(10)),fmt='-o',markersize=8,c='red',lw=1.2,label='C extension')

plt.xlim(2,7)
plt.xlabel('Log(Npart)',fontsize=16)
plt.ylabel('Log(CPU Time [s])',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='best')
plt.savefig('Time_gen.pdf')
#plt.show()
'''