import numpy as np
from OpOp.particle import  particle

class Model:

    def __init__(self,e=0):
        self.e=e

    def potential(self,R,Z=0,Phi=0):
        m=np.sqrt(R*R+((Z*Z)/e))
        pot=0
        return pot

    def density(self,R,Z=0,Phi=0):
        m=np.sqrt(R*R+((Z*Z)/e))
        den=0
        return den

    def mass(self,R,Z=0,Phi=0):
        mass=0
        return mass


a=[particle.Particle() for i in range(3)]

a[0].Id=0
a[1].Id=0
a[2].Id=1

idlist=np.array([o.Id for o in a])

print(idlist)
b=tuple([np.sum(idlist==i) for i in range(6)])
print(b)