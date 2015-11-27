import numpy as np
from astropy.constants import G as conG


class Plummer:

    def __init__(self,rc,Mtot,G='kpc km2 / (M_sun s2)'):
        """
        Analytic Plummer model
        :param rc: Plummer scale length
        :param Mtot:  Plummer total mass
        :param G: Value of the gravitational constant G, it can be a number of a string.
                    If G=1, the physical value of the potential will be Phi/G.
                    If string it must follow the rule of the unity of the module.astropy constants.
                    E.g. to have G in unit of kpc3/Msun s2, the input string is 'kpc3 / (M_sun s2)'
                    See http://astrofrog-debug.readthedocs.org/en/latest/constants/

        :return:
        """
        self.rc=rc
        self.Mtot=Mtot
        if isinstance(G,float) or isinstance(G,int): self.G=G
        else:
            GG=conG.to(G)
            self.G=GG.value

        self._use_nparray=True
        self._densnorm=(3*Mtot)/(4*np.pi*rc*rc*rc)
        self._potnorm=self.G*Mtot

    def _evaluatedens(R):

        dd= (1 + ( (R*R) / (self.rc*self.rc) ) )

        return self._densnorm*(dd)**(-2.5)

    def _evaluatemass(R):

        x=R/self.rc

        return self.Mtot*( (x*x*x) / (1+x*x)  )

    def _evaluatepot(R):

        den=np.sqrt(R*R + self.rc*self.rc)

        return self._potnorm/den