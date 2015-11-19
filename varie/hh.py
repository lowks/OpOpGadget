import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def df_h(e):
    c=np.sqrt(e)/( (1-e)*(1-e) )

    a=(1-2*e)*(8*e*e -8*e -3)

    b=(3* (np.arcsin(np.sqrt(e))) )/(np.sqrt(e*(1-e)))


    return c*(a+b)


e=np.logspace(np.log10(0.01),np.log10(1),512)

plt.plot(e,np.log(df_h(e)/np.max(df_h(e))))
plt.show()