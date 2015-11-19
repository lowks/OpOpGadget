import numpy as np

def simpson(y,a,b):
    m=len(y)/3
    const=(b-a)/(3*m)
