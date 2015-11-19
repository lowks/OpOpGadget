from numpy.ctypeslib import ndpointer
import numpy as np
import ctypes as ct

lib=ct.cdll.LoadLibrary("/Users/Giuliano/PycharmProjects/OpOp/ctest.so")
fun=lib.cfun
fun.restype=None
fun.argtypes=[ndpointer(ct.c_double, flags="C_CONTIGUOUS"),ndpointer(ct.c_double, flags="C_CONTIGUOUS")]
indata=np.ones(10)
outdata=np.empty(10)
print(outdata)
fun(indata,outdata)
print(outdata)