import numpy as np
def tiny_fgs1(T=64): return np.sin(np.linspace(0, 6.28, T)) + 0.01*np.random.RandomState(0).randn(T)
def tiny_airs(B=283): return np.linspace(0.1, 1.0, B)
