
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2,2,0,1],[3,2,1,0]])
mu= np.array([[1.25],[1.5]])

XMinusMu = X-mu
print(XMinusMu)
XMinusMu_transpose = XMinusMu.transpose()
product = np.dot(XMinusMu,XMinusMu.T)

sigma_calc = product/3 #instead of N, library uses N-1
print(sigma_calc)
sigma_func = np.cov(X)
print(sigma_func)

