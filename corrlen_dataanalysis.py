import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp(x, a, b):
    return a*np.exp(-b*x)

def critical(T, Tc, nu):
    return abs(T/(T-Tc))**nu

T = np.loadtxt('corr_data/T.csv')
r = np.loadtxt('corr_data/r.csv')

xis = []
xis_err = []
corr_T = np.empty((len(T), len(r)))
for t in range(len(T)):
    corr_T[t] = np.loadtxt(f'corr_data/corr_r {t}.csv')
    plt.scatter(r, corr_T[t])
    
    popt, pcov = curve_fit(exp, r, corr_T[t], p0=(0.5, 2))
    perr = np.sqrt(np.diag(pcov))
    xis.append(1/popt[1])
    xis_err.append(perr[1])

    fit = [exp(var, popt[0], popt[1]) for var in r]
    plt.plot(r, fit)

plt.show()

param, errors = curve_fit(critical, T, xis, p0=(2.25, 1))

print(param)

crit = [critical(temp, param[0], param[1]) for temp in T]
plt.plot(T, crit)

plt.errorbar(T, xis, xis_err, fmt='o')
plt.show()