import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp(x, xi):
    return np.exp(-x/xi)

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
    
    popt, pcov = curve_fit(exp, r, corr_T[t])
    perr = np.sqrt(np.diag(pcov))
    xis.append(popt[0])
    xis_err.append(perr[0])

    fit = [exp(var, popt[0]) for var in r]
    plt.plot(r, fit)

plt.show()

param, errors = curve_fit(critical, T, xis)

print(param)

crit = [critical(temp, param[0], param[1]) for temp in T]

plt.plot(T, crit)
plt.errorbar(T, xis, xis_err, fmt='o')
plt.show()