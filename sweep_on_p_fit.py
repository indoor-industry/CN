import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy as sp

M=N=10
Tc = np.loadtxt('p sweep plots/psweep_Tc_Cv.csv')
p = np.loadtxt('p sweep plots/psweep_p.csv')
var = np.loadtxt('p sweep plots/var_cv_in_p.csv')

stdev = np.sqrt(var)

line = sp.stats.linregress(p, Tc)
slope = line[0]
intercept = line[1]

plt.scatter(p, Tc)
#plt.errorbar(p, Tc, stdev, fmt='o')
plt.plot(p, slope*p+intercept, label='fit')

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
text1 = f'slope = {slope}'
text2 = f'intercept = {intercept}'
plt.legend([extra, extra],[text1, text2], title='Fit parameters')

plt.title('ER Critical temperature')
plt.xlabel('p')
plt.ylabel('Tc')
plt.show()