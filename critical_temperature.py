import numpy as np
import matplotlib.pyplot as plt

M_10 = np.genfromtxt('data/M_10x10_square_B=0.csv', delimiter=',', skip_header=0)
M_20 = np.genfromtxt('data/M_20x20_square_B=0.csv', delimiter=',', skip_header=0)
#etc

T = np.genfromtxt('data/T_5x5_square_B=0.csv', delimiter=',', skip_header=0)

plt.scatter(T, M_20)
plt.show()

#find max of chi
#such T is T0 for various lenghts

#fit this to find critical temperature and exponent
def T0_L(L, T_c, x_0, ni):
    return T_c*(1+x_0*L**(-1/ni))
