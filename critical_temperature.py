import numpy as np
import matplotlib.pyplot as plt

J=1
range_Tc = 100                       #range nel quale è stato preso il campione
T_max = 2.7
T_min = 2
Tc = (2*abs(J))/np.log(1+np.sqrt(2))
Error_Tc = (T_max - T_min) / 2

cv_data = np.genfromtxt('data/cv_10x10_square_B=0.csv', delimiter=',', skip_header=0)
xi_data = np.genfromtxt('data/xi_10x10_square_B=0.csv', delimiter=',', skip_header=0)
T_data = np.genfromtxt('data/T_10x10_square_B=0.csv', delimiter=',', skip_header=0)

cv_data_h = np.genfromtxt('data/cv_12x12_hexagonal_B=0.csv', delimiter=',', skip_header=0)
xi_data_h = np.genfromtxt('data/xi_12x12_hexagonal_B=0.csv', delimiter=',', skip_header=0)
T_data_h = np.genfromtxt('data/T_12x12_hexagonal_B=0.csv', delimiter=',', skip_header=0)


Tc_from_cv = T_data[cv_data.argmax()]
Tc_from_xi = T_data[xi_data.argmax()]

Tc_from_cv_h = T_data_h[cv_data_h.argmax()]
Tc_from_xi_h = T_data_h[xi_data_h.argmax()]

Tc_mean = (Tc_from_cv + Tc_from_xi) / 2
Tc_mean_h = (Tc_from_cv_h + Tc_from_xi_h) / 2

print("Square critical temperature: " + str(Tc_mean))
print("Hexagonal critical temperature: " + str(Tc_mean_h))



#plt.scatter(T_data, cv_data)
#plt.show()

#find max of chi
#such T is T0 for various lenghts

#fit this to find critical temperature and exponent
#def T0_L(L, T_c, x_0, ni):
#    return T_c*(1+x_0*L**(-1/ni))
