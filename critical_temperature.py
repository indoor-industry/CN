import numpy as np
import matplotlib.pyplot as plt

J=1
N = M = 10 # N = M quasi sempre
B = 0

range_Tc = 100                       #range nel quale è stato preso il campione
T_max = 2.7
T_min = 2

Tc = (2*abs(J))/np.log(1+np.sqrt(2))
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.sqrt(3)                       #Critical temperature of triangular lattice at J = 1 



cv_data = np.genfromtxt('data/cv_{}x{}_square_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
xi_data = np.genfromtxt('data/xi_{}x{}_square_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
T_data = np.genfromtxt('data/T_{}x{}_square_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)

cv_data_h = np.genfromtxt('data/cv_{}x{}_hexagonal_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
xi_data_h = np.genfromtxt('data/xi_{}x{}_hexagonal_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
T_data_h = np.genfromtxt('data/T_{}x{}_hexagonal_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)

cv_data_t = np.genfromtxt('data/cv_{}x{}_triangular_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
xi_data_t = np.genfromtxt('data/xi_{}x{}_triangular_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)
T_data_t = np.genfromtxt('data/T_{}x{}_triangular_B={}.csv'.format(N, M, B), delimiter=',', skip_header=0)




Tc_from_cv = T_data[cv_data.argmax()]
Tc_from_xi = T_data[xi_data.argmax()]

Tc_from_cv_h = T_data_h[cv_data_h.argmax()]
Tc_from_xi_h = T_data_h[xi_data_h.argmax()]

Tc_from_cv_t = T_data_t[cv_data_t.argmax()]
Tc_from_xi_t = T_data_t[xi_data_t.argmax()]

Tc_mean = (Tc_from_cv + Tc_from_xi) / 2
Tc_mean_h = (Tc_from_cv_h + Tc_from_xi_h) / 2
Tc_mean_t = (Tc_from_cv_t + Tc_from_xi_t) / 2

print("Square critical temperature: " + str(Tc_mean))
print("Hexagonal critical temperature: " + str(Tc_mean_h))
print("Triangular critical temperature: " + str(Tc_mean_t))


