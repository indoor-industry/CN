import numpy as np
import matplotlib.pyplot as plt
import heapq as he

J=1
N_s = M_s = 10 # N = M quasi sempre
N_t = M_t = 10

N_h = 6
M_h = 8

B = 0

range_Tc = 100                       #range nel quale è stato preso il campione
T_max = 2.7
T_min = 2
n_of_items = 2
n_of_items_h = 2
n_of_items_t = 2

Tc = (2*abs(J))/np.log(1+np.sqrt(2))
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1 



cv_data = np.genfromtxt('data/cv_{}x{}_square_B={}.csv'.format(N_s, M_s, B), delimiter=',', skip_header=0)
xi_data = np.genfromtxt('data/xi_{}x{}_square_B={}.csv'.format(N_s, M_s, B), delimiter=',', skip_header=0)
T_data = np.genfromtxt('data/T_{}x{}_square_B={}.csv'.format(N_s, M_s, B), delimiter=',', skip_header=0)

cv_data_h = np.genfromtxt('data/cv_{}x{}_hexagonal_B={}.csv'.format(N_h, M_h, B), delimiter=',', skip_header=0)
xi_data_h = np.genfromtxt('data/xi_{}x{}_hexagonal_B={}.csv'.format(N_h, M_h, B), delimiter=',', skip_header=0)
T_data_h = np.genfromtxt('data/T_{}x{}_hexagonal_B={}.csv'.format(N_h, M_h, B), delimiter=',', skip_header=0)

cv_data_t = np.genfromtxt('data/cv_{}x{}_triangular_B={}.csv'.format(N_t, M_t, B), delimiter=',', skip_header=0)
xi_data_t = np.genfromtxt('data/xi_{}x{}_triangular_B={}.csv'.format(N_t, M_t, B), delimiter=',', skip_header=0)
T_data_t = np.genfromtxt('data/T_{}x{}_triangular_B={}.csv'.format(N_t, M_t, B), delimiter=',', skip_header=0)




Tc_from_cv = T_data[cv_data.argmax()]
Tc_from_xi = T_data[xi_data.argmax()]

Tc_from_cv_h = T_data_h[cv_data_h.argmax()]
Tc_from_xi_h = T_data_h[xi_data_h.argmax()]

Tc_from_cv_t = T_data_t[cv_data_t.argmax()]
Tc_from_xi_t = T_data_t[xi_data_t.argmax()]

#Tc_mean = (Tc_from_cv + Tc_from_xi) / 2
#Tc_mean_h = (Tc_from_cv_h + Tc_from_xi_h) / 2
#Tc_mean_t = (Tc_from_cv_t + Tc_from_xi_t) / 2

Tc_mean = Tc_from_cv
Tc_mean_h = Tc_from_cv_h
Tc_mean_t = Tc_from_cv_t

print("Square critical temperature: " + str(Tc_mean))
print("Hexagonal critical temperature: " + str(Tc_mean_h))
print("Triangular critical temperature: " + str(Tc_mean_t))
#print di quelle teoriche
print("...............................\n Valori teorici:")

print(Tc)
print(Tc_h)
print(Tc_t)

print("...............................")


#Proviamo a fare una roba più fine
def Tc_arr_buider(max_arr, data_ndarray, T_data_ndarray):
    T_arr = []
    for i in max_arr:
        T_arr.append(T_data_ndarray[list(data_ndarray).index(i)])
    return T_arr

def two_list_mean(list_1, list_2):
    return (sum(list_1) + sum(list_2)) / (len(list_1)+ len(list_2))

#Square
cv_max_array = he.nlargest(n_of_items, cv_data)
xi_max_array = he.nlargest(n_of_items, xi_data)

Tc_cv_array = Tc_arr_buider(cv_max_array, cv_data, T_data)
Tc_xi_array = Tc_arr_buider(xi_max_array, xi_data, T_data)

Tc_mean = two_list_mean(Tc_cv_array, Tc_xi_array)
#hexagonal
cv_max_array_h = he.nlargest(n_of_items_h, cv_data_h)
xi_max_array_h = he.nlargest(n_of_items_h, xi_data_h)

Tc_cv_array_h = Tc_arr_buider(cv_max_array_h, cv_data_h, T_data_h)
Tc_xi_array_h = Tc_arr_buider(xi_max_array_h, xi_data_h, T_data_h)

Tc_mean_h = two_list_mean(Tc_cv_array_h, Tc_xi_array_h)
#triangular
cv_max_array_t = he.nlargest(n_of_items_t, cv_data_t)
xi_max_array_t = he.nlargest(n_of_items_t, xi_data_t)

Tc_cv_array_t = Tc_arr_buider(cv_max_array_t, cv_data_t, T_data_t)
Tc_xi_array_t = Tc_arr_buider(xi_max_array_t, xi_data_t, T_data_t)

Tc_mean_t = two_list_mean(Tc_cv_array_t, Tc_xi_array_t)


print(Tc_mean)
print(Tc_mean_h)
print(Tc_mean_t)


