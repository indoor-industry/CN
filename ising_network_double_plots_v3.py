import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'hexagonal'            #write square, triangular or hexagonal
J = 1                           #spin coupling
M = 18                           #lattice size MxN
N = 18
steps = 15000                         #number of timesteps of evolution per given temperature
B_sample = 5                       #number of samples between minimum and maximum values of B NEEDS TO BE ODD FOR SENSIBLE RESULTS
T_sample = 5                        #number of samples between minimum and maximum values of T
eq_steps = 10000
Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice
Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1 


T_min = 0.5*Tc_h                        #min temperature to explore
T_max = 2*Tc_h                        #max temperature to explore

B_min = 0.5                         #min magnetic field to explore
B_max = 2.0                          #max magnetic field to explore

T = np.linspace(T_min, T_max, T_sample)   #temperature range to explore

ones = np.ones(len(T))                  #convert to inverse temperature
beta = ones/T

B = np.linspace(B_min, B_max, B_sample)   #External magnetic field range to explore

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#function that counts numer of nodes
def num(G):
    n=0
    for node in G:
        n+=1
    return n

#function to step lattice in time
@jit(nopython=True)
def step(A_dense, beta, B, num):

    def mean(list):
        return sum(list)/len(list)
    
    def mean_square(data):
        sum_of_squares=0
        for element in range(len(data)):
            sum_of_squares += data[element]**2
        return np.sqrt(sum_of_squares/len(data))

    M_beta_J = np.empty((B_sample, T_sample))
    E_beta_J = np.empty((B_sample, T_sample))

    rand_spin = np.random.choice(np.asarray([-1, 1]), num)  #generate random spins for each node

    for i in range(len(B)):
        for j in range(len(beta)):              #run through different combinations of B and T
            
            spinlist = np.copy(rand_spin)
            #spinlist = np.random.choice(np.asarray([-1, 1]), num)  #generate random spins for each node

            M_time=[]
            E_time=[]
            for k in range(steps):                       #evolve the system trough steps number of timesteps
                A = np.copy(A_dense)                #create a new copy of the adjacency matrix at every step otherwise it will be distorted by the rest of the function

                #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour    
                for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                    for n in range(A.shape[1]):
                        if A[m,n]==1:
                            A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

                #sum over rows to get total spin of neighbouring atoms for each atom
                nnsum = np.sum(A,axis=1)

                #What decides the flip is
                dE = 2*J*np.multiply(nnsum, spinlist) + 2*B[i]*spinlist    #change in energy

                E = -J*sum(np.multiply(nnsum, spinlist)) - B[i]*sum(spinlist)   #total energy
                M = np.sum(spinlist)                         #total magnetisation

                #update spin configuration if energetically favourable or if thermal fluctuations contribute
                l = np.random.randint(num)
                if dE[l]<=0:
                    spinlist[l] *= -1
                elif np.exp(-dE[l]*beta[j]) > np.random.rand():    #thermal noise
                    spinlist[l] *= -1   

                E_time.append(E)
                M_time.append(M)

            M_beta_J[i, j] = mean(M_time[eq_steps:])          #store magnetisation values
            E_beta_J[i, j] = mean_square(E_time[eq_steps:])/abs(J)          #store energy values

        print(i)

    return E_beta_J, M_beta_J

def main():
    #create lattice
    G = lattice(M, N)
    #label nodes as integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)
    #extract adjacency matrix from network of spins ans convert it to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()
    #run program
    E_beta_J, M_beta_J = step(A_dense, beta, B, n)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 1 %5.1f secs" % (time_elapsed))

    #store values as csv's
    #np.savetxt("E.csv", E_beta_J/n, delimiter=",")
    #np.savetxt("M.csv", M_beta_J/n, delimiter=",")

    #plot
    ext = [T_min/Tc_h, T_max/Tc_h, B_min, B_max]
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(lattice_type, M, N, J, steps))
    
    im1 = ax1.imshow(M_beta_J/n, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('M/site')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T/Tc')
    
    im2 = ax2.imshow(E_beta_J/n, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax2.set_title('(E/J)/site')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B')
    ax2.set_xlabel('T/Tc')
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()