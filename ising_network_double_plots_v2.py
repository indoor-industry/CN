import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -0.2                           #spin coupling
M = 30                             #lattice size MxN
N = 30
steps = 30                         #number of timesteps of evolution per given temperature
sample = 50                        #number of samples between minimum and maximum values of B and T

T_min = 0.1                        #min temperature to explore
T_max = 1.5                        #max temperature to explore

B_min = -1                         #min magnetic field to explore
B_max = 1                          #max magnetic field to explore

T = np.linspace(T_min, T_max, sample)   #temperature range to explore

ones = np.ones(len(T))                  #convert to inverse temperature
beta = ones/T

B = np.linspace(B_min, B_max, sample)   #External magnetic field range to explore

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
    M_beta_J = np.empty((sample, sample))
    E_beta_J = np.empty((sample, sample))

    for i in range(len(B)):
        for j in range(len(beta)):              #run through different combinations of B and T

            spinlist = np.random.choice(np.asarray([-1, 1]), num)  #generate random spins for each node

            k=0
            while k <= steps:                       #evolve the system trough steps number of timesteps
                A = np.copy(A_dense)                #create a new copy of the adjacency matrix at every step otherwise it will be distorted by the rest of the function

                #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour    
                for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                    for n in range(A.shape[1]):
                        if A[m,n]==1:
                            A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

                #sum over rows to get total spin of neighbouring atoms for each atom
                nnsum = np.sum(A,axis=1)

                #What decides the flip is
                dE = -4*J*np.multiply(nnsum, spinlist) + 2*B[i]*spinlist    #change in energy

                E = J*sum(np.multiply(nnsum, spinlist)) - B[i]*sum(spinlist)   #total energy
                M = np.sum(spinlist)                         #total magnetisation

                #update spin configuration if energetically favourable or if thermal fluctuations contribute
                for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering trough the lattice
                    for l in range(offset,len(dE),2):
                        if dE[l]<=0:
                            spinlist[l] *= -1
                        elif np.exp(-dE[l]*beta[j]) > np.random.rand():    #thermal noise
                            spinlist[l] *= -1   
                k+=1

            M_beta_J[i, j] = M          #store magnetisation values
            E_beta_J[i, j] = E          #store energy values

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
    ext = [T_min, T_max, B_min, B_max]
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(lattice_type, M, N, J, steps))
    
    im1 = ax1.imshow(M_beta_J/n, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('M/site')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T')
    
    im2 = ax2.imshow(E_beta_J/n, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax2.set_title('E/site')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B')
    ax2.set_xlabel('T')
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()