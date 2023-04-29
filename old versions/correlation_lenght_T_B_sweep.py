#ONLY FOR PERIODIC LATTICES

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -1                           #spin coupling
M = 30                             #lattice size MxN
N = 30
steps = 40                         #number of timesteps of evolution per given temperature
sample = 50                        #number of samples between minimum and maximum values of B and T

T_min = 0.1                        #min temperature to explore
T_max = 1.5                        #max temperature to explore

B_min = -1                         #min magnetic field to explore
B_max = 1                          #max magnetic field to explore

T = np.linspace(T_min, T_max, sample)   #temperature range to explore

ones = np.ones(len(T))                  #convert to inverse temperature
beta = ones/T

B = np.linspace(B_min, B_max, sample)   #External magnetic field range to explore

max_r = 20

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        return lattice, 3
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        return lattice, 6
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
        return lattice, 4
    #return lattice

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

def distances(num, spl):
    lenghts = np.empty((num, num))
    for i in range(num):
        dictionary_per_node_i = spl[i][1]                   #returns an nxn matrix, element m,n is the distance between node m and node n
        for j in range(num):
            lenghts[i, j] = dictionary_per_node_i[j]
    return lenghts

@jit(nopython=True)
def step(A_dense, beta, num, nn_number, lenghts):

    max_corr_beta_J = np.empty((sample, sample))

    for i in range(len(B)):
        for j in range(len(beta)):              #run through different combinations of B and T
    
            spinlist = np.random.choice(np.array([1, -1]), num)   #create random spins for nodes
    
            l=0
            while l <= steps:                               #evolve trough steps number of timesteps

                A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

                for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                    for n in range(A.shape[1]):
                        if A[m,n]==1:
                            A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value
         
                #sum over rows to get total spin of neighbouring atoms for each atom
                nnsum = np.sum(A,axis=1)

                #What decides the flip is
                dE = -4*J*np.multiply(nnsum, spinlist) + 2*B[i]*spinlist    #change in energy
           
                #change spins if energetically favourable or according to thermal noise
                for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
                    for k in range(offset,len(dE),2):
                        if dE[k]<0:
                            spinlist[k] *= -1
                        elif dE[k]==0:
                            continue
                        elif np.exp(-dE[k]*beta[j]) > np.random.rand():     #thermal noise
                            spinlist[k] *= -1

                l+=1

            r = []
            corr_r = []
            for radius in np.arange(1, max_r+1):
                corr=0
                for atom in range(num):
                    mean = 0
                    for neighbour in range(num):
                        if lenghts[atom, neighbour] == radius:                  
                            corr += (spinlist[atom]*spinlist[neighbour])    #measures correlation for a given distance
                            mean += spinlist[neighbour]
                    corr2 = corr/(nn_number*radius)-mean**2/(nn_number*radius)**2
                corr3 = corr2/num
                r.append(radius)                        
                corr_r.append(abs(corr3))         #measures correlation in relation to distance  
            
            max_corr_beta_J[i, j] = max(corr_r)          #store energy values

        print(i+1)
    
    return max_corr_beta_J 

def main():

    #create lattice
    G, nn_number = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)

    spl = (list(nx.all_pairs_shortest_path_length(G)))
                
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    lenghts = distances(n, spl)

    #iterate steps and sweep trough beta
    max_corr_beta_J = step(A_dense, 1/T, n, nn_number, lenghts)

    ext = [T_min, T_max, B_min, B_max]
    
    plt.title('Maximum value of 2 point correlation [{}, size {}x{}, J={}, ev_steps={}]'.format(lattice_type, M, N, J, steps))
    plt.imshow(max_corr_beta_J, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    plt.colorbar()
    plt.ylabel('B')
    plt.xlabel('T')

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()