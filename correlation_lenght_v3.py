import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'ER'              #write square, triangular or hexagonal
p = 0.01

J = 1                             #spin coupling constant
B = 0                                #external magnetic field
M = 10                               #lattice size MxN
N = 10
steps = 30000                         #number of evolution steps per given temperature
max_r = 10
repeat = 10

Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1
Tc_ER = 78.5*p

if lattice_type == "square":
    T = np.linspace(0.5*Tc, 1.5*Tc, 10) 
elif lattice_type == "hexagonal":
    T = np.linspace(0.5*Tc_h, 1.5*Tc_h, 10) 
    Tc = Tc_h
elif lattice_type == "triangular":
    T = np.linspace(0.5*Tc_t, 1.5*Tc_t, 10) 
    Tc = Tc_t
elif lattice_type == "ER":
    T = np.linspace(0.5*Tc_ER, 1.5*Tc_ER, 10) 
    Tc = Tc_ER
else: print("Errore!")

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    elif lattice_type == 'ER':
        lattice = nx.erdos_renyi_graph(M*N, p, seed=None, directed=False)
    return lattice

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
            if j in dictionary_per_node_i.keys():           #for ER graphs some atoms may be disconnected, they shoud not count
                lenghts[i, j] = dictionary_per_node_i[j]
            else:
                lenghts[i, j] = max_r+100                   #set isolated atoms at a distance that will not be exlored (i.e. above max_r)
    return lenghts

def neighbour_number(lenghts, num):                         #returns array max_r x num where element radius, n is the number of atoms at a distance radius from atom n
    nn = np.zeros((max_r, num))
    for radius in range(max_r):
        for atom in range(num):
            for neighbour in range(num):
                if lenghts[atom, neighbour] == radius:
                    nn[radius, atom] += 1
    return nn

@jit(nopython=True)
def step(A_dense, beta, num, lenghts, spinlist, neighnum):
    
    corr_r_rep = np.empty((repeat, max_r))
    for rep in range(repeat):
    
        for l in range(steps):                               #evolve trough steps number of timesteps

            A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

            for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                for n in range(A.shape[1]):
                    if A[m,n]==1:
                        A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value
            
            #sum over rows to get total spin of neighbouring atoms for each atom
            nnsum = np.sum(A,axis=1)

            #What decides the flip is
            dE = 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

            i = np.random.randint(num)

            if dE[i]<=0:
                spinlist[i] *= -1
            elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
                spinlist[i] *= -1
        
        r = np.arange(0, max_r, 1)
        corr_atom_radius = np.empty((num, max_r))
        for atom in range(num):
            corr=0
            for radius in range(max_r):
                for neighbour in range(num):
                    if lenghts[atom, neighbour] == radius:
                        corr += spinlist[atom]*spinlist[neighbour]
                neigh_atom_per_radius = neighnum[:, atom]       #for each atom give number of neighbours as a function of radius
                neigh_atom_up_to_radius = np.sum(neigh_atom_per_radius[:radius+1])    #sum number of neighbours up to a certain radius    
                corr_atom_radius[atom, radius] = corr/neigh_atom_up_to_radius
        corr_r = np.sum(corr_atom_radius, axis=0)/num
        corr_r = np.abs(corr_r)

        corr_r_rep[rep] = corr_r

    corr_r_avg = np.sum(corr_r_rep, axis=0)/repeat  

    return corr_r_avg, r


def main():

    #create lattice
    G = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)

    spl = (list(nx.all_pairs_shortest_path_length(G)))
                
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    lenghts = distances(n, spl)
    neigh_num = neighbour_number(lenghts, n)

    rand_spin = np.random.choice(np.array([1, -1]), n)   #create random spins for nodes
    for j in range(len(T)):

        spinlist = np.copy(rand_spin)

        #iterate steps and sweep trough beta
        corr_r, r = step(A_dense, 1/T[j], n, lenghts, spinlist, neigh_num)

        plt.plot(r, corr_r, label=f'T={T[j]:.2f}')
        print(j)
    
    plt.xlabel('node distance r')
    plt.ylabel('$<<\sigma_i(x)\sigma_i(y)>_{|x-y|<r}$')
    plt.legend()
    plt.title(f'Tc = {Tc}')
    if lattice_type == 'ER':   
        plt.suptitle(f'type:{lattice_type}, J={J}, B={B}, ev_steps={steps}, no. atoms={n}, p={p}')
    else:
        plt.suptitle(f'type:{lattice_type}, J={J}, B={B}, ev_steps={steps}, no. atoms={n}')

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()