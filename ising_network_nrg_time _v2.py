import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import numba as nb

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -0.2                       #spin coupling constant
B = 0                       #external magnetic field
M = 30                          #lattice size MxN
N = 30
steps = 4000                      #number of evolution steps per given temperature
T = 0.1   #temperature range as of README

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

@jit(nopython=True)
def step(A_dense, beta, num):    

    spinlist = np.random.choice(np.asarray([-1, 1]), num)   #create random spins for nodes
        
    l=0
    E_time = nb.typed.List.empty_list(nb.f8)
    M_time = nb.typed.List.empty_list(nb.f8)
    while l <= steps:                               #evolve trough steps number of timesteps

        A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

        for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
            for n in range(A.shape[1]):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)

        #What decides the flip is
        dE = -4*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

        E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy
        M = np.sum(spinlist)                                       #total magnetisation

        choice = np.random.randint(len(spinlist))
        if dE[choice]<0:
            spinlist[choice]*=-1
        elif np.exp(-dE[choice]*beta) > np.random.rand():     #thermal noise
            spinlist[choice] *= -1


        #change spins if energetically favourable or according to thermal noise
        #for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
        #    for i in range(offset,len(dE),2):
        #        if dE[i]<0:
        #            spinlist[i] *= -1
        #        elif dE[i]==0:
        #            continue
        #        elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
        #            spinlist[i] *= -1

        E_time.append(E/num)            #list of energy trough time
        M_time.append(M/num)            #list of magnetisation trough time
        l+=1
    print(len(E_time))
    return E_time, M_time

def main():    

    #np.random.seed(seed)       #for debug purposes

    #create lattice
    G = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()
    #iterate steps and sweep trough beta
    E_time, M_time = step(A_dense, 1/T, n)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    t = [0]
    for q in range(steps):
        t.append(q+1)

    #plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.scatter(t, E_time, color = 'orange')
    ax1.set_ylabel('$E(time)$')
    ax2.scatter(t, M_time, color = 'blue')
    ax2.set_ylabel('$M(time)$')
    fig.suptitle('{} {}x{}  B={} J={}, ev_steps={}'.format(lattice_type, M, N, B, J, steps))
    fig.tight_layout()
    plt.show()

if __name__ =="__main__":
    main()