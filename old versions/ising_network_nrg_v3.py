import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

time_start = time.perf_counter()

J = -0.2                         #coupling constant
M = 20                            #lattice size MxN
N = 20
steps = 15                      #number of steps per given temperature
   
beta = np.logspace(1, 4, num=4, base=10.0)   #array of temperatures to sweep as of README

#function creates lattice
def lattice(M, N):
    #lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None) #must be tested, also no visualzation because no positions?
    
    #dim = (10, 10, 10, 10) #works for cubic and n dimensional lattices too! (only energy plot, no visualization)
    #lattice = nx.grid_graph(dim, periodic=True)
    return lattice

#function assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#function to step lattice in time
def step(G, beta):
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    #create ordered list of spins
    spin = nx.get_node_attributes(G, 'spin')
    spinlist = list(dict.values(spin))

    #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A = Adj.todense()
    for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
        for n in range(A.shape[1]):
            if A[m,n]==1:
                A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

    #sum over rows to get total spin of neighbouring atoms for each atom
    N = np.sum(A,axis=1).tolist()

    #What decides the flip is
    dE=2*J*np.multiply(N,spinlist) #change in energy

    E = 0.5*sum(dE)

    for offset in range(2):
        for i in range(offset,len(dE),2):
            if dE[i]<=0:
                G.nodes[i]['spin'] *= -1
            elif np.exp(-dE[i]*beta) > np.random.rand():
                G.nodes[i]['spin'] *= -1
    
    return G, E

#step function for one beta and collect energy in time 
def iter(G, beta, j):
    i=0
    E_time = np.asarray([])    
    while i <= steps:    
        G, E = step(G, beta[j])
        E_time = np.append(E_time, [E])
        i+=1
    return E_time

#run iter for different betas and plot energy/node for each
def beta_sweep(G, beta):
    n=0
    for node in G:     #count number of nodes
        n+=1
    
    for j in range(len(beta)):
        spinass(G)   #re-randomize the network for every new beta

        E_time = iter(G, beta, j)
    
        n_array = n*np.ones(len(E_time))
        plt.plot(E_time/n_array, label='beta={:10.1f}'.format(beta[j]))    #plot energy per atom

def main():
    #create lattice
    G = lattice(M, N)
    #assign random spins to lattice
    spinass(G)
    #iterate steps and sweep trough beta
    beta_sweep(G, beta)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.legend()
    plt.title('Energy against time')
    plt.legend(loc='upper right')
    plt.xlabel('timestep')
    plt.ylabel('energy per site [eV]')
    plt.show()

if __name__ =="__main__":
    main()