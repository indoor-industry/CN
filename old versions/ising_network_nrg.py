#Typical values of constants
#J usually from e-4 (magnetic interaction) to 1 ev (electrostatic interaction)
#T usually from 0 to e3 K
#k=e-4 (in units of ev/K)
#beta follows as:
#beta from 10 to inf
#hence Jbeta from e-3 to inf

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(42) #debug

time_start = time.perf_counter()

J = 0.1

#create lattice
def lattice(M, N):
    lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None) #must be tested, also no visualzation because no positions?
    
    #dim = (10, 10, 10, 10) #works for cubic and n dimensional lattices too! (only energy plot, no visualization)
    #lattice = nx.grid_graph(dim, periodic=True)
    return lattice

M = 20
N = 20
G = lattice(M, N)

n=0
for node in G:
    n+=1                #count number of atoms

#assign random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#run it
spinass(G)

#massive function for single step
def step(G, beta):
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

    #make it into a dictionary
    dEdict = {}
    i = 0
    for node in G:
        dEdict[node]=dE[i]
        i+=1

    #Now flip every spin whose dE<0
    for node in G:
        if dEdict[node]<=0:
            spin[node]*=-1
        elif np.exp(-dEdict[node]*beta) > np.random.rand():
            spin[node] *= -1

    #update spin values in graph
    nx.set_node_attributes(G, spin, 'spin')
    return E

#iterate steps and print
beta = np.linspace(0.1, 100, 5)

j=0
for j in range(len(beta)):
    spinass(G)   #re-randomize the network for every new beta

    i=0
    E_time = []
    while i <= 15:                   #15 steps is usually enough to reach stability
        E_time.append(step(G, beta[j]))
        i+=1

    n_array = n*np.ones(len(E_time))
    plt.plot(E_time/n_array, label='beta={:10.1f}'.format(beta[j]))    #plot energy per atom

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.legend()
plt.title('Tuning4Life')
plt.legend(loc='upper right')
plt.xlabel('timestep')
plt.ylabel('energy per site')
plt.savefig('plots/img({}x{},J={}).png'.format(M, N, J))
plt.show()