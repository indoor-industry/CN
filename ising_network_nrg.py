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

Jbeta = 0.2
steps = 10

#create lattice
def lattice(M, N):
    #lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.grid_2d_graph(M, N, periodic=False, create_using=None) #must be tested, also no visualzation because no positions?
    
    #dim = (10, 10, 10, 10) #works for cubic and n dimensional lattices too! (only energy plot, no visualization)
    #lattice = nx.grid_graph(dim, periodic=True)
    return lattice

G = lattice(10, 10)
N=0
for node in G:
    N+=1                #number of atoms
print(N)

#assign random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#run it
spinass(G)

#massive function for single step
E = []
def step(G, Jbeta):
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
    dEbeta=Jbeta*np.multiply(N,spinlist) #half the change in energy multiplied by beta (corresponds to the energy*beta)

    E.append(2*sum(dEbeta))

    #make it into a dictionary
    dEdict = {}
    i = 0
    for node in G:
        dEdict[node]=dEbeta[i]
        i+=1

    #Now flip every spin whose dE<0
    for node in G:
        if dEdict[node]<=0:
            spin[node]*=-1
        elif np.exp(-dEdict[node]) > np.random.rand():
            spin[node] *= -1

    #update spin values in graph
    nx.set_node_attributes(G, spin, 'spin')
    return E

#iterate steps and print
i=0
while i <= steps:
    step(G, Jbeta)
    i+=1

#energy per atom plot
E = step(G, Jbeta)
plt.plot(E)
plt.savefig('plots/plot({}).png'.format(i+1))
plt.show()