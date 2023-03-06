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

J = 0.01
beta = 40

#create lattice
def lattice(M, N):
    #lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    return lattice

G = lattice(20, 20)

#assign random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#run it
spinass(G)

#create color map
def colormap(G):
    color=[]
    for node in G:
        if G.nodes[node]['spin']==1:
            color.append('red')
        else:
            color.append('black')
    return color

#massive function for single step
def step(G):
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
    dE=J*np.multiply(N,spinlist) 

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


#first step otherwise it gets aligned to early
color = colormap(G)
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
plt.savefig('time_ev/img(0).png')


#iterate steps and print
i=0
while i <= 10:
    step(G)

    #update color map
    color = colormap(G)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)

    plt.savefig('time_ev/img({}).png'.format(i+1))

    i+=1