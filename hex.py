import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

T = 2.5e2
k_b = 1.4e-23
J=1e-3 # coupling between spins
beta = 1/(k_b*T) #inverse temperature in units of energy

#create lattice
def lattice(M, N):
    lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    return lattice

G = lattice(50, 50)

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
def step(G, J, beta):
    #create ordered list of spins
    spin = nx.get_node_attributes(G, 'spin')

    spinlist = list(dict.values(spin))

    #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A = Adj.todense()
    for m in range(A.shape[1]):
        for n in range(A.shape[1]):
            if A[m,n]==1:
                A[m,n]=spinlist[n]

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
        elif np.exp(-dEdict[node] * beta) > np.random.rand():
            spin[node] *= -1

    #update spin values in graph
    nx.set_node_attributes(G, spin, 'spin')

#first step otherwise it gets aligned to early
color = colormap(G)
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, node_color=color, node_size=10, edge_color='white', pos=pos, with_labels=False)
#node_labels = nx.get_node_attributes(G,'spin')
#nx.draw_networkx_labels(G, pos, labels = node_labels)
#plt.show()
plt.savefig('img/img(00).png')


#iterate steps and print
i=0
while i <= 15:
    step(G, J, beta)

    #update color map
    color = colormap(G)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, node_color=color, node_size=10, edge_color='white', pos=pos, with_labels=False)
    #node_labels = nx.get_node_attributes(G,'spin')
    #nx.draw_networkx_labels(G, pos, labels = node_labels)
    #plt.show()
    plt.savefig('img/img({}).png'.format(i))

    i+=1