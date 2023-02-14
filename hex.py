import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#amount of hexagons
M = 20
N = 20

#create lattice
G = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
#get position data
pos = nx.get_node_attributes(G, 'pos') #change this dictionary to get a more even shaped lattice (just an aesthetic thing)

#assign random spin up/down to nodes and create color map
color=[]
for node in G:
    s = np.random.choice([-1, 1])
    G.nodes[node]['spin']=s
    if s==1:
        color.append('gray')
    else:
        color.append('black')

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
#print(A)

#sum over rows to get total spin of neighbouring atoms for each atom
N = np.sum(A,axis=1).tolist()
#What decides the flip is
dE=np.multiply(N,spinlist)

print(dE)
#Now flip every spin whose dE<0
for node in G:
    


#draw (removed edges since they were ugly)
nx.draw(G, node_color=color, edge_color='white', pos=pos, with_labels=False)
#node_labels = nx.get_node_attributes(G,'spin')
#nx.draw_networkx_labels(G, pos, labels = node_labels)
plt.show()
