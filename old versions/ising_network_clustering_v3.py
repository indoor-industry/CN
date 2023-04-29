import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse

time_start = time.perf_counter()

M = 20
N = 20
J = 0.2
beta = 10
steps = 15

#create lattice
def lattice(M, N):
    #lattice = nx.hexagonal_lattice_graph(M, N, periodic=False, with_positions=True, create_using=None)
    lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#assign random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#create color map
def colormap(G):
    color=[]
    for node in G:
        if G.nodes[node]['spin']==1:
            color.append('red')
        else:
            color.append('black')
    return color

#function for single step
def step(G):
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    #create ordered list of spins
    spin = nx.get_node_attributes(G, 'spin')
    spinlist = list(dict.values(spin))
    
    #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A = Adj.todense()

    for m in range(len(spinlist)):
        for n in range(len(spinlist)):
            if A[m,n]==1:
                A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

    color = colormap(G)

    #sum over rows to get total spin of neighbouring atoms for each atom
    N = np.sum(A,axis=1).tolist()
    
    #What decides the flip is
    dE=2*J*np.multiply(N,spinlist) 
    
    #Now flip every spin whose dE<0
    for offset in range(2):
        for i in range(offset,len(dE),2):
            if dE[i]<=0:
                G.nodes[i]['spin'] *= -1
            elif np.exp(-dE[i]*beta) > np.random.rand():
                G.nodes[i]['spin'] *= -1

    return G, A, spinlist, color

def clustering(A, s):
    for m in range(len(s)):
        for n in range(len(s)):
            if A[m, n] == s[m]:
                A[m, n] = 1
            else:
                A[m,n] = 0          #now matrix A represents which adjacent atoms have the same spin value
    return A

def main():
    G = lattice(M, N)

    #run it
    spinass(G)

    #iterate some steps
    i=0
    while i <= steps:
        G, A, s, color = step(G)
        i+=1

    A = clustering(A, s)

    G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A)) #G2 only hasa the relevant edges

    pos = nx.get_node_attributes(G, 'pos')

    #clust = nx.clustering(G2)
    #print(clust)
    #bc = nx.betweenness_centrality(G2)
    #print(bc)
    #den = nx.density(G2)
    #print(den)
    ne = nx.number_of_edges(G2)
    print(ne)
    nn = nx.number_of_nodes(G2)
    print(nn)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    fig, ax = plt.subplots(1, 2)
    nx.draw(G2, node_color=color, node_size=20, ax=ax[0], edge_color='black', with_labels=False)
    ax[0].set_title('Clustering')
    nx.draw(G, node_color=color, node_size=20, ax=ax[1], edge_color='black', pos=pos, with_labels=False)
    ax[1].set_title('Lattice')
    plt.show()

if __name__ == "__main__":
    main()