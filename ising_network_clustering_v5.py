import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from numba import jit

np.random.seed(42)

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
M = 20
N = 20
J = 0.2
B = 0.0
T = 1
steps = 20-1   #steps one step further than V4

#creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = generate_grid_pos(lattice, M, N) #use for 2D grid network

    return lattice, pos

def generate_grid_pos(G, M, N):
    p = []
    for m in range(M):
        for n in range(N):
            p.append((n, m))
    
    grid_pos = {}
    k = 0
    for node in G:
        grid_pos[node]=p[k]
        k+=1
    return grid_pos

#function that counts numer of nodes
def num(G):
    n=0
    for node in G:
        n+=1
    return n

#assign random spin up/down to nodes
def spinass(G, spinlist):
    k=0
    for node in G:
        G.nodes[node]['spin']=spinlist[k]
        k+=1

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
@jit(nopython=True)
def step(A_dense, spinlist, beta):

    l=0
    while l <= steps:
    
        A = np.copy(A_dense)

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)
    
        #What decides the flip is
        dE=-4*J*np.multiply(nnsum, spinlist) + 2*B*spinlist
    
        #Now flip every spin whose dE<0
        for offset in range(2):
            for i in range(offset,len(dE),2):
                if dE[i]<=0:
                    spinlist[i] *= -1
                elif np.exp(-dE[i]*beta) > np.random.rand():
                    spinlist[i] *= -1

        A = np.copy(A_dense)                        #redo this so that adjacency matrix and spins are on the same step

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        l += 1
        print(l)

    return A, spinlist

def clustering(A, s):
    for m in range(len(s)):
        for n in range(len(s)):
            if A[m, n] == s[m]:
                A[m, n] = 1
            else:
                A[m,n] = 0          #now matrix A represents which adjacent atoms have the same spin value
    return A

def main():
    G, pos = lattice(M, N)

    n = num(G)

    spinlist = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node
    
    spinass(G, spinlist)

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    #iterate some steps
    A, s = step(A_dense, spinlist, 1/T)

    spinass(G, spinlist)
    color = colormap(G)

    A_clust = clustering(A, s)

    G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A_clust)) #G2 only hasa the relevant edges

    den = nx.density(G2)
    print('Density = {}'.format(den))
    ne = nx.number_of_edges(G2)
    print('numer of edges = {}'.format(ne))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 1 %5.1f secs" % (time_elapsed))

    fig, ax = plt.subplots(1, 2)
    nx.draw(G2, node_color=color, node_size=20, ax=ax[0], edge_color='black', with_labels=False)
    ax[0].set_title('Clustering')
    nx.draw(G, node_color=color, node_size=20, ax=ax[1], edge_color='black', pos=pos, with_labels=False)
    ax[1].set_title('Lattice')
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))
    
    plt.show()

if __name__ == "__main__":
    main()