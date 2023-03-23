import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from numba import jit

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
M = 20
N = 20
J = -0.2
steps = 30
sample = 20

T_min = 0.1                        #min temperature to explore
T_max = 1.5                        #max temperature to explore

B_min = -1                         #min magnetic field to explore
B_max = 1                          #max magnetic field to explore

T = np.linspace(T_min, T_max, sample)   #temperature range to explore

ones = np.ones(len(T))                  #convert to inverse temperature
beta = ones/T

B = np.linspace(B_min, B_max, sample)   #External magnetic field range to explore

#creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)

    return lattice

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

#function for single step
@jit(nopython=True)
def step(A_dense, spinlist, beta, magfield):
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
        dE=-4*J*np.multiply(nnsum, spinlist) + 2*magfield*spinlist
    
        #Now flip every spin whose dE<0
        for offset in range(2):
            for i in range(offset,len(dE),2):
                if dE[i]<0:
                    spinlist[i] *= -1
                elif dE[i]==0:
                    continue
                elif np.exp(-dE[i]*beta) > np.random.rand():
                    spinlist[i] *= -1

        A = np.copy(A_dense)                        #redo this so that adjacency matrix and spins are on the same step

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        l += 1
        #print(l)

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
    G = lattice(M, N)

    n = num(G)

    spinlist = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node
    
    spinass(G, spinlist)

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()


    den_beta_J = np.empty((sample, sample))

    for i in range(len(B)):
        for j in range(len(beta)):              #run through different combinations of B and T
            #iterate some steps
            A, s = step(A_dense, spinlist, 1/T[j], B[i])

            spinass(G, spinlist)

            A_clust = clustering(A, s)

            G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A_clust)) #G2 only hasa the relevant edges

            den = nx.density(G2)

            den_beta_J[i, j] = den          #store density values
        print('{}/{}'.format(i+1, sample))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 1 %5.1f secs" % (time_elapsed))

    ext = [T_min, T_max, B_min, B_max]
    plt.imshow(den_beta_J, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    plt.colorbar()

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))
    
    plt.show()

if __name__ == "__main__":
    main()