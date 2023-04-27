import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from numba import jit

time_start = time.perf_counter()

lattice_type = 'hexagonal'            #write square, triangular or hexagonal
M = 20
N = 20
J = 1
steps = 20000

Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.sqrt(3)                       #Critical temperature of triangular lattice at J = 1 
print(Tc_h)
print(Tc)

T_sample = 5
B_sample = 5

T_min = 0.5*Tc_h                        #min temperature to explore
T_max = 1.5*Tc_h                    #max temperature to explore

B_min = 0                         #min magnetic field to explore
B_max = 2                          #max magnetic field to explore

T = np.linspace(T_min, T_max, T_sample)   #temperature range to explore
B = np.linspace(B_min, B_max, B_sample)   #External magnetic field range to explore

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
def step(A_dense, spinlist, beta, magfield, num):

    for l in range(steps):
    
        A = np.copy(A_dense)

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)
    
        #What decides the flip is
        dE=-4*J*np.multiply(nnsum, spinlist) + 2*magfield*spinlist
        E = J*sum(np.multiply(nnsum, spinlist)) - magfield*sum(spinlist)   #total energy

        #Now flip every spin whose dE<0

        i = np.random.randint(num)

        if dE[i]<0:
            spinlist[i] *= -1
        elif np.exp(-dE[i]*beta) > np.random.rand():
            spinlist[i] *= -1

        A = np.copy(A_dense)                        #redo this so that adjacency matrix and spins are on the same step

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

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

    rand_spin = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node
    
    spinass(G, rand_spin)

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()


    den_beta_J = np.empty((B_sample, T_sample))
    btw_cen_beta_J = np.empty((B_sample, T_sample))

    for i in range(len(B)):
        for j in range(len(T)):              #run through different combinations of B and T

            spinlist = np.copy(rand_spin)

            #iterate some steps
            A, s = step(A_dense, spinlist, 1/T[j], B[i], n)

            spinass(G, spinlist)

            A_clust = clustering(A, s)

            G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A_clust)) #G2 only hasa the relevant edges

            den = nx.density(G2)
            btw = nx.betweenness_centrality(G2).get(7)

            den_beta_J[i, j] = den          #store density values
            btw_cen_beta_J[i, j] = btw

        print('{}/{}'.format(i+1, B_sample))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 1 %5.1f secs" % (time_elapsed))

    ext = [T_min/Tc_h, T_max/Tc_h, B_min, B_max]

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(lattice_type, M, N, J, steps))
    
    im1 = ax1.imshow(den_beta_J, cmap = 'binary', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('Density')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T/Tc')
    
    im2 = ax2.imshow(btw_cen_beta_J, cmap = 'binary', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax2.set_title('Betweeness')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B')
    ax2.set_xlabel('T/Tc')

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))
    
    plt.show()

if __name__ == "__main__":
    main()