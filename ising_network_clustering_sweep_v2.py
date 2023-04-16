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
J = -1
B = 0
steps = 1000

Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice
print(Tc)

T = np.linspace(0.01*Tc, 0.4*Tc, 50)
ones = np.ones(len(T))
beta = ones/T

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
def step(A_dense, spinlist, beta, num):

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
        E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy

        #Now flip every spin whose dE<0
        for offset in range(2):
            for i in range(offset,len(dE),2):
                if dE[i]<0:
                    spinlist[i] *= -1
                elif dE[i]==0:
                    if np.exp(-(E/num)*beta) > np.random.rand():
                        spinlist[i] *= -1
                    else:
                        continue
                elif np.exp(-dE[i]*beta) > np.random.rand():
                    spinlist[i] *= -1

        A = np.copy(A_dense)                        #redo this so that adjacency matrix and spins are on the same step

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        l += 1

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

    def mean(list):
        return sum(list)/len(list)

    G, pos = lattice(M, N)

    n = num(G)

    rand_spin = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node
    
    spinass(G, rand_spin)

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    diameter_of_giant_component_T = []
    density_T = []
    btw_T = []
    cc_T = []
    for i in range(len(T)):

        spinlist = np.copy(rand_spin)

        #iterate some steps
        A, s = step(A_dense, spinlist, beta[i], n)
        A_clust = clustering(A, s)
        G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A_clust)) #G2 only hasa the relevant edges
        
        ne = nx.number_of_edges(G2)
        density = nx.density(G2)
        cc = nx.number_connected_components(G2)
        
        #calculate the average betweennes centrality of nodes
        full_btw = nx.betweenness_centrality(G2)
        mean_btw = mean(full_btw.values())

        Gcc = sorted(nx.connected_components(G2), key=len, reverse=True)
        giant = G.subgraph(Gcc[0])
        diameter_of_giant_component_T.append(nx.diameter(giant))

        density_T.append(density)
        btw_T.append(mean_btw)
        cc_T.append(cc)

        print(i)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T/Tc, diameter_of_giant_component_T, color = 'orange')
    ax1.set_ylabel('diam. of giant component(T/Tc)')
    ax2.scatter(T/Tc, density_T, color = 'blue')
    ax2.set_ylabel('density(T/Tc)')
    ax3.scatter(T/Tc, btw_T, color = 'green')
    ax3.set_ylabel('|betweenness centrality|(T/Tc)')
    ax4.scatter(T/Tc, cc_T, color = 'black')
    ax4.set_ylabel('connected components(T/Tc)')
    fig.suptitle('{} {}x{}  B={} J={}, ev_steps={}'.format(lattice_type, M, N, B, J, steps))
    fig.tight_layout()
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))
    
    plt.show()


if __name__ == "__main__":
    main()