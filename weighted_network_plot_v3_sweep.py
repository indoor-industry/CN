#ONLY FOR PERIODIC LATTICES

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'square'            #write square, triangular or hexagonal
J = 1                        #spin coupling constant
B = 0                     #external magnetic field
M = 10                          #lattice size MxN
N = 10
steps = 20000                      #number of evolution steps per given temperature

Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.sqrt(3)                       #Critical temperature of triangular lattice at J = 1 

if lattice_type == "square":
    T = np.linspace(0.5*Tc, 1.5*Tc, 20) 
elif lattice_type == "hexagonal":
    T = np.linspace(0.5*Tc_h, 1.5*Tc_h, 20) 
    Tc = Tc_h
elif lattice_type == "triangular":
    T = np.linspace(0.5*Tc_t, 1.5*Tc_t, 20) 
    Tc = Tc_t
else: print("Errore!")

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        return lattice, 3
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        return lattice, 6
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
        return lattice, 4
    #return lattice

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

#creates color map
def colormap(spinlist, num):
    color=[]
    for i in range(num):
        if spinlist[i]==1:
            color.append('red')
        else:
            color.append('black')
    return color

@jit(nopython=True)
def step(A_dense, beta, num, rand_spin):

    corr_matrix = np.zeros((num, num))
    
    #spinlist = np.random.choice(np.array([1, -1]), num)   #create random spins for nodes
    spinlist = np.copy(rand_spin)   #create random spins for nodes

    for l in range(steps):                               #evolve trough steps number of timesteps

        A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

        for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
            for n in range(A.shape[1]):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value
         
        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)

        #What decides the flip is
        dE = 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy
    
        #change spins if energetically favourable or according to thermal noise
        i = np.random.randint(num)
        if dE[i]<=0:
            spinlist[i] *= -1
        elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
            spinlist[i] *= -1
        
        for atom in range(num):
            for neighbour in range(num):
                corr_matrix[atom][neighbour]+=(spinlist[atom]*spinlist[neighbour])# - (M/num)**2

        norm_corr_matrix = corr_matrix/steps

        ei2 = np.sum(norm_corr_matrix**2, axis=0)-1          #sum over weights squared for each node (labeled j) to any other node
        ei = np.sum(np.abs(norm_corr_matrix), axis=0)-1      #sum over weights for each node (labeled j) to any other node

    return ei, ei2, norm_corr_matrix


def main():

    #create lattice
    G, nn_number = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)
                
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    rand_spin = np.random.choice(np.array([1, -1]), n)   #create random spins for nodes

    disparity = np.empty(len(T))
    density = np.empty(len(T))
    C = np.empty(len(T))
    D = np.empty(len(T))
    avg_dist = np.empty(len(T))
    avg_btw = np.empty(len(T))
    for a in range(len(T)):

        #iterate steps and sweep trough beta
        ei, ei2, corr_matrix = step(A_dense, 1/T[a], n, rand_spin)

        #create complete (absolute value of) correlation-weighted network
        G_corr = nx.create_empty_copy(G, with_data=True)
        for i in range(n):
            for j in range(n):
                if j<i:
                    G_corr.add_edge(i, j, weight=abs(corr_matrix[i][j]))

        #calculate disparity as of Sundhar
        disparity_i = ei2/ei**2
        disparity[a] = sum(disparity_i)/n
        
        #calculate density
        density_i = ei/(n-1)
        density[a] = sum(density_i)/n
        
        #calculate clustering coefficient
        clust_nominator = 0
        clust_denominator = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i!=j and j!=k and i!=k:
                        clust_nominator += corr_matrix[i][j]*corr_matrix[j][k]*corr_matrix[k][i]
                        clust_denominator += corr_matrix[i][k]*corr_matrix[j][k]
        clust_coefficient = clust_nominator/clust_denominator
        C[a] = clust_coefficient
        
        #calculate eccentricity and average over nodes, we call this average geodesic distance
        ecc = nx.eccentricity(G_corr, v=None, sp=None, weight='weight')
        avg_dist[a] = 1/(sum(ecc.values())/n)

        #calculate diameter (maximum eccentricity)
        #diameter = max(ecc.values())
        #D[a] = diameter

        #calculate betweenness centrality and average over atoms
        btw = nx.betweenness_centrality(G_corr, k=None, normalized=True, weight='weight', endpoints=False, seed=None)
        avg_btw[a] = sum(btw.values())/n

        #MORE EXPLICIT BUT EQUIVALENT CALCULATION OF DIAMETER USING DIJKSTRA ALGORITHM
        dspl = list(nx.all_pairs_dijkstra_path_length(G_corr))      #get djkistra distances for each node
        maxw = 0
        for s in range(n):                                          #find max distance
            dspl_s = dspl[s]
            w = list(dspl_s[1].values())
            for e in range(n):
                if w[e] > maxw:
                    maxw = w[e]
        D[a] = maxw                                                 #store as diameter

        print(a)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.scatter(T/Tc, density, color = 'orange')
    ax1.set_ylabel('density')
    ax1.set_xlabel('T/Tc')
    ax2.scatter(T/Tc, disparity, color = 'blue')
    ax2.set_ylabel('disparity')
    ax2.set_xlabel('T/Tc')
    ax3.scatter(T/Tc, C, color = 'green')
    ax3.set_ylabel('clustering coefficient')
    ax3.set_xlabel('T/Tc')
    ax4.scatter(T/Tc, D, color = 'black')
    ax4.set_ylabel('diameter')
    ax4.set_xlabel('T/Tc')
    ax5.scatter(T/Tc, avg_dist, color = 'brown')
    ax5.set_ylabel('avg geodesic distance')
    ax5.set_xlabel('T/Tc')
    ax6.scatter(T/Tc, avg_btw, color = 'violet')
    ax6.set_ylabel('betweenness centrality')
    ax6.set_xlabel('T/Tc')
    fig.suptitle('{} no.atoms={}  B={} J={}, ev_steps={}'.format(lattice_type, n, B, J, steps))
    fig.tight_layout()

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))
    
    plt.show()

if __name__ =="__main__":
    main()