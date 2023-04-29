import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'square'             #write square, triangular or hexagonal
J = 1                            #spin coupling constant

Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice
print(Tc)

T_sample = 20
B_sample = 20

B_min = 0
B_max = 0.4
B = np.linspace(B_min, B_max, B_sample)                     #external magnetic field

M = 5                              #lattice size MxN
N = 5
steps = 20000                        #number of evolution steps per given temperature

T_min = 0.7*Tc
T_max = 1.5*Tc
T = np.linspace(T_min, T_max, T_sample)

ones = np.ones(len(T))
beta = ones/T

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
def step(A_dense, beta, num, B):

    corr_matrix = np.zeros((num, num))
    
    spinlist = np.random.choice(np.array([1, -1]), num) #create random spins for nodes
    
    for l in range(steps):                              #evolve trough steps number of timesteps

        A = np.copy(A_dense)                            #take new copy of adj. matrix at each step because it gets changed trough the function

        for m in range(A.shape[1]):                     #A.shape[1] gives number of nodes
            for n in range(A.shape[1]):
                if A[m,n]==1:
                    A[m,n]=spinlist[n]                  #assigned to every element in the adj matrix the corresponding node spin value
         
        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)

        #What decides the flip is
        dE = 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist        #change in energy

        #change spins if energetically favourable or according to thermal noise
        i = np.random.randint(num)
        if dE[i]<=0:
            spinlist[i] *= -1
        elif np.exp(-dE[i]*beta) > np.random.rand():         #thermal noise
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

    disparity = np.empty((B_sample, T_sample))
    density = np.empty((B_sample, T_sample))
    C = np.empty((B_sample, T_sample))
    avg_dist = np.empty((B_sample, T_sample))
    D = np.empty((B_sample, T_sample))

    #avg_btw = np.empty(len(T))
    for r in range(len(B)):
        for a in range(len(beta)):
        
            #iterate steps and sweep trough beta
            ei, ei2, corr_matrix = step(A_dense, 1/T[a], n, B[r])

            #create complete (absolute value of) correlation-weighted network
            G_corr = nx.create_empty_copy(G, with_data=True)
            for i in range(n):
                for j in range(n):
                    if j<i:
                        G_corr.add_edge(i, j, weight=abs(corr_matrix[i][j]))

            #calculate disparity as of Sundhar
            disparity_i = ei2/ei**2
            disparity[r, a] = sum(disparity_i)/n

            #calculate density
            density_i = ei/(n-1)
            density[r, a] = sum(density_i)/n
        
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
            C[r, a] = clust_coefficient
        
            #calculate eccentricity and average over nodes, we call this average geodesic distance
            ecc = nx.eccentricity(G_corr, v=None, sp=None, weight='weight')
            avg_dist[r, a] = sum(ecc.values())/n

            #caluclate diameter (maximum eccentricity)
            diameter = max(ecc.values())
            D[r, a] = diameter

            #calculate betweenness centrality and average over atoms
            #btw = nx.betweenness_centrality(G_corr, k=None, normalized=True, weight='weight', endpoints=False, seed=None)
            #avg_btw[a] = sum(btw.values())/n           
                   
        print(r)

    ext = [T_min/Tc, T_max/Tc, B_min, B_max]
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(lattice_type, M, N, J, steps))
    
    im1 = ax1.imshow(disparity, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('disparity')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T/Tc')
    
    im2 = ax2.imshow(density, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax2.set_title('density')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B')
    ax2.set_xlabel('T')

    im3 = ax3.imshow(C, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax3.set_title('C')
    fig.colorbar(im3, ax=ax3)
    ax3.set_ylabel('B')
    ax3.set_xlabel('T')

    im4 = ax4.imshow(D, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax4.set_title('D')
    fig.colorbar(im4, ax=ax4)
    ax4.set_ylabel('B')
    ax4.set_xlabel('T')

    fig.tight_layout()

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()