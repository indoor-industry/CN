#ONLY FOR PERIODIC LATTICES

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'PT86'            #write square, triangular or hexagonal

p = 0.04                            #for ER graph

J = 1                        #spin coupling constant
B = 0                     #external magnetic field
M = 20                          #lattice size MxN
N = 20
steps = 20000                      #number of evolution steps per given temperature

if lattice_type == "hexagonal":
    Tc = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
elif lattice_type == 'square':
    Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
elif lattice_type == "triangular": 
    Tc = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1 
else:
    Tc = 1

T = 2.2*Tc

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    elif lattice_type == 'ER':
        lattice = nx.erdos_renyi_graph(M*N, p, seed=None, directed=False)
    elif lattice_type == 'PT86':
        edges = np.loadtxt('PT/nnbond86.txt')
        adj = np.zeros((86, 86))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT226':
        edges = np.loadtxt('PT/nnbond226.txt')
        adj = np.zeros((226, 226))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT31':
        edges = np.loadtxt('PT/nnbond31.txt')
        adj = np.zeros((31, 31))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1
        lattice = nx.from_numpy_array(adj)
     
    elif lattice_type == 'PT601':
        edges = np.loadtxt('PT/nnbond601.txt')
        adj = np.zeros((601, 601))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1
        lattice = nx.from_numpy_array(adj)
    
    return lattice

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
def step(A_dense, beta, num):

    corr_matrix = np.zeros((num, num))
    
    spinlist = np.random.choice(np.array([1, -1]), num)   #create random spins for nodes
    
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
        M = np.sum(spinlist)
    
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
    
    di = []
    for a in range(num):
        den=0
        for q in range(num):
            if q != a:
                den += norm_corr_matrix[a][q]
        di.append(den/(num-1))
    
    density = sum(di)/num

    return norm_corr_matrix, spinlist, density


def main():

    #create lattice
    G = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)
                
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    #iterate steps and sweep trough beta
    corr_matrix, spins, density = step(A_dense, 1/T, n)

    print(density)

    G_corr = nx.create_empty_copy(G, with_data=True)

    for i in range(n):
        for j in range(n):
            if j<i:
                G_corr.add_edge(i, j, weight=corr_matrix[i][j])

    w = nx.get_edge_attributes(G_corr, 'weight')

    color = colormap(spins, n)
    nx.draw_networkx(G_corr, node_size=10, node_color=color, with_labels=False, edge_cmap=mpl.colormaps['seismic'], edge_vmin=-1, edge_vmax=1, edge_color=list(w.values()), width=0.1)#np.exp(abs(np.array(list(w.values())))))
    plt.title('{} T={}'.format(lattice_type, T))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))
    
    plt.show()

if __name__ =="__main__":
    main()