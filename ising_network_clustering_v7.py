import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from numba import jit

time_start = time.perf_counter()

lattice_type = 'PT226'            #write square, triangular or hexagonal or PT{N}
M = 10
N = 10
J = 1
B = 0

steps = 20000

if lattice_type == "hexagonal":
    Tc = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
elif lattice_type == 'square':
    Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
elif lattice_type == "triangular": 
    Tc = 4 / np.sqrt(3)                       #Critical temperature of triangular lattice at J = 1 
elif lattice_type == "ER":
    Tc = 1
else: print("Errore!")

T = 5*Tc

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
    elif lattice_type == 'ER':
        lattice = nx.erdos_renyi_graph(M*N, 0.04, seed=None, directed=False)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = generate_grid_pos(lattice, M, N) #use for 2D grid network
    elif lattice_type == 'PT86':
        edges = np.loadtxt('PT/nnbond86.txt')
        adj = np.zeros((86, 86))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate86.txt')
        pos = []
        for node in range(86):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT226':
        edges = np.loadtxt('PT/nnbond226.txt')
        adj = np.zeros((226, 226))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate226.txt')
        pos = []
        for node in range(226):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT31':
        edges = np.loadtxt('PT/nnbond31.txt')
        adj = np.zeros((31, 31))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate31.txt')
        pos = []
        for node in range(31):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
    elif lattice_type == 'PT601':
        edges = np.loadtxt('PT/nnbond601.txt')
        adj = np.zeros((601, 601))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate601.txt')
        pos = []
        for node in range(601):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
        lattice = nx.from_numpy_array(adj)
    
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

    for l in range(steps):
    
        A = np.copy(A_dense)

        for m in range(len(spinlist)):
            for n in range(len(spinlist)):
                if A[m,n]==1:
                    A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

        #sum over rows to get total spin of neighbouring atoms for each atom
        nnsum = np.sum(A,axis=1)
    
        #What decides the flip is
        dE= 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist

        i = np.random.randint(num)
        if dE[i]<=0:
            spinlist[i] *= -1
        elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
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
    G, pos = lattice(M, N)

    n = num(G)

    spinlist = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node
    
    spinass(G, spinlist)

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    #iterate some steps
    A, s = step(A_dense, spinlist, 1/T, n)

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