import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from numba import jit

time_start = time.perf_counter()

lattice_type = 'ER'            #write square, triangular or hexagonal
M = 10
N = 10
J = 1
B = 0
steps = 30000
repeat = 100

Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1 

if lattice_type == "square":
    T = np.linspace(0.5*Tc, 1.5*Tc, 20) 
elif lattice_type == "hexagonal":
    T = np.linspace(0.5*Tc_h, 1.5*Tc_h, 20) 
    Tc = Tc_h
elif lattice_type == "triangular":
    T = np.linspace(0.5*Tc_t, 1.5*Tc_t, 20) 
    Tc = Tc_t
elif lattice_type == "ER":
    T = np.linspace(2, 4, 20) 
    Tc = 1
else: print("Errore!")

ones = np.ones(len(T))
beta = ones/T

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
    elif lattice_type == 'ER':
        lattice = nx.erdos_renyi_graph(M*N, 0.04, seed=None, directed=False)
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

@jit(nopython=True)
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

    G = lattice(M, N)

    n = num(G)

    #rand_spin = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node

    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    diameter_of_giant_component_T = np.empty(len(T))
    density_T = np.empty(len(T))
    btw_T = np.empty(len(T))
    cc_T = np.empty(len(T))
    for i in range(len(T)):

        cc_rep = np.empty(repeat)
        density_rep = np.empty(repeat)
        btw_rep = np.empty(repeat)
        diameter_of_giant_component_rep = np.empty(repeat)
        for rep in range(repeat):

            #spinlist = np.copy(rand_spin)
            spinlist = np.random.choice(np.asarray([-1, 1]), n)  #generate random spins for each node

            #iterate some steps
            A, s = step(A_dense, spinlist, beta[i], n)        
            A_clust = clustering(A, s)      

            G2 = nx.from_scipy_sparse_array(sparse.csr_matrix(A_clust)) #G2 only hasa the relevant edges
        
            density = nx.density(G2)
            density_rep[rep] = density

            cc = nx.number_connected_components(G2)
            cc_rep[rep] = cc

            #calculate the average betweennes centrality of nodes
            full_btw = nx.betweenness_centrality(G2)
            mean_btw = mean(full_btw.values())
            btw_rep[rep] = mean_btw

            Gcc = sorted(nx.connected_components(G2), key=len, reverse=True)
            giant = G2.subgraph(Gcc[0])
            #diameter_of_giant_component_rep[rep] = nx.radius(giant)
            spl = list(nx.all_pairs_shortest_path_length(giant))      #get shortest path lenght for each node in giant component (cluster)
            maxw = 0
            for s in range(len(spl)):                                 #find max distance
                spl_s = spl[s]
                w = list(spl_s[1].values())
                for e in range(len(w)):
                    if w[e] > maxw:
                        maxw = w[e]
            diameter_of_giant_component_rep[rep] = maxw               #store as diameter       

        density_T[i] = mean(density_rep)
        cc_T[i] = mean(cc_rep)
        btw_T[i] = mean(btw_rep)
        diameter_of_giant_component_T[i] = mean(diameter_of_giant_component_rep)

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
    fig.suptitle('{} no. atoms={},  B={}, J={}, ev_steps={}'.format(lattice_type, n, B, J, steps))
    fig.tight_layout()
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))
    
    plt.show()


if __name__ == "__main__":
    main()