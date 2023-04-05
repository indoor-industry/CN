#ONLY FOR PERIODIC LATTICES

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from numba import jit
from scipy import optimize

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'square'            #write square, triangular or hexagonal
J = -1                            #spin coupling constant
sample = 10         

B_min = 0
B_max = 1
B = np.linspace(B_min, B_max, sample)                     #external magnetic field

M = 20                          #lattice size MxN
N = 20
steps = 100                      #number of evolution steps per given temperature

T_min = 0.2
T_max = 2
T = np.linspace(T_min, T_max, sample)

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
def step(A_dense, beta, num, B):

    den_beta_B = np.empty((sample, sample))
    for r in range(len(B)):
        for p in range(len(beta)):

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
                dE = -4*J*np.multiply(nnsum, spinlist) + 2*B[r]*spinlist    #change in energy     
                E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy    

                #change spins if energetically favourable or according to thermal noise
                for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
                    for i in range(offset,len(dE),2):
                        if dE[i]<0:
                            spinlist[i] *= -1
                        elif dE[i]==0:
                            continue
                        elif np.exp(-dE[i]*beta[p]) > np.random.rand():     #thermal noise
                            spinlist[i] *= -1


                for atom in range(num):
                    for neighbour in range(num):
                        corr_matrix[atom][neighbour]+=(spinlist[atom]*spinlist[neighbour]) 

            norm_corr_matrix = corr_matrix/steps

            di = []
            for a in range(num):
                den=0
                for q in range(num):
                    if q != a:
                        den += norm_corr_matrix[a][q]
                di.append(den/(num-1))
    
            density = sum(di)/num
            
            den_beta_B[r, p] = density

    return norm_corr_matrix, spinlist, den_beta_B

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

    #iterate steps and sweep trough beta
    corr_matrix, spins, den_matrix = step(A_dense, beta, n, B)

    ext = [T_min, T_max, B_min, B_max]
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(1, 1, 1)
    #ax2 = fig.add_subplot(2, 2, 2)
    
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(lattice_type, M, N, J, steps))
    
    im1 = ax1.imshow(den_matrix, cmap = 'coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('density')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T')
    
    #im2 = ax2.imshow(E_beta_J/n, cmap = 'Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    #ax2.set_title('E/site')
    #fig.colorbar(im2, ax=ax2)
    #ax2.set_ylabel('B')
    #ax2.set_xlabel('T')

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()

    #G_corr = nx.create_empty_copy(G, with_data=True)

    #for i in range(n):
    #    for j in range(n):
    #        if j<i:
    #            G_corr.add_edge(i, j, weight=corr_matrix[i][j])

    #w = nx.get_edge_attributes(G_corr, 'weight')

    #color = colormap(spins, n)
    #nx.draw_networkx(G_corr, node_size=10, node_color=color, with_labels=False, edge_cmap=mpl.colormaps['seismic'], edge_vmin=-1, edge_vmax=1, edge_color=list(w.values()), width=np.exp(abs(np.array(list(w.values())))))
    #plt.show()

if __name__ =="__main__":
    main()