import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import numba as nb

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -0.2                       #spin coupling constant
B = 0                       #external magnetic field
M = 30                          #lattice size MxN
N = 30
steps = 200                      #number of evolution steps per given temperature

T = np.linspace(0.1, 2, 50)   #temperature range as of README

ones = np.ones(len(T))
beta = ones/T

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

@jit(nopython=True)
def step(A_dense, beta, num):

    rand_spins = np.random.choice(np.asarray([-1, 1]), num)   #create random spins for nodes
    
    cv_beta = nb.typed.List.empty_list(nb.f8)
    xi_beta = nb.typed.List.empty_list(nb.f8)
    E_beta = nb.typed.List.empty_list(nb.f8)
    M_beta = nb.typed.List.empty_list(nb.f8)

    for j in range(len(beta)):                      #raster trough temperatures

        spinlist = np.copy(rand_spins)

        l=0
        E_time = nb.typed.List.empty_list(nb.f8)
        M_time = nb.typed.List.empty_list(nb.f8)

        w = nb.typed.List.empty_list(nb.f8)
        while l <= steps:                               #evolve trough steps number of timesteps

            A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

            for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                for n in range(A.shape[1]):
                    if A[m,n]==1:
                        A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

            #sum over rows to get total spin of neighbouring atoms for each atom
            nnsum = np.sum(A,axis=1)

            #What decides the flip is
            dE = -4*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

            E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy
            M = np.sum(spinlist)                                       #total magnetisation

            #change spins if energetically favourable or according to thermal noise
            for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
                for i in range(offset,len(dE),2):
                    if dE[i]<0:
                        spinlist[i] *= -1
                    elif dE[i] == 0:
                        continue
                    elif np.exp(-dE[i]*beta[j]) > np.random.rand():     #thermal noise
                        spinlist[i] *= -1

            E_time.append(E)            #list of energy trough time

            w.append(np.exp(-E*beta[j]))

            M_time.append(abs(M))            #list of magnetisation trough time
            l+=1
        
        def weighted_variance(data, weights):             #variance function needed for specific heat and magnetic susceptibility
            mean = 0
            mean2 = 0
            for e in range(len(data)):
                mean += weights[e]*data[e]
                mean2 += weights[e]*(data[e]**2)
            norm_mean = mean/sum(weights)
            norm_mean2 = mean2/sum(weights)
            variance = norm_mean2 - norm_mean**2
            return variance

        def weighted_mean(data, weights):
            mean = 0
            for i in range(len(data)):
                mean += weights[i]*data[i]
            norm_mean = mean/sum(weights)
            return norm_mean

        var_E = weighted_variance(E_time[steps//2:], w[steps//2:])     #variance of energy (start acquiring half trough evolution to let system reach equilibrium)
        var_M = weighted_variance(M_time[steps//2:], w[steps//2:])     #same as above for magnetisation
        mean_E = weighted_mean(E_time[steps//2:], w[steps//2:])
        mean_M = weighted_mean(M_time[steps//2:], w[steps//2:])

        cv_beta.append(var_E*beta[j]**2)    #used to plot specific heat against temperature
        xi_beta.append(var_M*beta[j])       #used to plot magnetic susceptibility against temperature
        E_beta.append(mean_E)               #used to plot energy against temperature
        M_beta.append(mean_M)               #used to plot magnetisation against temperature

        print(j)

    return E_beta, M_beta, cv_beta, xi_beta, num

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
    E_beta, M_beta, cv_beta, xi_beta, n = step(A_dense, beta, n)

    #for normalization purposes
    n_normalize = n*np.ones(len(E_beta)) 

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    #plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T, E_beta/n_normalize, color = 'orange')
    ax1.set_ylabel('$<E(T)>$')
    ax2.scatter(T, M_beta/n_normalize, color = 'blue')
    ax2.set_ylabel('$<|M(T)|>$')
    ax3.scatter(T, cv_beta/n_normalize, color = 'green')
    ax3.set_ylabel('$C_v(T)$')
    ax4.scatter(T, xi_beta/n_normalize, color = 'black')
    ax4.set_ylabel('$\Xi(T)$')
    fig.suptitle('{} {}x{}  B={} J={}, ev_steps={}'.format(lattice_type, M, N, B, J, steps))
    fig.tight_layout()
    plt.show()

if __name__ =="__main__":
    main()