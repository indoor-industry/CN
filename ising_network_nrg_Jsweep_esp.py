import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import numba as nb

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'square'             #write square, triangular or hexagonal
J = np.linspace(-0.5, 0.5, 50)      #spin coupling constant
B = 0.3                               #external magnetic field
M = 30                              #lattice size MxN
N = 30
steps = 200                         #number of evolution steps per given temperature

T = np.linspace(0.1, 2, 5)          #temperature range as of README
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
def step(A_dense, beta, num, J):

    rand_spins = np.random.choice(np.asarray([-1, 1]), num)   #create random spins for nodes
    
    cv_J = nb.typed.List.empty_list(nb.f8)
    xi_J = nb.typed.List.empty_list(nb.f8)
    E_J = nb.typed.List.empty_list(nb.f8)
    M_J = nb.typed.List.empty_list(nb.f8)

    for j in range(len(J)):                      #raster trough temperatures

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
            dE = -4*J[j]*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

            E = J[j]*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy
            M = np.sum(spinlist)                                       #total magnetisation

            #change spins if energetically favourable or according to thermal noise
            for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
                for i in range(offset,len(dE),2):
                    if dE[i]<0:
                        spinlist[i] *= -1
                    elif dE[i] == 0:
                        continue
                    elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
                        spinlist[i] *= -1

            E_time.append(E)            #list of energy trough time

            w.append(np.exp(-abs(E)*beta*k_b))
            
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

        cv_J.append(var_E*beta**2)    #used to plot specific heat against temperature
        xi_J.append(var_M*beta)       #used to plot magnetic susceptibility against temperature
        E_J.append(mean_E)               #used to plot energy against temperature
        M_J.append(mean_M)               #used to plot magnetisation against temperature

    return E_J, num, M_J, cv_J, xi_J

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


    #plot Energy and magnetisation per site as a function of J for various Ts
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)  
    ax1.set_ylabel('$<E(J)>$')
    ax2.set_ylabel('$<|M(J)|>$')
    ax3.set_ylabel('$C_v(J)$')
    ax4.set_ylabel('$\Xi(J)$')
    fig.suptitle('{} {}x{}  B={}, ev_steps={}'.format(lattice_type, M, N, B, steps))
    fig.tight_layout()
    
    for a in range(len(T)):

        E_J, n, M_J, cv_J, xi_J = step(A_dense, beta[a], n, J)
        n_normalize = n*np.ones(len(E_J)) 

        ax1.plot(J, E_J/n_normalize, label='T={:10.2f}'.format(T[a]))
        ax2.plot(J, M_J/n_normalize)
        ax3.plot(J, cv_J/n_normalize)
        ax4.plot(J, xi_J/n_normalize)

        print('{}/{}'.format(a+1, len(T)))

    fig.legend()
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))
    
    plt.show()


if __name__ =="__main__":
    main()