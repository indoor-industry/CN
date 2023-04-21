import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'square'            #write square, triangular or hexagonal, ER
J = 1                       #spin coupling constant
B = 0                       #external magnetic field
M = 10                          #lattice size MxN
N = 10
steps = 30000                      #number of evolution steps per given temperature
steps_to_eq = 20000                   #steps until equilibrium is reached
repeat = 1                     #number of trials per temperature to average over

Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice
print(Tc)

T = np.linspace(1, 10, 30)   #temperature range

ones = np.ones(len(T))
beta = ones/(T)

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    elif lattice_type == 'ER':
        lattice = nx.erdos_renyi_graph(M*N, 0.01, seed=None, directed=False)
    return lattice

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

@jit(nopython=True, nogil=True)
def step(A_dense, beta, num):

    def variance(data):             #variance function needed for specific heat and magnetic susceptibility
        # Number of observations
        size = len(data)
        # Mean of the data
        mean = sum(data) / size
        # Square deviations
        deviations = [(x - mean) ** 2 for x in data]
        # Variance
        variance = sum(deviations) / size
        return variance

    def mean(list):
        return sum(list)/len(list)
    
    rand_spin = np.random.choice(np.asarray([-1, 1]), num)   #create random spins for nodes

    cv_beta = np.empty(len(beta))
    xi_beta = np.empty(len(beta))
    E_beta = np.empty(len(beta))
    M_beta = np.empty(len(beta))

    for j in range(len(beta)):              #raster trough temperatures

        spinlist = np.copy(rand_spin)   #create random spins for nodes

        l=0
        E_time = np.empty(steps)
        M_time = np.empty(steps)
        for h in range(steps):                               #evolve trough steps number of timesteps

            A = np.copy(A_dense)                        #take new copy of adj. matrix at each step because it gets changed trough the function

            for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
                for n in range(A.shape[1]):
                    if A[m,n]==1:
                        A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

            #sum over rows to get total spin of neighbouring atoms for each atom
            nnsum = np.sum(A,axis=1)

            #What decides the flip is
            dE = 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

            E = -J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy
            M = np.sum(spinlist)                                       #total magnetisation

            #change spins if energetically favourable or according to thermal noise
                
            i = np.random.randint(num)

            if dE[i]<=0:
                spinlist[i] *= -1
            elif np.exp(-dE[i]*beta[j]) > np.random.rand():     #thermal noise
                spinlist[i] *= -1

            E_time[h] = E            #list of energy trough time
            M_time[h] = M            #list of magnetisation trough time
            l+=1

        var_E = variance(E_time[steps_to_eq:])     #variance of energy (start aquiring after equilibrium is reached)
        var_M = variance(M_time[steps_to_eq:])     #same as above for magnetisation
        mean_E = mean(E_time[steps_to_eq:])
        mean_M = mean(M_time[steps_to_eq:])

        cv_beta[j] = var_E*beta[j]**2    #used to plot specific heat against temperature
        xi_beta[j] = var_M*beta[j]       #used to plot magnetic susceptibility against temperature
        E_beta[j] = mean_E               #used to plot energy against temperature
        M_beta[j] = abs(mean_M)              #used to plot magnetisation against temperature

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
    
    #save the data
    np.savetxt(f"data/E_{M}x{N}_{lattice_type}_B={B}.csv", E_beta, delimiter=",")
    np.savetxt(f"data/M_{M}x{N}_{lattice_type}_B={B}.csv", M_beta, delimiter=",")
    np.savetxt(f"data/cv_{M}x{N}_{lattice_type}_B={B}.csv", cv_beta, delimiter=",")
    np.savetxt(f"data/xi_{M}x{N}_{lattice_type}_B={B}.csv", xi_beta, delimiter=",")
    np.savetxt(f"data/T_{M}x{N}_{lattice_type}_B={B}.csv", T, delimiter=",")

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
    ax1.scatter(T/Tc, E_beta/(n_normalize*abs(J)), color = 'orange')
    ax1.set_ylabel('$<E>/J$')
    ax1.set_xlabel('T/Tc')
    ax2.scatter(T/Tc, M_beta/n_normalize, color = 'blue')
    ax2.set_ylabel('$<|M|>$')
    ax2.set_xlabel('T/Tc')
    ax3.scatter(T/Tc, cv_beta/n_normalize, color = 'green')
    ax3.set_ylabel('$C_v$')
    ax3.set_xlabel('T/Tc')
    ax4.scatter(T/Tc, xi_beta/n_normalize, color = 'black')
    ax4.set_ylabel('$\Xi$')
    ax4.set_xlabel('T/Tc')
    fig.suptitle('{} no.atoms={}  B={} J={}, ev_steps={}, samples/T={}'.format(lattice_type, n, B, J, steps, repeat))
    fig.tight_layout()
    plt.show()

if __name__ =="__main__":
    main()