import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'PT86'            #write square, triangular or hexagonal, ER
J = 1                       #spin coupling constant
B = 0                       #external magnetic field
M = 5                      #lattice size MxN
N = 5
steps = 30000                      #number of evolution steps per given temperature
steps_to_eq = 20000                   #steps until equilibrium is reached
repeat = 1000                     #number of trials per temperature to average over
nbstrap = 1000

Tc = (2*abs(J))/np.log(1+np.sqrt(2))        #Critical temperature
Tc_h = 2/np.log(2 + np.sqrt(3))             #Critical temperature of hexagonal lattic  at J = 1
Tc_t = 4 / np.log(3)                       #Critical temperature of triangular lattice at J = 1 

if lattice_type == "square":
    T = np.linspace(0.5*Tc, 1.5*Tc, 30) 
elif lattice_type == "hexagonal":
    T = np.linspace(0.5*Tc_h, 1.5*Tc_h, 30) 
    Tc = Tc_h
elif lattice_type == "triangular":
    T = np.linspace(0.5*Tc_t, 1.5*Tc_t, 30) 
    Tc = Tc_t
elif lattice_type == 'ER':
    T = np.linspace(2, 4, 30)
    Tc = 1 
elif lattice_type == 'PT86' or lattice_type == 'PT31' or lattice_type == 'PT226' or lattice_type == 'PT601':
    T = np.linspace(1, 3, 30)
    Tc = 1

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
        lattice = nx.erdos_renyi_graph(M*N, 4/(M*N), seed=None, directed=False)
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

@jit(nopython=True, nogil=True)
def step(A_dense, beta, num):

    def mean_square(data):
        sum_of_squares=0
        for element in range(len(data)):
            sum_of_squares += data[element]**2
        return np.sqrt(sum_of_squares/len(data))

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

        rep_var_E = np.empty(repeat)
        rep_var_M = np.empty(repeat)
        rep_mean_E = np.empty(repeat)
        rep_mean_M = np.empty(repeat)

        for t in range(repeat):             #repeat and average over runs

            spinlist = np.copy(rand_spin)   #create random spins for nodes

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

            def bootstrap(G):
                G_bootstrap = []
                for i in range(steps-steps_to_eq):
                    alpha = int(np.random.uniform(0, steps-steps_to_eq))
                    G_bootstrap.append(G[alpha])
                return G_bootstrap

            def bs_mean(G):                                             # MC avg of G
                G_bs_mean = np.empty(steps-steps_to_eq)        
                for n in range(steps-steps_to_eq):                                  # compute MC averages
                    avg_G = 0
                    for alpha in range(len(G)):
                        avg_G += G[alpha][n]
                    avg_G = avg_G/len(G)
                    G_bs_mean[n] = avg_G
                return G_bs_mean

            bsM_time = np.empty((nbstrap, steps-steps_to_eq))
            bsE_time = np.empty((nbstrap, steps-steps_to_eq))
            for p in range(nbstrap):
                g = bootstrap(E_time[steps_to_eq:])
                f = bootstrap(M_time[steps_to_eq:])
                bsE_time[p] = g
                bsM_time[p] = f
            bsE_time_avg = bs_mean(bsE_time)
            bsM_time_avg = bs_mean(bsM_time)

            var_E = variance(bsE_time_avg)     #variance of energy (start aquiring after equilibrium is reached)
            var_M = variance(bsM_time_avg)     #same as above for magnetisation
            mean_E = mean(bsE_time_avg)
            mean_M = mean_square(bsM_time_avg)

            rep_mean_E[t] = mean_E                   #done 'repeat' number of times
            rep_mean_M[t] = mean_M
            rep_var_E[t] = var_E
            rep_var_M[t] = var_M

        avg_mean_E = mean(rep_mean_E)                   #average over repeats
        avg_mean_M = mean(rep_mean_M)
        avg_var_E = mean(rep_var_E)
        avg_var_M = mean(rep_var_M)

        cv_beta[j] = avg_var_E*beta[j]**2    #used to plot specific heat against temperature
        xi_beta[j] = avg_var_M*beta[j]       #used to plot magnetic susceptibility against temperature
        E_beta[j] = avg_mean_E               #used to plot energy against temperature
        M_beta[j] = abs(avg_mean_M)              #used to plot magnetisation against temperature

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
    ax2.set_ylabel('$<\sqrt{|M^2|}>$')
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