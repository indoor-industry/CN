import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import scipy as sp

time_start = time.perf_counter()

J = 1  # spin coupling constant
B = 0  # external magnetic field
M = 5  # lattice size MxN
N = 5
steps = 21000  # number of evolution steps per given temperature
steps_to_eq = 20000  # steps until equilibrium is reached
repeat = 2  # number of trials per temperature to average over
nbstrap = 1000

#original parameters
#J = 1  # spin coupling constant
#B = 0  # external magnetic field
#M = 10  # lattice size MxN
#N = 10
#steps = 30000  # number of evolution steps per given temperature
#steps_to_eq = 20000  # steps until equilibrium is reached
#repeat = 10  # number of trials per temperature to average over
#nbstrap = 1000

lenth_p_sweep = np.arange(1/(N*M), 10/(N*M), 1/(M*N))
print(lenth_p_sweep)

T = np.linspace(0.5, 8, 30)
ones = np.ones(len(T))
beta = ones/(T)

# function creates lattice
def lattice(M, N, p):
    lattice = nx.erdos_renyi_graph(M*N, p, seed=None, directed=False)
    return lattice

# count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

@jit(nopython=True)
def step(A_dense, beta, num):

    def mean_square(data):
        sum_of_squares = 0
        for element in range(len(data)):
            sum_of_squares += data[element]**2
        return np.sqrt(sum_of_squares/len(data))

    def variance(data):  # variance function needed for specific heat and magnetic susceptibility
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

    # create random spins for nodes
    rand_spin = np.random.choice(np.asarray([-1, 1]), num)

    cv_beta = np.empty(len(beta))
    xi_beta = np.empty(len(beta))
    E_beta = np.empty(len(beta))
    M_beta = np.empty(len(beta))

    for j in range(len(beta)):  # raster trough temperatures

        rep_var_E = np.empty(repeat)
        rep_var_M = np.empty(repeat)
        rep_mean_E = np.empty(repeat)
        rep_mean_M = np.empty(repeat)

        for t in range(repeat):  # repeat and average over runs

            spinlist = np.copy(rand_spin)  # create random spins for nodes

            l = 0
            E_time = np.empty(steps)
            M_time = np.empty(steps)
            for h in range(10*steps):  # evolve trough steps number of timesteps

                # take new copy of adj. matrix at each step because it gets changed trough the function
                A = np.copy(A_dense)

                for m in range(A.shape[1]):  # A.shape[1] gives number of nodes
                    for n in range(A.shape[1]):
                        if A[m, n] == 1:
                            # assigned to every element in the adj matrix the corresponding node spin value
                            A[m, n] = spinlist[n]

                # sum over rows to get total spin of neighbouring atoms for each atom
                nnsum = np.sum(A, axis=1)

                # What decides the flip is
                dE = 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist  # change in energy

                E = -J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)  # total energy
                M = np.sum(spinlist)  # total magnetisation

                # change spins if energetically favourable or according to thermal noise
                i = np.random.randint(num)
                if dE[i] <= 0:
                    spinlist[i] *= -1
                elif np.exp(-dE[i]*beta[j]) > np.random.rand():  # thermal noise
                    spinlist[i] *= -1

                if h % 10 == 0:  # acquire every 10 steps to reduce correlations between aquisitions
                    E_time[l] = E  # list of energy trough time
                    M_time[l] = M  # list of magnetisation trough time
                    l += 1

            def bootstrap(G):
                G_bootstrap = []
                for i in range(steps-steps_to_eq):
                    alpha = int(np.random.uniform(0, steps-steps_to_eq))
                    G_bootstrap.append(G[alpha])
                return G_bootstrap

            def bs_mean(G):                                             # MC avg of G
                G_bs_mean = np.empty(steps-steps_to_eq)
                # compute MC averages
                for n in range(steps-steps_to_eq):
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

            # variance of energy (start aquiring after equilibrium is reached)
            var_E = variance(bsE_time_avg)
            var_M = variance(bsM_time_avg)  # same as above for magnetisation
            mean_E = mean(bsE_time_avg)
            mean_M = mean_square(bsM_time_avg)

            rep_mean_E[t] = mean_E  # done 'repeat' number of times
            rep_mean_M[t] = mean_M
            rep_var_E[t] = var_E
            rep_var_M[t] = var_M

        avg_mean_E = mean(rep_mean_E)  # average over repeats
        avg_mean_M = mean(rep_mean_M)
        avg_var_E = mean(rep_var_E)
        avg_var_M = mean(rep_var_M)

        # used to plot specific heat against temperature
        cv_beta[j] = avg_var_E*beta[j]**2
        # used to plot magnetic susceptibility against temperature
        xi_beta[j] = avg_var_M*beta[j]
        E_beta[j] = avg_mean_E  # used to plot energy against temperature
        # used to plot magnetisation against temperature
        M_beta[j] = abs(avg_mean_M)

        print(j)

    return E_beta, M_beta, cv_beta, xi_beta, num

def main():
    cv_array = []
    xi_array = []
    m_array = []
    E_array = []
    for p in lenth_p_sweep:
        # create lattice
        G = lattice(M, N, p)

        # convert node labels to integers
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

        # get number of nodes
        n = num(G)

        # extract adjacency matrix and convert to numpy dense array
        Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
        A_dense = Adj.todense()

        # iterate steps and sweep trough beta
        E_beta, M_beta, cv_beta, xi_beta, n = step(A_dense, beta, n)

        E_array.append(E_beta)
        cv_array.append(cv_beta)
        xi_array.append(xi_beta)
        m_array.append(M_beta)

    critical = []
    for a in range(len(lenth_p_sweep)):
        critical_index = cv_array[a].argmax(axis=0)
        critical_temp = T[critical_index]
        critical.append(critical_temp)

    print(critical)
    np.savetxt("psweep_Tc.csv", critical, delimiter=",")
    np.savetxt("psweep_p.csv", lenth_p_sweep, delimiter=",")

    plt.scatter(lenth_p_sweep, critical)
    plt.title('ER Critical temperature')
    plt.xlabel('p')
    plt.ylabel('Tc')
    plt.show()

    # for normalization purposes
    n_normalize = n*np.ones(len(E_beta))

    time_elapsed = (time.perf_counter() - time_start)
    print("checkpoint %5.1f secs" % (time_elapsed))

    #plot magnetisation for different p's
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.plot(T, m_array[0]/n_normalize, color='orange', label='p={}/N*M'.format(round(lenth_p_sweep[0]*N*M)))
    ax1.plot(T, m_array[1]/n_normalize, color='blue', label='p={}/N*M'.format(round(lenth_p_sweep[1]*N*M)))
    ax1.plot(T, m_array[2]/n_normalize, color='green', label='p={}/N*M'.format(round(lenth_p_sweep[2]*N*M)))
    ax1.plot(T, m_array[3]/n_normalize, color='black', label='p={}/N*M'.format(round(lenth_p_sweep[3]*N*M)))
    ax1.plot(T, m_array[4]/n_normalize, color='purple', label='p={}/N*M'.format(round(lenth_p_sweep[4]*N*M)))
    ax1.plot(T, m_array[5]/n_normalize, color='yellow', label='p={}/N*M'.format(round(lenth_p_sweep[5]*N*M)))
    ax1.set_ylabel('$<\sqrt{|M^2|}>$')
    ax1.set_xlabel('T')

    ax2.plot(T, E_array[0]/n_normalize, color='orange', label='p={}/N*M'.format(round(lenth_p_sweep[0]*N*M)))
    ax2.plot(T, E_array[1]/n_normalize, color='blue', label='p={}/N*M'.format(round(lenth_p_sweep[1]*N*M)))
    ax2.plot(T, E_array[2]/n_normalize, color='green', label='p={}/N*M'.format(round(lenth_p_sweep[2]*N*M)))
    ax2.plot(T, E_array[3]/n_normalize, color='black', label='p={}/N*M'.format(round(lenth_p_sweep[3]*N*M)))
    ax2.plot(T, E_array[4]/n_normalize, color='purple', label='p={}/N*M'.format(round(lenth_p_sweep[4]*N*M)))
    ax2.plot(T, E_array[5]/n_normalize, color='yellow', label='p={}/N*M'.format(round(lenth_p_sweep[5]*N*M)))
    ax2.set_ylabel('$E/node$')
    ax2.set_xlabel('T')

    fig.tight_layout()
    plt.legend()
    plt.show()

    # plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.scatter(T, xi_array[0]/n_normalize, color='orange')
    ax1.set_ylabel('$\Xi$')
    ax1.set_xlabel('T')
    ax1.set_title('{}/N*M'.format(round(lenth_p_sweep[0]*N*M), 2))
    ax2.scatter(T, xi_array[1]/n_normalize, color='blue')
    ax2.set_ylabel('$\Xi$')
    ax2.set_xlabel('T')
    ax2.set_title('{}/N*M'.format(round(lenth_p_sweep[1]*N*M, 2)))
    ax3.scatter(T, xi_array[2]/n_normalize, color='green')
    ax3.set_ylabel('$\Xi$')
    ax3.set_xlabel('T')
    ax3.set_title('{}/N*M'.format(round(lenth_p_sweep[2]*N*M, 2)))
    ax4.scatter(T, xi_array[3]/n_normalize, color='black')
    ax4.set_ylabel('$\Xi$')
    ax4.set_xlabel('T')
    ax4.set_title('{}/N*M'.format(round(lenth_p_sweep[3]*N*M, 2)))
    ax5.scatter(T, xi_array[4]/n_normalize, color='purple')
    ax5.set_ylabel('$\Xi$')
    ax5.set_xlabel('T')
    ax5.set_title('{}/N*M'.format(round(lenth_p_sweep[4]*N*M, 2)))
    ax6.scatter(T, xi_array[5]/n_normalize, color='yellow')
    ax6.set_ylabel('$\Xi$')
    ax6.set_xlabel('T')
    ax6.set_title('{}/N*M'.format(round(lenth_p_sweep[5]*N*M, 2)))
    fig.suptitle('ER no.atoms={}  B={} J={}, ev_steps={}, samples/T={}'.format(n, B, J, steps, repeat))
    fig.tight_layout()
    plt.show()


    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.scatter(T, cv_array[0]/n_normalize, color='orange')
    ax1.set_ylabel('$C_v$')
    ax1.set_xlabel('T')
    ax1.set_title('{}/N*M'.format(round(lenth_p_sweep[0]*N*M), 2))
    ax2.scatter(T, cv_array[1]/n_normalize, color='blue')
    ax2.set_ylabel('$C_v$')
    ax2.set_xlabel('T')
    ax2.set_title('{}/N*M'.format(round(lenth_p_sweep[1]*N*M, 2)))
    ax3.scatter(T, cv_array[2]/n_normalize, color='green')
    ax3.set_ylabel('$C_v$')
    ax3.set_xlabel('T')
    ax3.set_title('{}/N*M'.format(round(lenth_p_sweep[2]*N*M, 2)))
    ax4.scatter(T, cv_array[3]/n_normalize, color='black')
    ax4.set_ylabel('$C_v$')
    ax4.set_xlabel('T')
    ax4.set_title('{}/N*M'.format(round(lenth_p_sweep[3]*N*M, 2)))
    ax5.scatter(T, cv_array[4]/n_normalize, color='purple')
    ax5.set_ylabel('$C_v$')
    ax5.set_xlabel('T')
    ax5.set_title('{}/N*M'.format(round(lenth_p_sweep[4]*N*M, 2)))
    ax6.scatter(T, cv_array[5]/n_normalize, color='yellow')
    ax6.set_ylabel('$C_v$')
    ax6.set_xlabel('T')
    ax6.set_title('{}/N*M'.format(round(lenth_p_sweep[5]*N*M, 2)))
    fig.suptitle('ER no.atoms={}  B={} J={}, ev_steps={}, samples/T={}'.format(n, B, J, steps, repeat))
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
