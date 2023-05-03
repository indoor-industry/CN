import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

k_b = 8.617333262e-5
lattice_type = 'ER'  # write square, triangular or hexagonal, ER
J = 1  # spin coupling constant
B = 0  # external magnetic field
M = 10  # lattice size MxN
N = 10
steps = 15000  # number of evolution steps per given temperature
steps_to_eq = 10000  # steps until equilibrium is reached
repeat = 1  # number of trials per temperature to average over
nbstrap = 1000

lenth_p_sweep = np.linspace(1/(N*M), 4/(N*M), 4)  # 0.5 1 2.5 2

Tc = (2*abs(J))/np.log(1+np.sqrt(2))  # Critical temperature
# Critical temperature of hexagonal lattic  at J = 1
Tc_h = 2/np.log(2 + np.sqrt(3))
Tc_t = 4 / np.sqrt(3)  # Critical temperature of triangular lattice at J = 1


T = np.linspace(0.4*Tc, 2*Tc, 20)
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


@jit(nopython=True, nogil=True)
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
                dE = 2*J*np.multiply(nnsum, spinlist) + \
                    2*B*spinlist  # change in energy

                E = -J*sum(np.multiply(nnsum, spinlist)) - \
                    B*sum(spinlist)  # total energy
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

    for p in lenth_p_sweep:
        # create lattice
        G = lattice(M, N, p)

        # convert node labels to integers
        G = nx.convert_node_labels_to_integers(
            G, first_label=0, ordering='default', label_attribute=None)

        # get number of nodes
        n = num(G)

        # extract adjacency matrix and convert to numpy dense array
        Adj = nx.adjacency_matrix(
            G, nodelist=None, dtype=None, weight='weight')
        A_dense = Adj.todense()

        # iterate steps and sweep trough beta
        E_beta, M_beta, cv_beta, xi_beta, n = step(A_dense, beta, n)

        cv_array.append(cv_beta)
        xi_array.append(xi_beta)
        m_array.append(M_beta)

    print(len(cv_array))

    # for normalization purposes
    n_normalize = n*np.ones(len(E_beta))

    time_elapsed = (time.perf_counter() - time_start)
    print("checkpoint %5.1f secs" % (time_elapsed))

    # plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T, m_array[0]/n_normalize, color='orange',
                label='p={}/N*M'.format(round(lenth_p_sweep[0]*N*M)))
    ax1.scatter(T, m_array[1]/n_normalize, color='blue',
                label='p={}/N*M'.format(round(lenth_p_sweep[1]*N*M)))
    ax1.scatter(T, m_array[2]/n_normalize, color='green',
                label='p={}/N*M'.format(round(lenth_p_sweep[2]*N*M)))
    ax1.scatter(T, m_array[3]/n_normalize, color='black',
                label='p={}/N*M'.format(round(lenth_p_sweep[3]*N*M)))
    ax1.set_ylabel('$<\sqrt{|M^2|}>$')
    ax1.set_xlabel('T/Tc_square')
    fig.tight_layout()
    plt.show()

    # plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T/Tc, xi_array[0]/n_normalize, color='orange')
    ax1.set_ylabel('$\Xi$')
    ax1.set_xlabel('T/Tc_square')
    ax1.set_title('{}/N*M'.format(round(lenth_p_sweep[0]*N*M), 2))
    ax2.scatter(T/Tc, xi_array[1]/n_normalize, color='blue')
    ax2.set_ylabel('$\Xi$')
    ax2.set_xlabel('T/Tc_square')
    ax2.set_title('{}/N*M'.format(round(lenth_p_sweep[1]*N*M, 2)))
    ax3.scatter(T/Tc, xi_array[2]/n_normalize, color='green')
    ax3.set_ylabel('$\Xi$')
    ax3.set_xlabel('T/Tc_square')
    ax3.set_title('{}/N*M'.format(round(lenth_p_sweep[2]*N*M, 2)))
    ax4.scatter(T/Tc, xi_array[3]/n_normalize, color='black')
    ax4.set_ylabel('$\Xi$')
    ax4.set_xlabel('T/Tc_square')
    ax4.set_title('{}/N*M'.format(round(lenth_p_sweep[3]*N*M, 2)))
    fig.suptitle('{} no.atoms={}  B={} J={}, ev_steps={}, samples/T={}'.format(
        lattice_type, n, B, J, steps, repeat))
    fig.tight_layout()
    plt.show()

    # plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T, m_array[0]/n_normalize, color='orange')
    ax1.set_ylabel('$<\sqrt{|M^2|}>$')
    ax1.set_xlabel('T')
    ax1.set_title('p={}/N*M'.format(round(lenth_p_sweep[0]*N*M), 2))
    ax2.scatter(T, m_array[1]/n_normalize, color='blue')
    ax2.set_ylabel('$<\sqrt{|M^2|}>$')
    ax2.set_xlabel('T')
    ax2.set_title('p={}/N*M'.format(round(lenth_p_sweep[1]*N*M, 2)))
    ax3.scatter(T, m_array[2]/n_normalize, color='green')
    ax3.set_ylabel('$<\sqrt{|M^2|}>$')
    ax3.set_xlabel('T')
    ax3.set_title('p={}/N*M'.format(round(lenth_p_sweep[2]*N*M, 2)))
    ax4.scatter(T, m_array[3]/n_normalize, color='black')
    ax4.set_ylabel('$<\sqrt{|M^2|}>$')
    ax4.set_xlabel('T')
    ax4.set_title('p={}/N*M'.format(round(lenth_p_sweep[3]*N*M, 2)))
    fig.suptitle('{} no.atoms={}  B={} J={}, ev_steps={}, samples/T={}'.format(
        lattice_type, n, B, J, steps, repeat))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
