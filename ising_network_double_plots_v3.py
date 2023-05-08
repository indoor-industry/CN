import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

time_start = time.perf_counter()

lattice_type = 'square'     # write square, triangular or hexagonal
J = 1                       # spin coupling
M = 10                      # lattice size MxN
N = 10
steps = 30000       # number of timesteps of evolution per given temperature
B_sample = 20       # number of samples between minimum and maximum values of B NEEDS TO BE ODD FOR SENSIBLE RESULTS
T_sample = 20       # number of samples between minimum and maximum values of T
steps_to_eq = 25000
nbstrap = 1000

if lattice_type == 'square':
    Tc = (2*abs(J))/np.log(1+np.sqrt(2))  # Critical temperature
    T_min = 0.5*Tc  # min temperature to explore
    T_max = 1.5*Tc  # max temperature to explore
elif lattice_type == 'triangular:':
    Tc = 2/np.log(2 + np.sqrt(3))     # Critical temperature of hexagonal lattic  at J = 1
    T_min = 0.5*Tc  # min temperature to explore
    T_max = 1.5*Tc  # max temperature to explore
elif lattice_type == 'hexagonal':
    Tc = 4 / np.log(3)  # Critical temperature of triangular lattice at J = 1
    T_min = 0.5*Tc  # min temperature to explore
    T_max = 1.5*Tc  # max temperature to explore

B_min = 0  # min magnetic field to explore
B_max = 1.5  # max magnetic field to explore

T = np.linspace(T_min, T_max, T_sample)  # temperature range to explore

ones = np.ones(len(T))  # convert to inverse temperature
beta = ones/T

# External magnetic field range to explore
B = np.linspace(B_min, B_max, B_sample)

# function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(
            M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(
            M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

# function that counts numer of nodes
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

# function to step lattice in time
@jit(nopython=True)
def step(A_dense, beta, B, num):

    def mean(list):
        return sum(list)/len(list)

    def mean_square(data):
        sum_of_squares = 0
        for element in range(len(data)):
            sum_of_squares += data[element]**2
        return np.sqrt(sum_of_squares/len(data))

    M_beta_J = np.empty((B_sample, T_sample))
    E_beta_J = np.empty((B_sample, T_sample))

    # generate random spins for each node
    rand_spin = np.random.choice(np.asarray([-1, 1]), num)

    for i in range(len(B)):
        for j in range(len(beta)):  # run through different combinations of B and T

            spinlist = np.copy(rand_spin)
            # spinlist = np.random.choice(np.asarray([-1, 1]), num)  #generate random spins for each node

            M_time = []
            E_time = []
            for k in range(steps):  # evolve the system trough steps number of timesteps
                # create a new copy of the adjacency matrix at every step otherwise it will be distorted by the rest of the function
                A = np.copy(A_dense)

                # create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
                for m in range(A.shape[1]):  # A.shape[1] gives number of nodes
                    for n in range(A.shape[1]):
                        if A[m, n] == 1:
                            # assigned to every element in the adj matrix the corresponding node spin value
                            A[m, n] = spinlist[n]

                # sum over rows to get total spin of neighbouring atoms for each atom
                nnsum = np.sum(A, axis=1)

                # What decides the flip is
                dE = 2*J*np.multiply(nnsum, spinlist) + 2*B[i]*spinlist  # change in energy
                E = -J*sum(np.multiply(nnsum, spinlist)) - B[i]*sum(spinlist)  # total energy
                M = np.sum(spinlist)  # total magnetisation

                # update spin configuration if energetically favourable or if thermal fluctuations contribute
                l = np.random.randint(num)
                if dE[l] <= 0:
                    spinlist[l] *= -1
                elif np.exp(-dE[l]*beta[j]) > np.random.rand():  # thermal noise
                    spinlist[l] *= -1

                E_time.append(E)
                M_time.append(abs(M))

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

            # store magnetisation values
            M_beta_J[i, j] = mean(bsM_time_avg)
            E_beta_J[i, j] = mean(bsE_time_avg)/abs(J)

        print(i)

    return E_beta_J, M_beta_J


def main():
    # create lattice
    G = lattice(M, N)
    # label nodes as integers
    G = nx.convert_node_labels_to_integers(
        G, first_label=0, ordering='default', label_attribute=None)
    # get number of nodes
    n = num(G)
    # extract adjacency matrix from network of spins ans convert it to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()
    # run program
    E_beta_J, M_beta_J = step(A_dense, beta, B, n)

    # store values as csv's
    # np.savetxt("E.csv", E_beta_J/n, delimiter=",")
    # np.savetxt("M.csv", M_beta_J/n, delimiter=",")

    # plot
    ext = [T_min/Tc, T_max/Tc, B_min, B_max]

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    fig.suptitle('{}, size {}x{}, J={}, ev_steps={}'.format(
        lattice_type, M, N, J, steps))

    im1 = ax1.imshow(M_beta_J/n, cmap='coolwarm', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax1.set_title('M/site')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B')
    ax1.set_xlabel('T/Tc')

    im2 = ax2.imshow(E_beta_J/n, cmap='Reds', origin='lower', extent=ext, aspect='auto', interpolation='spline36')
    ax2.set_title('(E/J)/site')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B')
    ax2.set_xlabel('T/Tc')

    time_elapsed = (time.perf_counter() - time_start)
    print("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()


if __name__ == "__main__":
    main()
