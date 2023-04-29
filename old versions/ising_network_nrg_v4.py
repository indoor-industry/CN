import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

seed = 42
np.random.seed(seed)

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -0.2                        #coupling constant
B = 0.1                        #external field
M = 10                            #lattice size MxN
N = 10
steps = 25                      #number of steps per given temperature

T = np.linspace(0.1, 0.8, 30)   #temperature as of README

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#function assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#function to step lattice in time
def step(G, beta):
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    #create ordered list of spins
    spin = nx.get_node_attributes(G, 'spin')
    spinlist = np.asarray(list(dict.values(spin)))

    #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A = Adj.todense()
    for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
        for n in range(A.shape[1]):
            if A[m,n]==1:
                A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

    #sum over rows to get total spin of neighbouring atoms for each atom
    nnsum = np.sum(A,axis=1).tolist()

    #What decides the flip is
    dE = -4*J*np.multiply(nnsum, spinlist) + 2*B*spinlist    #change in energy

    E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy
    M = np.sum(spinlist)                                       #magnetization
    
    np.random.seed(np.random.randint(1000000000))      #this function needs a truly random seed
    for offset in range(2):
        for i in range(offset,len(dE),2):
            if dE[i]<=0:
                G.nodes[i]['spin'] *= -1
            elif np.exp(-dE[i]*beta) > np.random.rand():
                G.nodes[i]['spin'] *= -1
    np.random.seed(seed)            #reset the original seed

    return G, E, M

#step function for one beta and collect energy in time 
def iter(G, beta, j):
    i=0
    E_time = np.asarray([])
    M_time = np.asarray([])  
    while i <= steps:    
        G, E, M = step(G, beta[j])
        E_time = np.append(E_time, [E])
        M_time = np.append(M_time, [M])
        i+=1

    return E_time, M_time

#run iter for different betas and plot energy/node for each
def beta_sweep(G, beta):
    n=0
    for node in G:     #count number of nodes
        n+=1
    
    cv_beta = []
    xi_beta = []
    E_beta = []
    M_beta = []

    for j in range(len(beta)):
        spinass(G)   #re-randomize the network for every new beta

        E_time, M_time = iter(G, beta, j)
        
        cv_beta.append(np.var(E_time[steps//2:])*beta[j]**2)    #used to plot specific heat against temperature
        xi_beta.append(np.var(M_time[steps//2:])*beta[j])       #used to plot magnetic susceptibility against temperature
        E_beta.append(E_time[len(E_time)-1])        #used to plot energy against temperature
        M_beta.append(M_time[len(M_time)-1])        #used to plot magnetisation against temperature


        n_array = n*np.ones(len(E_time))
        plt.plot(E_time/n_array, label='β={:10.1f}'.format(beta[j]))    #plot energy per atom
        #plt.plot(M_time/n_array)
        print(j)


    plt.legend()
    plt.title('Energy against time')
    plt.legend(loc='upper right')
    plt.xlabel('timestep')
    plt.ylabel('energy per site [eV]')
    #plt.show()
        
    return E_beta, M_beta, cv_beta, xi_beta, n

def main():
    np.random.seed(seed)
    
    #create lattice
    G = lattice(M, N)
    #assign random spins to lattice
    spinass(G)
    #iterate steps and sweep trough beta
    E_beta, M_beta, cv_beta, xi_beta, n = beta_sweep(G, 1/T)

    n_normalize = n*np.ones(len(E_beta))    
    #plot Energy and magnetisation per site as a function of temperature
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.scatter(T, E_beta/n_normalize, color = 'orange')
    ax1.set_ylabel('$E(T)$')
    ax2.scatter(T, M_beta/n_normalize, color = 'blue')
    ax2.set_ylabel('$M(T)$')
    ax3.scatter(T, cv_beta/n_normalize, color = 'green')
    ax3.set_ylabel('$C_v(T)$')
    ax4.scatter(T, xi_beta/n_normalize, color = 'black')
    ax4.set_ylabel('$\Xi(T)$')
    fig.suptitle('{} {}x{}  B={} J={}'.format(lattice_type, M, N, B, J))
    fig.tight_layout()
    plt.show()

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))


if __name__ =="__main__":
    main()