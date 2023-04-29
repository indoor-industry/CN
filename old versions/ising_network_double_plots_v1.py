import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

time_start = time.perf_counter()

lattice_type = 'square'          #write square, triangular or hexagonal
J = -0.2           
M = 10                           #lattice size MxN
N = 10
steps = 20                      #number of steps per given temperature
sample = 20

T_min_mag = 0.1
T_max_mag = 1.5

T_min_nrg = 0.1
T_max_nrg = 1.5

B_over_J_min = -0.5
B_over_J_max = 0.5

T_mag = np.linspace(T_min_mag, T_max_mag, sample)   #temperature to explore magnetisation 
T_nrg = np.linspace(T_min_nrg, T_max_nrg, sample)   #temperature to explore energy
B_over_J = np.linspace(B_over_J_min, B_over_J_max, sample)

#function creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
    return lattice

#function that counts numer of nodes
def num(G):
    n=0
    for node in G:     #count number of nodes
        n+=1
    return n

#function assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#function to step lattice in time
def step(G, beta, B_over_J):
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
    dE_over_J = -4*J*np.multiply(nnsum, spinlist) + 2*B_over_J*spinlist    #change in energy

    E_over_J = J*sum(np.multiply(nnsum, spinlist)) - B_over_J*sum(spinlist)   #total energy
    M = np.sum(spinlist) 

    for offset in range(2):
        for i in range(offset,len(dE_over_J),2):
            if dE_over_J[i]<=0:
                G.nodes[i]['spin'] *= -1
            elif np.exp(-dE_over_J[i]*beta) > np.random.rand():
                G.nodes[i]['spin'] *= -1   
    
    return G, E_over_J, M

#step function for one beta and collect energy in time 

def iter(G, beta, B_over_J, j, i):
    k=0
    while k <= steps:    
        G, E_over_J, M = step(G, beta[j], B_over_J[i])
        #print(k)
        k+=1
     
    return E_over_J, M

#run iter for different betas and plot energy/node for each
def mag_sweep(G, beta, B_over_J):
    M_beta_J = np.empty((sample, sample))

    for i in range(len(B_over_J)):
        for j in range(len(beta)):
            spinass(G)   #re-randomize the network for every new beta
            
            E_over_J, M = iter(G, beta, B_over_J, j, i)
            M_beta_J[i, j] = M

    return M_beta_J

def nrg_sweep(G, beta, B_over_J):
    E_beta_J = np.empty((sample, sample))

    for i in range(len(B_over_J)):
        for j in range(len(beta)):
            spinass(G)   #re-randomize the network for every new beta
            
            E_over_J, M = iter(G, beta, B_over_J, j, i)
            E_beta_J[i, j] = E_over_J       #used to plot energy against temperature

    return E_beta_J

def main():
    #create lattice
    G = lattice(M, N)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    #assign random spins to lattice
    spinass(G)

    #get number of nodes
    n = num(G)

    #iterate steps and sweep trough beta
    M_beta_J = mag_sweep(G, 1/T_mag, B_over_J)
    E_beta_J = nrg_sweep(G, 1/T_nrg, B_over_J)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 1 %5.1f secs" % (time_elapsed))

    np.savetxt("E.csv", E_beta_J/n, delimiter=",")
    np.savetxt("M.csv", M_beta_J/n, delimiter=",")

    #plot
    ext_mag = [T_min_mag, T_max_mag, B_over_J_min, B_over_J_max]
    ext_nrg = [T_min_nrg, T_max_nrg, B_over_J_min, B_over_J_max]
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    fig.suptitle('{}, size {}x{}'.format(lattice_type, M, N))
    
    im1 = ax1.imshow(M_beta_J/n, cmap = 'coolwarm', origin='lower', extent=ext_mag, aspect='auto', interpolation='spline36')
    ax1.set_title('M/site')
    fig.colorbar(im1, ax=ax1)
    ax1.set_ylabel('B/J')
    ax1.set_xlabel('T')
    
    im2 = ax2.imshow(E_beta_J/n, cmap = 'Reds', origin='lower', extent=ext_nrg, aspect='auto', interpolation='spline36')
    ax2.set_title('E/site')
    fig.colorbar(im2, ax=ax2)
    ax2.set_ylabel('B/J')
    ax2.set_xlabel('T')
    
    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint 2 %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ =="__main__":
    main()