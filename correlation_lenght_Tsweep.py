import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import numba as nb
from scipy import optimize

time_start = time.perf_counter()

lattice_type = 'square'            #write square, triangular or hexagonal
J = -0.5                       #spin coupling constant
B = 0                     #external magnetic field
M = 20                          #lattice size MxN
N = 20
steps = 500                      #number of evolution steps per given temperature
max_r = 15

Tc = (2*abs(J))/np.log(1+np.sqrt(2))         #Onsager critical temperature for square lattice

T = np.linspace(0.1, 2, 50)

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

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

def distances(num, spl):
    lenghts = np.empty((num, num))
    for i in range(num):
        dictionary_per_node_i = spl[i][1]                   #returns an nxn matrix, element m,n is the distance between node m and node n
        for j in range(num):
            lenghts[i, j] = dictionary_per_node_i[j]
    return lenghts

@jit(nopython=True)
def step(A_dense, beta, num, nn_number, lenghts):

    spinlist = np.random.choice(np.array([-1, 1]), num)   #create random spins for nodes
    
    l=0
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

        #change spins if energetically favourable or according to thermal noise
        for offset in range(2):                 #offset to avoid interfering with neighboring spins while rastering
            for i in range(offset,len(dE),2):
                if dE[i]<0:
                    spinlist[i] *= -1
                elif dE[i]==0:
                    if np.exp(-(E/num)*beta) > np.random.rand():
                        spinlist[i] *= -1
                    else:
                        continue
                elif np.exp(-dE[i]*beta) > np.random.rand():     #thermal noise
                    spinlist[i] *= -1
        l+=1
        #print(l)
    
    r = []
    corr_r = []
    for radius in np.arange(1, max_r+1):
        corr=0
        for atom in range(num):
            mean = 0
            for neighbour in range(num):
                if lenghts[atom, neighbour] == radius:                  
                    corr += (spinlist[atom]*spinlist[neighbour])    #measures correlation for a given distance
                    mean += spinlist[neighbour]
            corr2 = corr/(nn_number*radius)-mean**2/(nn_number*radius)**2
        corr3 = corr2/num
        r.append(radius)                        
        corr_r.append(abs(corr3))         #measures correlation in relation to distance  

    return corr_r, r 


def main():

    #create lattice
    G, nn_number = lattice(M, N)
    #convert node labels to integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    #get number of nodes
    n = num(G)

    spl = (list(nx.all_pairs_shortest_path_length(G)))
                
    #extract adjacency matrix and convert to numpy dense array
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A_dense = Adj.todense()

    lenghts = distances(n, spl)

    cov = []
    xis = []
    for i in range(len(T)):
        #iterate steps and sweep trough beta
        corr_r, r = step(A_dense, 1/T[i], n, nn_number, lenghts)

        def func(x, cl):
            return np.exp(-x/cl)

        xi, pcov = optimize.curve_fit(func, r, corr_r)
        
        print('{}/{}'.format(i, len(T)))
        print(xi[0])

        xis.append(xi[0])
        cov.append(np.sqrt(pcov[0][0]))

        y=[]
        for i in range(max_r):
            y.append(func(r[i], xi).item())

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    index_max=0
    max = 0
    for t in range(len(xis)):
        if xis[t]>max:
            max = xis[t]
            index_max = t
    
    def exp(x, a, b):
        return np.exp(a*(x+b))

    decay_fit, decay_cov = optimize.curve_fit(exp, T[index_max:], xis[index_max:], p0=[-1, -1], sigma=cov[index_max:])
    raising_fit, raising_cov = optimize.curve_fit(exp, T[:index_max], xis[:index_max], p0=[1, 1], sigma=cov[:index_max])

    decay_exp=[]
    for c in range(len(xis)):
        if c<index_max:
            continue
        else:
            decay_exp.append(exp(T[c], decay_fit[0], decay_fit[1]))

    raising_exp=[]
    for c in range(len(xis)):
        if c>=index_max:
            continue
        else:
            raising_exp.append(exp(T[c], raising_fit[0], raising_fit[1]))

    plt.plot(T[index_max:], decay_exp, label='exp({}(T+{})'.format(decay_fit[0], decay_fit[1]))
    plt.plot(T[:index_max], raising_exp, label='exp({}(T+{})'.format(raising_fit[0], raising_fit[1]))
    plt.legend()

    plt.plot(T/Tc, xis)
    plt.errorbar(T/Tc, xis, cov, fmt='.')
    plt.title('correlation lenght vs T/Tc')
    plt.show()

if __name__ =="__main__":
    main()