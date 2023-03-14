import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

lattice_type = 2            #select 1 for square, 2 for triangular, 3 for hexagonal
J = 0.2                         #coupling constant
h = 0.02                          #external field
M = 2 #rows                           #lattice size MxN
N = 3 #columns
steps = 1                      #number of steps per given temperature
pbc = False #periodic boundary conditions
   
#beta = np.logspace(1, 4, num=4, base=10.0)   #array of temperatures to sweep as of README
beta = 10 #np.linspace(0.1, 10000, 1000)

#function creates lattice
def lattice(M, N):
    if lattice_type == 3:
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=pbc, with_positions=False, create_using=None)
    elif lattice_type == 2:
        lattice = nx.triangular_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
    elif lattice_type == 1:
        lattice = nx.grid_2d_graph(M, N, periodic=pbc, create_using=None)
    return lattice

#function assigns random spin up/down to nodes
def spinass(G):
    i = 0
    for node in G:
        G.nodes[node]['spin'] = np.random.choice([-1, 1])
        G.nodes[node]['label'] = i
        i += 1

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

    for offset in range(2):
        for i in range(offset,len(dE),2):
            if dE[i]<=0:
                G.nodes[i]['spin'] *= -1
            elif np.exp(-dE[i]*beta) > np.random.rand():
                G.nodes[i]['spin'] *= -1
    
    return G, E

def H(G):
    Adj = nx.adjacency_matrix(G).todense() #mi serve la matrice di adiacenza per capire se c'è un collegamento tra i due nodi!
    E = 0
    for i in G.nodes():
        for j in G.nodes():
            if Adj[G.nodes[i]['label']][G.nodes[j]['label']] != 0: #ci potrebbe essere un modo più efficientedi farlo ma mi da come risultato un (1,2) che non so cosa sia
                E += -J*G.nodes[i]["spin"]*G.nodes[j]["spin"]
    return E

def Metropolis(G):
    k = np.random.choice(G.number_of_nodes()) #scelgo un nodo random
    E = H(G)    
    for i in G.nodes()
        if k != 0 and G.nodes[i]['label'] == k
            G.nodes[i]['spin'] *= -1 #spin flip
        if E - H(G) > 0
            G.nodes[i]['spin'] *= -1 #se è piu alta c'è perdita di energia e quidi ritorno a prima


def main():

    #create lattice
    G = lattice(M, N)
    spinass(G)
   
    for i in G.nodes :
        print(G.nodes[i]['spin'])

    print(nx.adjacency_matrix(G).todense()[1][2])
    print(H(G))
    
    i = np.random.choice(G.number_of_nodes())
    print(i)
    #print(G.nodes(data = True))

    #nx.draw_spectral(G)
    #plt.show() 

if __name__ =="__main__":
    main()