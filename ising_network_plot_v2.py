import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

time_start = time.perf_counter()

J = -1e-4
beta = 10000
steps = 15

#creates lattice
def lattice(M, N):
    lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    return lattice

#assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#creates color map
def colormap(G):
    color=[]
    for node in G:
        if G.nodes[node]['spin']==1:
            color.append('red')
        else:
            color.append('black')
    return color

#function for single step
def step(G):
    #create ordered list of spins
    spin = nx.get_node_attributes(G, 'spin')

    spinlist = list(dict.values(spin))

    #create adjacency matrix and change all values of adjacency (always 1) with the value of spin of the neighbour
    Adj = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')
    A = Adj.todense()
    for m in range(A.shape[1]):  #A.shape[1] gives number of nodes
        for n in range(A.shape[1]):
            if A[m,n]==1:
                A[m,n]=spinlist[n] #assigned to every element in the adj matrix the corresponding node spin value

    #sum over rows to get total spin of neighbouring atoms for each atom
    N = np.sum(A,axis=1).tolist()

    #What decides the flip is
    dE=2*J*np.multiply(N,spinlist) 

    #make it into a dictionary
    dEdict = {}
    i = 0
    for node in G:
        dEdict[node]=dE[i]
        i+=1

    #Now flip every spin whose dE<0
    for node in G:
        if dEdict[node]<=0:
            spin[node]*=-1
        elif np.exp(-dEdict[node]*beta) > np.random.rand():
            spin[node] *= -1

    #update spin values in graph
    nx.set_node_attributes(G, spin, 'spin')

#iterate steps and print
def iter(G, steps):
    i=0
    while i <= steps:
        step(G)

        #update color map
        color = colormap(G)

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)

        plt.savefig('time_ev/step({}).png'.format(i+1))
    
        print(i)
    
        i+=1

def main():
    #create lattice
    G = lattice(20, 20)
    #run it
    spinass(G)

    #run first step to visualize initial condiditons
    color = colormap(G)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
    plt.savefig('time_ev/step(0).png')

    #iterate given number of times
    iter(G, steps)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

if __name__ =="__main__":
    main()