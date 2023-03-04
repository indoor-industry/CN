import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

Jbeta = 0.2

#create lattice
def lattice(M, N):
    lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    #lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
    return lattice

G = lattice(20, 20)

#assign random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#run it
spinass(G)

#create color map
def colormap(G):
    color=[]
    for node in G:
        if G.nodes[node]['spin']==1:
            color.append('red')
        else:
            color.append('black')
    return color

#massive function for single step
E = []
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
    dEbeta=Jbeta*np.multiply(N,spinlist) #half the change in energy multiplied by beta (corresponds to the energy*beta)

    E.append(2*sum(dEbeta))

    #make it into a dictionary
    dEdict = {}
    i = 0
    for node in G:
        dEdict[node]=dEbeta[i]
        i+=1

    #Now flip every spin whose dE<0
    for node in G:
        if dEdict[node]<=0:
            spin[node]*=-1
        elif np.exp(-dEdict[node]) > np.random.rand():
            spin[node] *= -1

    #update spin values in graph
    nx.set_node_attributes(G, spin, 'spin')
    return E


#first step otherwise it gets aligned to early
color = colormap(G)
pos = nx.get_node_attributes(G, 'pos')
#nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
#plt.savefig('img/img(0).png')


#iterate steps and print
i=0
while i <= 50:
    step(G)

    #update color map
    color = colormap(G)

    pos = nx.get_node_attributes(G, 'pos')
#    nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
    #node_labels = nx.get_node_attributes(G,'spin')
    #nx.draw_networkx_labels(G, pos, labels = node_labels)
    #plt.show()

#    plt.savefig('img/img({}).png'.format(i+1))

    i+=1

#energy plot
E = step(G)
print(E)
plt.plot(E)
plt.show()
