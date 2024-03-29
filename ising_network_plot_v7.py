import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

time_start = time.perf_counter()

lattice_type = 'PT226'            #write square, triangular or hexagonal, PT31, PT86, PT226
M = 20
N = 20                      #MxN size of lattice, don't count for PT
J = 1                    #spin-spin coupling strenght
B = 0
T = 1                  #external field (actually is mu*B where mu is magnetic moment of atoms)
steps = 1000                 #evolution timesteps

#creates lattice
def lattice(M, N):
    if lattice_type == 'hexagonal':
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 'triangular':
        lattice = nx.triangular_lattice_graph(M, N, periodic=True, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 'square':
        lattice = nx.grid_2d_graph(M, N, periodic=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = generate_grid_pos(lattice, M, N) #use for 2D grid network
    elif lattice_type == 'PT86':
        edges = np.loadtxt('PT/nnbond86.txt')
        adj = np.zeros((86, 86))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate86.txt')
        pos = []
        for node in range(86):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT226':
        edges = np.loadtxt('PT/nnbond226.txt')
        adj = np.zeros((226, 226))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate226.txt')
        pos = []
        for node in range(226):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
        lattice = nx.from_numpy_array(adj)
    
    elif lattice_type == 'PT31':
        edges = np.loadtxt('PT/nnbond31.txt')
        adj = np.zeros((31, 31))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate31.txt')
        pos = []
        for node in range(31):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
     
    elif lattice_type == 'PT601':
        edges = np.loadtxt('PT/nnbond601.txt')
        adj = np.zeros((601, 601))
        for m in range(len(edges)):
                bond = edges[m]
                i = int(bond[0]) -1
                j = int(bond[1]) -1
                adj[i][j] = 1

        positions = np.loadtxt('PT/coordinate601.txt')
        pos = []
        for node in range(601):
            position_node = positions[node]
            pos.append((position_node[0], position_node[1]))
        lattice = nx.from_numpy_array(adj)
    
    return lattice, pos

def generate_grid_pos(G, M, N):
    p = []
    for m in range(M):
        for n in range(N):
            p.append((n, m))
    
    grid_pos = {}
    k = 0
    for node in G:
        grid_pos[node]=p[k]
        k+=1
    return grid_pos

#assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin']=np.random.choice([-1, 1])

#count number of sites in lattice
def num(G):
    n = 0
    for node in G:
        n += 1
    return n

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
def step(G, num):
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
    dE= 2*J*np.multiply(nnsum, spinlist) + 2*B*spinlist

    i = np.random.randint(num)
    
    if dE[i]<=0:
        G.nodes[i]['spin'] *= -1
    elif np.exp(-dE[i]/T) > np.random.rand():
        G.nodes[i]['spin'] *= -1    

    return G

#iterate steps and print
def iter(G, steps, pos, num):
    
    #run first step to visualize initial condiditons
    color = colormap(G)

    nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
        
    #plt.savefig('time_ev/step(0).png')         #save images
    plt.pause(1)                                  #animation
    
    i=0
    while i <= steps:
        G = step(G, num)
        if i % 100==0:            #skip steps if needed
            #update color map
            color = colormap(G)
            nx.draw(G, node_color=color, node_size=20, edge_color='black', pos=pos, with_labels=False)
            plt.pause(0.01)                                    #this shows an animation
            plt.savefig('time_ev/step({}).png'.format(i+1))  #this saves the series of images
        print(i)
        i+=1

def main():
    #create lattice
    G, pos = lattice(M, N)
    #number of nodes
    n = num(G)
    #run it
    spinass(G)
    #iterate given number of times
    iter(G, steps, pos, n)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

if __name__ =="__main__":
    main()