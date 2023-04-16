import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

time_start = time.perf_counter()

lattice_type = 'triangular'            #write square, triangular or hexagonal
M = 70
N = 70                      #MxN size of lattice
J = -1                    #spin-spin coupling strenght
B = 0
T = 1                  #external field (actually is mu*B where mu is magnetic moment of atoms)
steps = 100                 #evolution timesteps

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
    dE= -4*J*np.multiply(nnsum, spinlist) + 2*B*spinlist
    E = J*sum(np.multiply(nnsum, spinlist)) - B*sum(spinlist)   #total energy

    for offset in range(2):
        for i in range(offset,len(dE),2):
            if dE[i]<0:
                G.nodes[i]['spin'] *= -1
            elif dE[i] == 0:
                if np.exp(-(E/num)/T) > np.random.rand():
                    G.nodes[i]['spin'] *= -1
                else:
                    continue
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
        if i % 10==0:            #skip steps if needed
            #update color map
            color = colormap(G)
            nx.draw(G, node_color=color, node_size=20, edge_color='white', pos=pos, with_labels=False)
            plt.pause(0.1)                                    #this shows an animation
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