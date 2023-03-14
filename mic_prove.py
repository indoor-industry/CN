import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

lattice_type = 2            #select 1 for square, 2 for triangular, 3 for hexagonal
J = -0.2                         #coupling constant
B = 0.02                          #external field
M = 2 #rows                           #lattice size MxN
N = 3 #columns
steps = 1                      #number of steps per given temperature
pbc = False #periodic boundary conditions
   
#beta = np.logspace(1, 4, num=4, base=10.0)   #array of temperatures to sweep as of README
beta = 10 #np.linspace(0.1, 10000, 1000)

#function creates lattice
def lattice(M, N):
    if lattice_type == 3:
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
    elif lattice_type == 2:
        lattice = nx.triangular_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
    elif lattice_type == 1:
        lattice = nx.grid_2d_graph(M, N, periodic=pbc, create_using=None)
    return lattice

#function assigns random spin up/down to nodes
def spinass(G):
    for node in G:
        G.nodes[node]['spin'] = np.random.choice([-1, 1])

def main():
    #create lattice
    G = lattice(M, N)
    spinass(G)
   
    for i in G.nodes :
        print(G.nodes[i]['spin'])

    print(G.number_of_nodes())

if __name__ =="__main__":
    main()