import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.colors

lattice_type = 2            #select 1 for square, 2 for triangular, 3 for hexagonal
J = 0.2                         #coupling constant
h = 0.02                          #external field
M = 3 #rows                           #lattice size MxN
N = 2 #columns
steps = 10                      #number of steps per given temperature
pbc = False #periodic boundary conditions
   
#beta = np.logspace(1, 4, num=4, base=10.0)   #array of temperatures to sweep as of README
beta = 10 #np.linspace(0.1, 10000, 1000)
#spero abbiano senso le unità di misura
beta_lsp = np.linspace(0.1, 1000, 100)
h_lsp = np.linspace(0.1, 1000, 100)

#function creates lattice
def lattice(M, N):
    if lattice_type == 3:
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
    elif lattice_type == 2:
        lattice = nx.triangular_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
    elif lattice_type == 1:
        lattice = nx.grid_2d_graph(M, N, periodic=pbc, create_using=None)
    return lattice

#creates lattice con posizione????
def lattice_v2(M, N):
    if lattice_type == 3:
        lattice = nx.hexagonal_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 2:
        lattice = nx.triangular_lattice_graph(M, N, periodic=pbc, with_positions=True, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = nx.get_node_attributes(lattice, 'pos') #use for any shape other than square
    elif lattice_type == 1:
        lattice = nx.grid_2d_graph(M, N, periodic=pbc, create_using=None)
        lattice = nx.convert_node_labels_to_integers(lattice, first_label=0, ordering='default', label_attribute=None)
        pos = generate_grid_pos(lattice, M, N) #use for 2D grid network

    return lattice, pos

#function assigns random spin up/down to nodes
def spinass(G):
    i = 0
    for node in G:
        G.nodes[node]['spin'] = np.random.choice([-1, 1])
        G.nodes[node]['label'] = i
        i += 1

#Questa funzione è letteralmente l'amiltoniana e quindi l'energia totele del sistema
def H(G):
    Adj = nx.adjacency_matrix(G).todense() #mi serve la matrice di adiacenza per capire se c'è un collegamento tra i due nodi!
    E = 0
    for i in G.nodes():
        for j in G.nodes():
            if Adj[G.nodes[i]['label']][G.nodes[j]['label']] != 0: #ci potrebbe essere un modo più efficientedi farlo ma mi da come risultato un (1,2) che non so cosa sia
                E += -J*G.nodes[i]["spin"]*G.nodes[j]["spin"]
            E += - h*G.nodes[i]['spin'] #questa è la parte di energia relativa al campo
    return E

#algoritmo metropolis preso da wikipedia/Emiliano
def Metropolis(G):
    k = np.random.choice(G.number_of_nodes()) #scelgo un nodo random
    E = H(G)    
    for i in G.nodes():
        if k != 0 and G.nodes[i]['label'] == k:
            G.nodes[i]['spin'] *= -1 #spin flip
            if E - H(G) > 0 and np.random.rand() < np.exp(-(E - H(G))*beta):
                G.nodes[i]['spin'] *= -1 #se è piu alta c'è perdita di energia e quidi ritorno a prima
    #return G

#si potrebbe migliorare quella cosa della Adj e capire dove inserire il continue perchè non serve che scorra tutti gi nodi e capire se la k potrebbe essere anche 0

def Metropolis_v2(G,x,y): #qui voglio fare returnare le varie liste
    beta = 1/(y*J)
    k = np.random.choice(G.number_of_nodes()) #scelgo un nodo random
    E = H(G) 
    btw_cnt = np.array([])
    for b in range(len(beta)):
        for j in range(steps):
            for i in G.nodes():
                if k != 0 and G.nodes[i]['label'] == k:
                    G.nodes[i]['spin'] *= -1 #spin flip
                    if E - H(G) > 0 and np.random.rand() < np.exp(-(E - H(G))*beta[b]):
                        G.nodes[i]['spin'] *= -1 #se è piu alta c'è perdita di energia e quidi ritorno a prima
        np.append(btw_cnt, nx.betweenness_centrality(G)[3])
    return btw_cnt

def ciao(x,y):
    c = x*y 
    return c
        

#creates color map
def colormap(G):
    color=[]
    for node in G:
        if G.nodes[node]['spin']==1:
            color.append('red')
        else:
            color.append('black')
    return color


def main():

    #create lattice
    G, pos = lattice_v2(M, N)
    #assegno gli spin
    spinass(G)

    #color = colormap(G)
    #nx.draw(G, node_color=color, node_size=100, edge_color='white', pos=pos, with_labels=False)
    #plt.show() 

    #for i in range(steps):
    #    Metropolis(G)

    #color = colormap(G)
    #nx.draw(G, node_color=color, node_size=100, edge_color='white', pos=pos, with_labels=False)
    #plt.show() 

    #ho visto che nel paper plotta
    #y = 1 / (beta_lsp*J)
    #x = h_lsp / J
    

    x = y = np.linspace(0.1,1.5)
    X,Y = np.meshgrid(x,x)
    print(type(X))
    z = Metropolis_v2(G,X,Y)
    print(type(z))
    Z = np.exp(-(X**2+Y**2))
    fig, (ax, ax2) = plt.subplots(ncols=2)

    colors=["red", "orange", "gold", "limegreen", "k", 
            "#550011", "purple", "seagreen"]

    ax.set_title("contour with color list")
    contour = ax.contourf(X,Y,Z, colors=colors)

    ax2.set_title("contour with colormap")
    cmap = matplotlib.colors.ListedColormap(colors)
    contour = ax2.contourf(X,Y,z, cmap=cmap)
    fig.colorbar(contour)

    plt.show()
        

if __name__ =="__main__":
    main()