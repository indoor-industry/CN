#Credits to Mr. P Solver for the inspiration

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

function = 'animation'          #select 'animation' or 'energy plot'

#specify size of lattice
M=500
N=500

#temperature
J = -0.5
B = 0
T = 1
beta = 1/T
steps = 100

#create lattice with random values of 1 and -1

def lattice(N, M):
    return np.random.choice([-1, 1], size=(N, M))

#sum over nearest neighbours of element mn
#calculate difference in energy dE
#if negative flip spin to reach more stable configuration
#account for thermal fluctuations

@jit
def update(field, n, m, beta, E):
    nnsum = 0
    N, M = field.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
            if i == n and j == m:
                continue
            nnsum += field[i % N, j % M] #sum over nearest neighbours
    dE = -4*field[n, m] * nnsum * J + 2*B*field[n, m]  #change in energy
    E = J*field[n, m]*nnsum - B*field[n, m]    #energy of single site
    if dE <= 0:
        field[n, m] *= -1
    #elif dE==0:
    #    if np.exp(-(E/(M*N))*beta) > np.random.rand():
    #        field[n, m] *= -1
    elif np.exp(-dE*beta) > np.random.rand():
        field[n, m] *= -1
    return E

#raster the spin update function trough the lattice
#account for offset to avoid bias of changing one spin and then counting it in the next iteration

@jit
def step(field, beta):
    E=0 #total energy of lattice
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    update(field, n, m, beta, E)
                    E+=update(field, n, m, beta, E)
    return field, E

#define a lattice and print timesteps of evolution
L = lattice(M, N)

if function == 'animation':
    t = 0
    #animation
    im = plt.imshow(L, cmap='gray', vmin=-1, vmax=1, interpolation='none')
    while t<steps:
        im.set_data(L)
        plt.draw()
        L, E = step(L, beta)
        plt.pause(.001)
        t += 1

elif function == 'energy plot':
    t = 0
    #energy plot
    steps = 50
    N_array = M*N*np.ones(steps)
    nrg = []
    time = []
    while t < steps:
        L, E = step(L, beta)
        nrg.append(E)
        time.append(t)
        t += 1
    A = plt.plot(time, nrg/N_array)
    plt.show()


