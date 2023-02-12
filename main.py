#HINT: SCALE BETA BY K BOLTZMANN TO MAKE MORE SENSE OF IT

import matplotlib.pyplot as plt
import numpy as np
import math

#specify size of lattice
M=200
N=200

J=1 # coupling between spins
beta = 0.4 #inverse temperature in units of energy 

#notice the balance between J and beta are crucial for the formation of coherent zones of aligned spin

#create lattice with random values of 1 and -1
def lattice(N, M):
    return np.random.choice([-1, 1], size=(N, M))

#sum over nearest neighbours of element mn
#calculate difference in energy dE
#if negative flip spin to reach more stable configuration
#account for thermal fluctuations
def update(field, n, m, beta):
    E = 0
    N, M = field.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
            if i == n and j == m:
                continue
            E += field[i % N, j % M]
    dE = J * field[n, m] * E
    if dE <= 0:
        field[n, m] *= -1
    elif np.exp(-dE * beta) > np.random.rand():
        field[n, m] *= -1

#raster the spin update function trough the lattice
#account for offset to avoid bias of changing one spin and then counting it in the next iteration
def step(field, beta):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    update(field, n, m, beta)
    return field

#define a lattice and print timesteps of evolution
L = lattice(M, N)

im = plt.imshow(L, cmap='gray', vmin=-1, vmax=1, interpolation='none')
t = 0
while t<20:
    im.set_data(L)
    plt.draw()
    L = step(L, beta)
    plt.pause(.001)
    t += 1