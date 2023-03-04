import matplotlib.pyplot as plt
import numpy as np
from numba import jit

#specify size of lattice
M=500
N=500

#temperature
Jbeta = 0.8

#create lattice with random values of 1 and -1

def lattice(N, M):
    return np.random.choice([-1, 1], size=(N, M))

#sum over nearest neighbours of element mn
#calculate difference in energy dE
#if negative flip spin to reach more stable configuration
#account for thermal fluctuations

@jit
def update(field, n, m, Jbeta):
    nnsum = 0
    N, M = field.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
            if i == n and j == m:
                continue
            nnsum += field[i % N, j % M] #sum over nearest neighbours
    dEbeta = field[n, m] * nnsum * Jbeta #half the change in energy multiplied by beta (corresponds to the energy*beta)
    if dEbeta <= 0:
        field[n, m] *= -1
    elif np.exp(-dEbeta) > np.random.rand():
        field[n, m] *= -1
    return dEbeta

#raster the spin update function trough the lattice
#account for offset to avoid bias of changing one spin and then counting it in the next iteration

@jit
def step(field, Jbeta):
    E=0 #total energy of lattice
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    update(field, n, m, Jbeta)
                    E+=update(field, n, m, Jbeta)
    return field, E

#define a lattice and print timesteps of evolution
L = lattice(M, N)

t = 0

#animation
im = plt.imshow(L, cmap='gray', vmin=-1, vmax=1, interpolation='none')
while t<100:
    im.set_data(L)
    plt.draw()
    L, E = step(L, Jbeta)
    plt.pause(.001)
    t += 1


#energy plot
nrg = []
time = []
#while t<100:
#    L, E = step(L, Jbeta)
#    nrg.append(E)
#    time.append(t)
#    t += 1
#A = plt.plot(time, nrg)
#plt.show()