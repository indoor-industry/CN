# Monte Carlo simulation of the 2D Ising model

Using the Metropolis-Hastings algorithm

main.py gives a visual simulation (or energy plot) of a 2d grid iding model to test for parameter values such as J and beta

ising_network_plot.py saves in time_ev snapshots of an ising network

ising_network_nrg.py sweeps trough values of beta and plots the time evolution of energy/lattice site
ising_network_nrg_opt.py is faster

clustering.py creates a new graph with only clusters of neighbours with same spins, to calculate relevant network parameters

Typical values of constants:
T usually from 0 to 1e3 K
set k=1 (then T is measured in terms of eV)
since beta=1/T: beta goes from 1e-3 K^(-1) to inf K^(-1)
but since k=1e-4eV/K, if set to 1 this implies 1 K^(-1)=1e4 eV^(-1)
hence notice beta in the code will be given in values ranging from 10 ev^(-1) to inf ev^(-1)

meanwhile J usually around 1e-4 (magnetic interaction)

for T=0.1    beta=    10
for T=  1    beta=    1
for T=  10   beta=  0.1
for T=  100  beta= 0.01
for T=  1000 beta=0.001