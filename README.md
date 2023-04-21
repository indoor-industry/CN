# Monte Carlo simulation of the 2D Ising model

Using the Metropolis-Hastings algorithm in a complex network setting
The code works for square, triangular and hexagonal lattice types and also for a random Erdos-Renyi graph (work in progress)

ising_v3.py gives a visual simulation (or energy plot) of a 2d grid iding model to test for consistency tests of the other scripts

ising_network_plot_v5.py saves in time_ev snapshots of an evolving ising network

ising_network_nrg_v8.py sweeps trough values of beta and plots various parameters against temperature

ising_network_nrg_time_v3.py plots magnetisation and energy in time

ising_network_double_plots_v3.py plots a 2D colormap of energy and magnetisation in both T and external field B

ising_network_clustering_v6.py plots the clustered version of the network after equilibrium is reached
The clustered version of the network is the one in which only edges between equal spins are kept, hence it shows clusters of aligned spins

ising_network_clustering_sweep_v2.py plots some network parameters of the clustered network against temperature

ising_network_clustering_doublesweep_v2.py is similar to double_plots above but for the clustered network

weighted_network_plot_v2.py creates a fully connected network of nodes with edges the average correlation in time and trough the lattice between spins

weighted_network_doucle_plots_v2.py as above but with density

correlation_lenght.py plots and fits the correlation in a specific temperature and fits it to an exponential to find the correlation lenght (work in progress)

correlation_lenght sweeps speak for themselves