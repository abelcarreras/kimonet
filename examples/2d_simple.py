from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import system_test_info
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis import plot_polar_plot
from kimonet import calculate_kmc, calculate_kmc_parallel
from kimonet.system.state import State
import numpy as np


# states list
gs = State(label='gs', energy=0.0, multiplicity=1)
s1 = State(label='s1', energy=1.0, multiplicity=1)

# list of transfer functions by state
transfer_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                              electronic_coupling_function=forster_coupling,
                              description='Forster')
                   ]

# list of decay functions by state
decay_scheme = [DecayRate(initial_states=s1, final_states=gs,
                          decay_rate_function=einstein_radiative_decay,
                          description='singlet_radiative_decay')
                ]



molecule = Molecule(#states=[State(label='gs', energy=0.0),   # eV
                    #        State(label='s1', energy=4.0)],  # eV
                    vibrations=MarcusModel(reorganization_energies={(s1, gs): 0.08,  # eV
                                                                    (gs, s1): 0.08},
                                           temperature=300),  # Kelvin
                    transition_moment={(s1, gs): [0.1, 0.0]},  # Debye
                    decays=decay_scheme,
                    )


# physical conditions of the system
conditions = {'refractive_index': 1}

# define system as a crystal
system = crystal_system(conditions=conditions,
                        molecules=[molecule],  # molecule to use as reference
                        scaled_site_coordinates=[[0.0, 0.0]],
                        unitcell=[[5.0, 1.0],
                                  [1.0, 5.0]],
                        dimensions=[2, 2],  # supercell size
                        orientations=[[0.0, 0.0, np.pi/2]])  # if element is None then random, if list then Rx Ry Rz

# set initial exciton
system.add_excitation_index(s1, 0)

# set additional system parameters
system.transfer_scheme = transfer_scheme
system.cutoff_radius = 8  # interaction cutoff radius in Angstrom

# some system analyze functions
system_test_info(system)

#visualize_system(system)
#visualize_system(system, dipole='s1')

# do the kinetic Monte Carlo simulation
trajectories = calculate_kmc(system,
                             num_trajectories=5,    # number of trajectories that will be simulated
                             max_steps=1000,         # maximum number of steps for trajectory allowed
                             silent=False)

# specific trajectory plot
trajectories[0].plot_graph().show()
trajectories[0].plot_2d().show()

# resulting trajectories analysis
analysis = TrajectoryAnalysis(trajectories)

for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {:9.5e} Angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime:              {:9.5e} ns'.format(analysis.lifetime(state)))
    print('diffusion length:      {:9.5e} Angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))

    print('diffusion length tensor (Angs)')
    print(analysis.diffusion_length_square_tensor(state))

    plot_polar_plot(analysis.diffusion_coeff_tensor(state),
                    title='Diffusion', plane=[0, 1])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state),
                    title='Diffusion length square', crystal_labels=True, plane=[0, 1])


analysis.plot_excitations('s1').show()
analysis.plot_2d('s1').show()
analysis.plot_distances('s1').show()
analysis.plot_histogram('s1').show()
analysis.plot_histogram('s1').savefig('histogram_s1.png')

store_trajectory_list(trajectories, 'example_simple.h5')
