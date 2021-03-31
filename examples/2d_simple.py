from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import system_test_info
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis import plot_polar_plot
from kimonet import calculate_kmc, calculate_kmc_parallel, calculate_kmc_parallel_py2
from kimonet.system.state import State
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition

import numpy as np


# states list
s1 = State(label='s1', energy=20.0, multiplicity=1)

# transition moments
transitions = [Transition(s1, gs,
                          tdm=[0.1, 0.0],  # a.u.
                          reorganization_energy=0.08)] # eV



# define system as a crystal
molecule = Molecule()

#print(molecule, molecule.state, molecule.state.get_center())

molecule2 = Molecule(site_energy=2)

print(molecule2, molecule2.state, molecule2.state.get_center())
print(molecule, molecule.state, molecule.state.get_center())


system = crystal_system(molecules=[molecule, molecule],  # molecule to use as reference
                        scaled_site_coordinates=[[0.0, 0.0],
                                                 [0.0, 0.5]],
                        unitcell=[[5.0, 1.0],
                                  [1.0, 5.0]],
                        dimensions=[2, 2],  # supercell size
                        orientations=[[0.0, 0.0, np.pi/2],
                                      [0.0, 0.0, 0.0]])  # if element is None then random, if list then Rx Ry Rz

print([m.site_energy for m in system.molecules])
print(system.get_ground_states())


# set initial exciton
system.add_excitation_index(s1, 0)
system.add_excitation_index(s1, 1)

# set additional system parameters
system.process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                    electronic_coupling_function=forster_coupling,
                                    description='Forster coupling',
                                    arguments={'ref_index': 1,
                                               'transitions': transitions},
                                    vibrations=MarcusModel(transitions=transitions) # eV
                                    ),
                        DecayRate(initial_state=s1, final_state=gs,
                                  decay_rate_function=einstein_radiative_decay,
                                  arguments={'transitions': transitions},
                                  description='custom decay rate')
                        ]

system.cutoff_radius = 8  # interaction cutoff radius in Angstrom

# some system analyze functions
system_test_info(system)
visualize_system(system)

# do the kinetic Monte Carlo simulation
trajectories = calculate_kmc(system,
                             num_trajectories=5,    # number of trajectories that will be simulated
                             max_steps=100,         # maximum number of steps for trajectory allowed
                             silent=False)

# specific trajectory plot
trajectories[0].plot_graph().show()
trajectories[0].plot_2d().show()

# resulting trajectories analysis
analysis = TrajectoryAnalysis(trajectories)

print('diffusion coefficient: {:9.5e} Angs^2/ns'.format(analysis.diffusion_coefficient()))
print('lifetime:              {:9.5e} ns'.format(analysis.lifetime()))
print('diffusion length:      {:9.5e} Angs'.format(analysis.diffusion_length()))

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

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=[[5.0, 1.0],
                                                                              [1.0, 5.0]]),
                    title='Diffusion length square', crystal_labels=True, plane=[0, 1])


analysis.plot_exciton_density('s1').show()
analysis.plot_2d('s1').show()
analysis.plot_distances('s1').show()
analysis.plot_histogram('s1').show()
analysis.plot_histogram('s1').savefig('histogram_s1.png')

store_trajectory_list(trajectories, 'example_simple.h5')
