from scipy.cluster.hierarchy import cut_tree

from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step, system_test_info
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate, SimpleRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel, SimpleOverlap
from kimonet import calculate_kmc, calculate_kmc_parallel
from kimonet.system.state import State
from kimonet.core.processes.transitions import Transition


# define states
from kimonet.system.state import ground_state as gs  # ground state
s1 = State(label='s1', energy=4.37, multiplicity=1)  # fist excited state

# define transition between states
transitions = [Transition(s1, gs,
                          tdm=[1.0, 0.0, 0.0]), # a.u.
               ]

# define molecules (sites) on which the states are placed
molecule = Molecule(name='my_molecule', site_energy=2, vdw_radius=1.7) # angs

# define the whole system in cheihc the simulation runs
system = crystal_system(molecules=[molecule, molecule, molecule, molecule],
                          scaled_site_coordinates=[[0.0, 0.0, 0.0],  # relative coordinates
                                                   [0.5, 0.0, 0.0],  # relative coordinates
                                                   [0.0, 0.5, 0.0],  # relative coordinates
                                                   [0.0, 0.0, 0.5],  # relative coordinates
                                                   ], # relative coordinates
                          unitcell=[[5.00000000, 0.0000000, 0.00000000],   # angs
                                    [0.00000000, 5.0000000, 0.00000000],   # angs
                                    [0.00000000, 0.0000000, 5.00000000]],  # angs
                          dimensions=[1, 1, 1], # supercell dimensions
                          orientations=[[0.0, 0.0, 0.0], # (Rx, Ry, Rz) rotations in radiants
                                        [0.0, 0.0, 0.0], # (Rx, Ry, Rz) rotations in radiants
                                        [0.0, 0.0, 0.0], # (Rx, Ry, Rz) rotations in radiants
                                        [0.0, 0.0, 0.0], # (Rx, Ry, Rz) rotations in radiants
                                        None]) # None means random orientation

# define the list of processes that can happen in the system
system.process_scheme = [# exciton transfers
                        # SimpleRate(initial_states=(s1, gs), final_states=(gs, s1),
                        #           description='singlet transfer',
                        #           rate_constant=900), # ns-1
                        GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                    electronic_coupling_function=forster_coupling,
                                    description='Forster coupling',
                                    arguments={'ref_index': 1, 'transitions': transitions},
                                    vibrations=SimpleOverlap(fcwd=0.1) # eV
                                    ),
                        # decays
                        DecayRate(initial_state=s1, final_state=gs,
                                  decay_rate_function=einstein_radiative_decay,
                                  arguments={'transitions': transitions},
                                  description='singlet_radiative_decay'),
                        ]

# define the cutoff of the interactions between states
system.cutoff_radius = 2.6 # angs

# define initial positions of excitons (states)
system.add_excitation_index(s1, index=0) # in particular site
#system.add_excitation_random(s1, n_states=1) # random position

# visualize data of the initial system
system_test_info(system)
visualize_system(system, length=2)


# run kinetic Monte Carlo simulation
trajectories = calculate_kmc_parallel(system,
                                      processors=4,
                                      num_trajectories=100,  # number of trajectories that will be simulated
                                      max_steps=500,  # maximum number of steps for trajectory allowed
                                      silent=False)

# properties analysis
analysis = TrajectoryAnalysis(trajectories)

analysis.plot_distances('s1').show()
analysis.plot_histogram('s1').show()
print('diffusion length S1: {} angs'.format(analysis.diffusion_length('s1')))
print('lifetime: {} ns'.format(analysis.lifetime('s1')))


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))

