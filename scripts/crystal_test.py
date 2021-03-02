# This works only in python 3
from kimonet.system.generators import regular_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet import system_test_info
from kimonet.system.molecule import Molecule
from kimonet.system.state import State
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition
from kimonet import calculate_kmc, calculate_kmc_parallel, calculate_kmc_parallel_py2


# states list
s1 = State(label='s1', energy=1.0, multiplicity=1)

# transition moments
transition_moment = {Transition(s1, gs): [0.1, 0.0]}

molecule = Molecule()

#################################################################################

num_trajectories = 500                          # number of trajectories that will be simulated
max_steps = 100                              # maximum number of steps for trajectory allowed

system = regular_system(molecule=molecule,
                        lattice={'size': [3, 3],
                                 'parameters': [3.0, 3.0]},  # Angstroms
                        orientation=[0, 0, 0])

visualize_system(system)
system.cutoff_radius = 4.0  # Angstroms

system.process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                    electronic_coupling_function=forster_coupling,
                                    arguments={'ref_index': 1,
                                               'transition_moment': transition_moment},
                                    vibrations=MarcusModel(),
                                    description='forster'),
                         DecayRate(initial_state=s1, final_state=gs,
                                   decay_rate_function=einstein_radiative_decay,
                                   arguments={'transition_moment': transition_moment},
                                   description='decay')]


system.add_excitation_index(s1, 1)
system_test_info(system)


trajectories = calculate_kmc(system,
                             num_trajectories=5,    # number of trajectories that will be simulated
                             max_steps=1000,         # maximum number of steps for trajectory allowed
                             silent=False)


analysis = TrajectoryAnalysis(trajectories)

print('diffusion coefficient (average): {} angs^2/ns'.format(analysis.diffusion_coefficient('s1')))
print('lifetime: {} ns'.format(analysis.lifetime('s1')))
print('diffusion length: {} angs'.format(analysis.diffusion_length('s1')))

print('diffusion tensor')
print(analysis.diffusion_coeff_tensor('s1'))
print('diffusion length tensor')
print(analysis.diffusion_length_square_tensor('s1'))
# print(np.sqrt(analysis.diffusion_coeff_tensor()*analysis.lifetime()*2))

plt = analysis.plot_2d()
plt.figure()
analysis.plot_distances()
plt.show()
