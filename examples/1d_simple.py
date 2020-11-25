from kimonet.core.processes import DecayRate, DirectRate
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc, calculate_kmc_parallel_py2, calculate_kmc_parallel
from kimonet.system.state import ground_state as gs
import numpy as np


# custom transfer functions
def transfer_rate(initial, final, custom_constant=1):

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()
    distance = np.linalg.norm(r_vector)

    return custom_constant/distance**2


# custom decay functions
def decay_rate(initial, final):
    rates = {'TypeA': 1. / 100,
             'TypeB': 1. / 50,
             'TypeC': 1. / 25}
    return rates[initial[0].get_center().name]


# states list
s1 = State(label='s1', energy=1.0, multiplicity=1, size=1)
s2 = State(label='s2', energy=1.5, multiplicity=1)

# setup molecules
molecule = Molecule()

molecule1 = molecule.copy()
molecule1.set_coordinates([0])
molecule1.name = 'TypeA'

molecule2 = molecule.copy()
molecule2.set_coordinates([1])
molecule2.name = 'TypeB'

molecule3 = molecule.copy()
molecule3.set_coordinates([2])
molecule3.name = 'TypeC'


# setup system
system = System(molecules=[molecule1, molecule2, molecule3],
                conditions={'custom_constant': 1},
                supercell=[[3]])

# set initial exciton
system.add_excitation_index(s1, 1)
system.add_excitation_index(s2, 2)


# set additional system parameters
system.process_scheme = [DirectRate(initial_states=(s1, gs), final_states=(gs, s1),
                                    rate_constant_function=transfer_rate,
                                    description='custom'),
                         DirectRate(initial_states=(s2, gs), final_states=(gs, s2),
                                    rate_constant_function=transfer_rate,
                                    description='custom'),
                         DecayRate(initial_states=(s1), final_states=(gs),  # TO SYSTEM
                                   decay_rate_function=decay_rate,
                                   description='custom decay rate'),
                         DecayRate(initial_states=(s2), final_states=(gs),  # TO SYSTEM
                                   decay_rate_function=decay_rate,
                                   description='custom decay rate')
                         ]

system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

# some system analyze functions
# system_test_info(system)
# visualize_system(system)

# do the kinetic Monte Carlo simulation
parallel_run = False
if parallel_run:
    # Only Python 2.7
    trajectories = calculate_kmc_parallel_py2(system,
                                          processors=6,
                                          num_trajectories=1000,    # number of trajectories that will be simulated
                                          max_steps=100000,         # maximum number of steps for trajectory allowed
                                          silent=False)
else:
    trajectories = calculate_kmc(system,
                                 num_trajectories=10,    # number of trajectories that will be simulated
                                 max_steps=100000,         # maximum number of steps for trajectory allowed
                                 silent=False)


# Results analysis
analysis = TrajectoryAnalysis(trajectories)

print('diffusion coefficient: {:9.5e} Angs^2/ns'.format(analysis.diffusion_coefficient('s1')))
print('lifetime:              {:9.5e} ns'.format(analysis.lifetime('s1')))
print('diffusion length:      {:9.5e} Angs'.format(analysis.diffusion_length('s1')))
print('diffusion tensor (Angs^2/ns)')
print(analysis.diffusion_coeff_tensor('s1'))

print('diffusion length square tensor (Angs)')
print(analysis.diffusion_length_square_tensor('s1'))

analysis.plot_excitations('s1').show()
analysis.plot_distances('s1').show()
analysis.plot_histogram('s1', normalized=True, bins=20).show()
