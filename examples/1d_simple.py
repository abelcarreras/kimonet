from kimonet.core.processes.couplings import intermolecular_vector
from kimonet.core.processes import DecayRate, DirectRate
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc
import numpy as np
from kimonet.core.processes.types import Transition
from kimonet.system.state import ground_state as gs


# custom transfer functions
def transfer_rate(donor, acceptor, conditions, supercell, cell_increment):
    distance = np.linalg.norm(intermolecular_vector(donor, acceptor, supercell, cell_increment))
    constant = conditions['custom_constant']

    return constant/distance**2


# custom decay functions
def decay_rate(initial_state, final_state, molecule):
    rates = {'TypeA': 1/100,
             'TypeB': 1/50,
             'TypeC': 1/25}
    return rates[molecule.name]


# states list
# gs = State(label='gs', energy=0.0, multiplicity=1)
s1 = State(label='s1', energy=1.0, multiplicity=1)
s2 = State(label='s2', energy=1.5, multiplicity=1)

# setup molecules
molecule = Molecule(decays=[DecayRate(initial_states=s1, final_states=gs,  # TO SYSTEM
                                      decay_rate_function=decay_rate,
                                      description='custom decay rate'),
                            DecayRate(initial_states=s2, final_states=gs,  # TO SYSTEM
                                      decay_rate_function=decay_rate,
                                      description='custom decay rate')
                            ],
                    )

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

print('**NO_list_ini: ', [mol for mol in system.molecules])

# set additional system parameters
system.transfer_scheme = [DirectRate(initial_states=(s1, gs), final_states=(gs, s1),
                                     rate_constant_function=transfer_rate,
                                     description='custom'),
                          DirectRate(initial_states=(s2, gs), final_states=(gs, s2),
                                     rate_constant_function=transfer_rate,
                                     description='custom'),
                          ]
system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

# some system analyze functions
system_test_info(system)
visualize_system(system)

# do the kinetic Monte Carlo simulation
parallel_run = False
if parallel_run:
    from kimonet import calculate_kmc_parallel
    trajectories = calculate_kmc_parallel(system,
                                          processors=10,
                                          num_trajectories=1000,    # number of trajectories that will be simulated
                                          max_steps=100000,         # maximum number of steps for trajectory allowed
                                          silent=False)
else:
    trajectories = calculate_kmc(system,
                                 num_trajectories=100,    # number of trajectories that will be simulated
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
