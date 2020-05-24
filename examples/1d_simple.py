from kimonet.core.processes.couplings import intermolecular_vector
from kimonet.core.processes import DecayRate, DirectRate
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet import system_test_info, calculate_kmc
import numpy as np


# custom transfer functions
def transfer_rate(donor, acceptor, conditions, supercell, cell_increment):
    distance = np.linalg.norm(intermolecular_vector(donor, acceptor, supercell, cell_increment))
    constant = conditions['custom_constant']

    return constant/distance**2


# custom decay functions
def decay_rate(molecule):
    rates = {'TypeA': 1/100,
             'TypeB': 1/50,
             'TypeC': 1/25}
    return rates[molecule.name]


# setup molecules
molecule = Molecule(state_energies={'gs': 0, 's1': 1.0},  # eV
                    transition_moment={('s1', 'gs'): [1.0]},  # Debye
                    decays={DecayRate(initial='s1', final='gs', description='custom decay rate'): decay_rate},
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
system.add_excitation_index('s1', 1)
#system.add_excitation_index('s1', 2)

# set additional system parameters
system.transfer_scheme = {DirectRate(initial=('s1', 'gs'), final=('gs', 's1'), description='custom'): transfer_rate}
system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

# some system analyze functions
system_test_info(system)
visualize_system(system)

# do the kinetic Monte Carlo simulation
trajectories = calculate_kmc(system,
                             num_trajectories=1000,    # number of trajectories that will be simulated
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
