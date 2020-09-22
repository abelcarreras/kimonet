from kimonet.core.processes.couplings import intermolecular_vector
from kimonet.core.processes import DecayRate, DirectRate
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate
from kimonet.system.vibrations import MarcusModel
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.types import Transition

import unittest
import numpy as np

# states list
gs = State(label=gs.label, energy=0.0, multiplicity=1)
s1 = State(label='s1', energy=1.0, multiplicity=1)


# custom transfer function
def transfer_rate(initial, final, conditions, supercell, cell_increment):

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)

    distance = np.linalg.norm(intermolecular_vector(initial[0].get_center(),
                                                    initial[1].get_center(),
                                                    supercell,
                                                    cell_increment))

    constant = conditions['custom_constant']

    return constant / distance ** 2


# custom decay function
def decay_rate(initial, final):
    rates = {'TypeA': 1 / 100,
             'TypeB': 1 / 50,
             'TypeC': 1 / 25}
    return rates[initial[0].get_center().name]


class Test1DFast(unittest.TestCase):

    def setUp(self):

        # custom decay functions
        def decay_rate(initial, final):
            rates = {'TypeA': 1 / 100,
                     'TypeB': 1 / 50,
                     'TypeC': 1 / 25}
            return rates[initial[0].get_center().name]

        # setup molecules
        molecule = Molecule(decays=[DecayRate(initial_states=s1, final_states=gs,
                                              decay_rate_function=decay_rate,
                                              description='custom decay rate')],
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
        self.system = System(molecules=[molecule1, molecule2, molecule3],
                             conditions={'custom_constant': 1},
                             supercell=[[3]])

        # set initial exciton
        self.system.add_excitation_index(s1, 1)

    def test_kmc_algorithm(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

        # custom decay functions

        # set additional system parameters
        self.system.transfer_scheme = [DirectRate(initial_states=(s1, gs), final_states=(gs, s1),
                                                  rate_constant_function=transfer_rate,
                                                  description='custom')]

        self.system.decay_scheme = [DecayRate(initial_states=s1, final_states=gs,
                                              decay_rate_function=decay_rate,
                                              description='custom decay rate')]

        self.system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

        # some system analyze functions
        system_test_info(self.system)

        trajectories = calculate_kmc(self.system,
                                     num_trajectories=10,  # number of trajectories that will be simulated
                                     max_steps=100000,  # maximum number of steps for trajectory allowed
                                     silent=True)

        # Results analysis
        analysis = TrajectoryAnalysis(trajectories)

        print('diffusion coefficient: {:9.5f} Angs^2/ns'.format(analysis.diffusion_coefficient('s1')))
        print('lifetime:              {:9.5f} ns'.format(analysis.lifetime('s1')))
        print('diffusion length:      {:9.5f} Angs'.format(analysis.diffusion_length('s1')))
        print('diffusion tensor (Angs^2/ns)')
        print(analysis.diffusion_coeff_tensor('s1'))

        print('diffusion length square tensor (Angs)')
        print(analysis.diffusion_length_square_tensor('s1'))

        test = {'diffusion coefficient': np.around(analysis.diffusion_coefficient('s1'), decimals=4),
                'lifetime': np.around(analysis.lifetime('s1'), decimals=4),
                'diffusion length': np.around(analysis.diffusion_length('s1'), decimals=4),
                'diffusion tensor': np.around(analysis.diffusion_coeff_tensor('s1'), decimals=4).tolist(),
                'diffusion length tensor': np.around(np.sqrt(analysis.diffusion_length_square_tensor('s1')), decimals=6).tolist()
                }

        ref = {'diffusion coefficient': 6.6671,
               'lifetime': 35.9001,
               'diffusion length': 18.3685,
               'diffusion tensor': [[6.6671]],
               'diffusion length tensor': [[18.368451]]
               }

        self.assertDictEqual(ref, test)

    def test_kmc_algorithm_2(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

        # set additional system parameters
        self.system.transfer_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                                  electronic_coupling_function=forster_coupling,
                                                  description='Forster coupling',
                                                  arguments={'ref_index': 1,
                                                             'transition_moment': {Transition(s1, gs): [0.01]}},
                                                  vibrations=MarcusModel(reorganization_energies={
                                                      Transition(gs, s1, symmetric=False): 0.07,
                                                      Transition(s1, gs, symmetric=False): 0.07})
                                                  )
                                       ]

        self.system.decay_scheme = [DecayRate(initial_states=s1, final_states=gs,
                                              decay_rate_function=decay_rate,
                                              description='custom decay rate')]

        self.system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

        # some system analyze functions
        system_test_info(self.system)

        trajectories = calculate_kmc(self.system,
                                     num_trajectories=10,  # number of trajectories that will be simulated
                                     max_steps=10000,  # maximum number of steps for trajectory allowed
                                     silent=True)

        # Results analysis
        analysis = TrajectoryAnalysis(trajectories)

        print('diffusion coefficient: {:9.5f} Angs^2/ns'.format(analysis.diffusion_coefficient('s1')))
        print('lifetime:              {:9.5f} ns'.format(analysis.lifetime('s1')))
        print('diffusion length:      {:9.5f} Angs'.format(analysis.diffusion_length('s1')))
        print('diffusion tensor (Angs^2/ns)')
        print(analysis.diffusion_coeff_tensor('s1'))

        print('diffusion length square tensor (Angs)')
        print(analysis.diffusion_length_square_tensor('s1'))

        test = {'diffusion coefficient': np.around(analysis.diffusion_coefficient('s1'), decimals=4),
                'lifetime': np.around(analysis.lifetime('s1'), decimals=4),
                'diffusion length': np.around(analysis.diffusion_length('s1'), decimals=4),
                'diffusion tensor': np.around(analysis.diffusion_coeff_tensor('s1'), decimals=4).tolist(),
                'diffusion length tensor': np.around(np.sqrt(analysis.diffusion_length_square_tensor('s1')), decimals=6).tolist()
                }

        print(test)
        ref = {'diffusion coefficient': 0.7894,
               'lifetime': 40.8351,
               'diffusion length': 8.0561,
               'diffusion tensor': [[0.7894]],
               'diffusion length tensor': [[8.056054]]
               }

        self.assertDictEqual(ref, test)
