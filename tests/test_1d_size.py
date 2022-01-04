from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition

import unittest
import numpy as np

# states list
gs = gs.copy()
s1 = State(label='s1', energy=1.0, multiplicity=1, size=1)
tt = State(label='tt', energy=1.0, multiplicity=1, size=2, connected_distance=1.0)


# custom transfer function
def transfer_rate(initial, final, custom_constant=1):

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()
    distance = np.linalg.norm(r_vector)

    return custom_constant / distance ** 2


# custom decay function
def decay_rate(initial, final):
    rates = {'TypeA': 1. / 100,
             'TypeB': 1. / 50,
             'TypeC': 1. / 25}
    return rates[initial[0].get_center().name]


class Test1DFast(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

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
        self.system = System(molecules=[molecule1, molecule2, molecule3],
                             supercell=[[3]])

        # set initial exciton
        # self.system.add_excitation_index(tt, 1)
        # self.system.add_excitation_random(s1, 2)
        self.system.add_excitation_index(s1, 0)
        self.system.add_excitation_index(s1, 1)

    def test_kmc_algorithm(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

        # set additional system parameters
        self.system.process_scheme = [DirectRate(initial_states=(s1, gs), final_states=(gs, s1),
                                                 rate_constant_function=transfer_rate,
                                                 arguments={'custom_constant': 1},
                                                 description='transport s1'),
                                      DirectRate(initial_states=(s1, s1), final_states=(tt,),
                                                 rate_constant_function=transfer_rate,
                                                 arguments={'custom_constant': 1},
                                                 description='merge s1-s1 -> tt'),
                                      DirectRate(initial_states=(s1,), final_states=(gs,),
                                                 rate_constant_function=decay_rate,
                                                 description='decay s1'),
                                      DirectRate(initial_states=(tt, gs), final_states=(s1, s1, s1),
                                                 rate_constant_function=decay_rate,
                                                 description='split tt -> s1-s1-s1')
                                      ]

        self.system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

        p = DirectRate(initial_states=(s1, s1), final_states=(tt,),
                       rate_constant_function=transfer_rate,
                       arguments={'custom_constant': 1},
                       description='custom2')
        #print(p.get_transition_connections())
        #exit()

        # some system analyze functions
        system_test_info(self.system)

        trajectories = calculate_kmc(self.system,
                                     num_trajectories=10,  # number of trajectories that will be simulated
                                     max_steps=50,  # maximum number of steps for trajectory allowed
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
                'diffusion length tensor': np.around(np.sqrt(analysis.diffusion_length_square_tensor('s1')), decimals=4).tolist()
                }

        print(test)
        ref = {'diffusion coefficient': 0.0767,
               'lifetime': 14.191,
               'diffusion length': 0.68590,
               'diffusion tensor': [[0.0767]],
               'diffusion length tensor': [[0.68590]]
               }

        self.assertDictEqual(ref, test)

