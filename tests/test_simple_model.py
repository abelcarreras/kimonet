from kimonet.system.generators import regular_system
from kimonet.analysis import Trajectory, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc
from kimonet.core.processes.couplings import forster_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, SimpleRate
from kimonet.system.vibrations import MarcusModel
from kimonet.system.generators import regular_system, crystal_system
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition

import unittest
import numpy as np


np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

s1 = State(label='s1', energy=1.0, multiplicity=1)


def get_analytical_model(distance, dimension, transfer, decay):

    k_list = [transfer] * 2 * dimension
    t_rad = 1. / decay
    diff_m = np.sum(k_list) * distance ** 2 / (2 * dimension)
    ld_m = np.sqrt(2 * dimension * diff_m * t_rad)

    return {'diffusion coefficient': diff_m,
            'diffusion length': ld_m,
            'lifetime': t_rad}


class TestKimonet(unittest.TestCase):

    def setUp(self):

        # list of decay functions by state
        molecule = Molecule()

        # define system as a crystal
        self.system = crystal_system(molecules=[molecule],  # molecule to use as reference
                                     scaled_site_coordinates=[[0.0, 0.0]],
                                     unitcell=[[5.0, 0.0],
                                               [0.0, 5.0]],
                                     dimensions=[2, 2],  # supercell size
                                     orientations=[[0.0, 0.0, 0.0]])  # if element is None then random, if list then Rx Ry Rz

        # set initial exciton
        self.system.add_excitation_index(s1, 0)

        # set additional system parameters
        self.system.cutoff_radius = 5.5  # interaction cutoff radius in Angstrom


    def test_kmc_algorithm(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

        transitions = [Transition(s1, gs, symmetric=True)]


        # list of transfer functions by state
        self.system.process_scheme = [SimpleRate(initial_states=(s1, gs), final_states=(gs, s1),
                                                 rate_constant=0.1),
                                      SimpleRate(initial_states=(s1,), final_states=(gs,),
                                                 rate_constant=0.01)]

        # some system analyze functions
        system_test_info(self.system)

        trajectories = calculate_kmc(self.system,
                                     num_trajectories=500,  # number of trajectories that will be simulated
                                     max_steps=1000,  # maximum number of steps for trajectory allowed
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
                'diffusion tensor': np.around(analysis.diffusion_coeff_tensor('s1', unit_cell=[[0.0, 0.5],
                                                                                               [0.2, 0.0]]), decimals=4).tolist(),
                'diffusion length tensor': np.around(analysis.diffusion_length_square_tensor('s1', unit_cell=[[0.0, 0.5],
                                                                                                              [0.2, 0.0]]), decimals=6).tolist()
                }
        print(test)

        ref = {'diffusion coefficient': 2.4439,
               'lifetime': 92.7496,
               'diffusion length': 30.1679,
               'diffusion tensor': [[2.1338, 0.1308],
                                    [0.1308, 2.7539]],
               'diffusion length tensor': [[798.7, 25.7],
                                           [25.7, 1021.5]]
               }


        # This is just for visual comparison (not accounted in the test)
        decay = 0.01
        transfer = 0.1
        distance = 5

        print('analytical model')
        print('----------------')
        data = get_analytical_model(distance, analysis.n_dim, transfer, decay)
        print('results analytical:', data)

        self.assertDictEqual(ref, test)
