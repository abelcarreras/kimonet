from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc
from kimonet.core.processes.couplings import forster_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate
from kimonet.system.vibrations import MarcusModel
from kimonet.system.generators import regular_system, crystal_system
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition

import unittest
import numpy as np


# states list
# gs = State(label='gs', energy=0.0, multiplicity=1)
s1 = State(label='s1', energy=1.0, multiplicity=1)


class Test1DFast(unittest.TestCase):

    def setUp(self):
        # list of decay functions by state
        molecule = Molecule()

        # define system as a crystal
        self.system = crystal_system(molecules=[molecule],  # molecule to use as reference
                                     scaled_site_coordinates=[[0.0, 0.0]],
                                     unitcell=[[5.0, 1.0],
                                               [1.0, 5.0]],
                                     dimensions=[2, 2],  # supercell size
                                     orientations=[[0.0, 0.0, np.pi / 2]])  # if element is None then random, if list then Rx Ry Rz

        # set initial exciton
        self.system.add_excitation_index(s1, 1)
        self.system.add_excitation_index(s1, 0)

        # set additional system parameters
        self.system.cutoff_radius = 8  # interaction cutoff radius in Angstrom

    def test_kmc_algorithm(self):
        np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references

        marcus = MarcusModel(reorganization_energies={(s1, gs): 0.08,  # eV
                                                      (gs, s1): 0.08},
                             temperature=300)  # Kelvin

        # list of transfer functions by state
        self.system.process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                                 electronic_coupling_function=forster_coupling_extended,
                                                 description='Forster',
                                                 arguments={'ref_index': 2, 'longitude': 2, 'n_divisions': 30,
                                                             'transition_moment': {Transition(s1, gs): [0.3, 0.1]}}, # Debye
                                                 vibrations=marcus
                                                 ),
                                      DecayRate(initial_states=s1, final_states=gs,
                                                decay_rate_function=einstein_radiative_decay,
                                                arguments={'transition_moment': {Transition(s1, gs): [0.3, 0.1]}}, # Debye
                                                description='singlet_radiative_decay')
                                      ]

        self.system.cutoff_radius = 10.0  # interaction cutoff radius in Angstrom

        # some system analyze functions
        system_test_info(self.system)

        trajectories = calculate_kmc(self.system,
                                     num_trajectories=10,  # number of trajectories that will be simulated
                                     max_steps=100,  # maximum number of steps for trajectory allowed
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
                'diffusion length tensor': np.around(np.sqrt(analysis.diffusion_length_square_tensor('s1', unit_cell=[[0.0, 0.5],
                                                                                                                      [0.2, 0.0]])), decimals=4).tolist()
                }

        ref = {'diffusion coefficient': 3.0024,
               'lifetime': 60.7097,
               'diffusion length': 31.0016,
               'diffusion tensor': [[3.3135, 0.3028],
                                    [0.3028, 2.6914]],
               'diffusion length tensor': [[23.2884, 9.9925],
                                           [9.9925, 20.4634]]
               }

        self.assertDictEqual(ref, test)
