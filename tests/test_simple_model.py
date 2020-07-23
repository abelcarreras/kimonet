from kimonet.system.generators import regular_system
from kimonet.analysis import Trajectory, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet.system.state import State
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate
from kimonet.system.vibrations import MarcusModel

import unittest
import numpy as np
np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references


transfer_scheme = [GoldenRule(initial=('s1', 'gs'), final=('gs', 's1'),
                              electronic_coupling_function=forster_coupling,
                              description='forster couplings'),
                  ]

decay_scheme = [DecayRate(initial='s1', final='gs',
                          decay_rate_function=einstein_radiative_decay,
                          description='singlet_radiative_decay'),
                ]


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

        self.parameters = [3.0, 3.0]

        vib = MarcusModel(reorganization_energies={('s1', 'gs'): 0.5,
                                                   ('gs', 's1'): 0.5})

        self.molecule = Molecule(states=[State(label='gs', energy=0.0),
                                         State(label='s1', energy=3.0)],
                                 transition_moment={('s1', 'gs'): [1.0, 0]},  # transition dipole moment of the molecule (Debye)
                                 vibrations=vib,
                                 decays=decay_scheme,
                                 )

        conditions = {'temperature': 273.15,            # temperature of the system (K)
                      'refractive_index': 1,            # refractive index of the material (adimensional)
                      'cutoff_radius': 3.1}             # maximum interaction distance (Angstroms)

        self.system = regular_system(conditions=conditions,
                                     molecule=self.molecule,
                                     lattice={'size': [3, 3], 'parameters': self.parameters},  # Angstroms
                                     orientation=[0, 0, 0])

        self.system.cutoff_radius = 3.1
        self.system.transfer_scheme = transfer_scheme

    def test_kmc_algorithm(self):
        num_trajectories = 10                           # number of trajectories that will be simulated
        max_steps = 100000                              # maximum number of steps for trajectory allowed

        trajectories = []
        for j in range(num_trajectories):

            # print('traj', j)
            self.system.add_excitation_center('s1')

            trajectory = Trajectory(self.system)
            for i in range(max_steps):

                change_step, step_time = do_simulation_step(self.system)

                if self.system.is_finished:
                    break

                trajectory.add_step(change_step, step_time)

            self.system.reset()

            trajectories.append(trajectory)

        analysis = TrajectoryAnalysis(trajectories)
        print(analysis)

        print('n_dim: ', analysis.n_dim)

        test = {'diffusion coefficient': np.around(analysis.diffusion_coefficient('s1'), decimals=4),
                'lifetime': np.around(analysis.lifetime('s1'), decimals=4),
                'diffusion length': np.around(analysis.diffusion_length('s1'), decimals=4),
                'diffusion tensor': np.around(analysis.diffusion_coeff_tensor('s1'), decimals=4).tolist(),
                'diffusion length tensor': np.around(np.sqrt(analysis.diffusion_length_square_tensor('s1')), decimals=6).tolist()
                }

        ref = {'diffusion coefficient': 31.5846,
               'lifetime': 259.8353,
               'diffusion length': 203.5827,
               'diffusion tensor': [[59.3142, 11.4197],
                                    [11.4197, 3.8549]],
               'diffusion length tensor': [[200.448497, 76.072334],
                                           [76.072334, 35.585109]]
               }

        # This is just for visual comparison (not accounted in the test)
        try:
            from kimonet.core.processes import get_transfer_rates, get_decay_rates

            self.system.add_excitation_index('s1', 0)
            transfer_x, _, transfer_y, _ = get_transfer_rates(0, self.system)[1]
            decay, = get_decay_rates(0, self.system)[1]

            print('analytical model')
            print('----------------')
            data = get_analytical_model(self.parameters[0], analysis.n_dim, transfer_x, decay)
            print('x:', data)
            data = get_analytical_model(self.parameters[1], analysis.n_dim, transfer_y, decay)
            print('y:', data)
        except ValueError:
            pass

        self.assertDictEqual(ref, test)
