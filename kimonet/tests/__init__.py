from kimonet.system.generators import regular_system
from kimonet.analysis import Trajectory, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay
import kimonet.core.processes as processes

import unittest
import numpy as np
np.random.seed(0)  # set random seed in order for the examples to reproduce the exact references


processes.transfer_scheme = {Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='forster'): forster_coupling,
                             # Transfer(initial=('s1', 'gs'), final=('gs', 's2'), description='test'): forster_coupling,
                             # Transfer(initial=('s2', 'gs'), final=('gs', 's1'), description='test2'): forster_coupling,
                             # Transfer(initial=('s2', 'gs'), final=('gs', 's2'), description='test3'): forster_coupling
                             }

decay_scheme = {Decay(initial='s1', final='gs', description='singlet_radiative_decay'): einstein_singlet_decay,
                # Decay(initial='s1', final='s2', description='singlet_radiative_decay'): singlet_decay,
                # Decay(initial='s2', final='gs', description='singlet_radiative_decay'): singlet_decay,
                }


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

        self.molecule = Molecule(state_energies={'gs': 0, 's1': 1},
                                 reorganization_energies={'gs': 0, 's1': 0.2},
                                 transition_moment=[2.0, 0],  # transition dipole moment of the molecule (Debye)
                                 decays=decay_scheme
                                 )

        conditions = {'temperature': 273.15,            # temperature of the system (K)
                      'refractive_index': 1,            # refractive index of the material (adimensional)
                      'cutoff_radius': 3.1}             # maximum interaction distance (Angstroms)

        self.system = regular_system(conditions=conditions,
                                     molecule=self.molecule,
                                     lattice={'size': [3, 3], 'parameters': self.parameters},  # Angstroms
                                     orientation=[0, 0, 0])

    def test_kmc_algorithm(self):
        num_trajectories = 50                           # number of trajectories that will be simulated
        max_steps = 100000                              # maximum number of steps for trajectory allowed

        trajectories = []
        for j in range(num_trajectories):

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

        test = {'diffusion coefficient': np.around(analysis.diffusion_coefficient('s1'), decimals=6),
                'lifetime': np.around(analysis.lifetime('s1'), decimals=6),
                'diffusion length': np.around(analysis.diffusion_length('s1'), decimals=6),
                'diffusion tensor': np.around(analysis.diffusion_coeff_tensor('s1'), decimals=6).tolist(),
                'diffusion length tensor': np.around(analysis.diffusion_length_tensor('s1'), decimals=6).tolist()
                }

        print(test)
        ref = {'diffusion coefficient': 2.713221,
               'lifetime': 1601.843545,
               'diffusion length': 129.200077,
               'diffusion tensor': [[5.169843, 0.318825],
                                    [0.318825, 0.256599]],
               'diffusion length tensor': [[126.724347, 32.767362],
                                           [32.767362, 25.171412]]
               }


        # This is just for visual comparison (not accounted in the test)
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

        self.assertDictEqual(ref, test)
