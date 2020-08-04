from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, intermolecular_vector, unit_vector
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
import kimonet.core.processes as processes
import concurrent.futures as futures
import sys

import numpy as np
np.random.seed(1)  # for testing


def triplet_coupling(donor, acceptor, conditions, supercell):
    r_vector = intermolecular_vector(donor, acceptor)  # position vector between donor and acceptor
    # print('direction', np.dot(unit_vector(r_vector), [1, 0, 0]), np.dot(unit_vector(r_vector), [0, 1, 0]))
    if np.dot(unit_vector(r_vector), [1, 0, 0])**2 > 0.7:
        # print('a')
        return 0.0
    elif np.dot(unit_vector(r_vector), [0, 1, 0])**2 > 0.7:
        # print('b')
        return 1.2e-3
    else:
        return 7.2e-3


def singlet_coupling(donor, acceptor, conditions, supercell):
    r_vector = intermolecular_vector(donor, acceptor)  # position vector between donor and acceptor
    # print('direction', np.dot(unit_vector(r_vector), [1, 0, 0]), np.dot(unit_vector(r_vector), [0, 1, 0]))
    if np.dot(unit_vector(r_vector), [1, 0, 0])**2 > 0.7:
        # print('a')
        return 12.61e-3
    elif np.dot(unit_vector(r_vector), [0, 1, 0])**2 > 0.7:
        # print('b')
        return 41.85e-3
    else:
        return 27.51e-3


processes.transfer_scheme = {
                             GoldenRule(initial_states=('s1', 'gs'), final_states=('gs', 's1'), description='singlet transport'): forster_coupling,  # lambda x, y, z, k: 0.04048,
                             # Transfer(initial=('t1', 'gs'), final=('gs', 't1'), description='triplet transport'): dexter_coupling  # lambda x, y, z, k: 0.00586,
                             # Direct(initial=('s1', 'gs'), final=('tp', 'tp'), description='singlet fission'): lambda x, y, z, k: 8.3,
                             # Direct(initial=('tp', 'tp'), final=('s1', 'gs'), description='triplet fusion'): lambda x, y, z, k: 1.0,
                             # Direct(initial=('tp', 'tp'), final=('t1', 't1'), description='triplet dissociation'): lambda x, y, z, k: 2.0,
                             # Direct(initial=('tp', 'tp'), final=('s1', 'gs'), description='triplet annihilation'): lambda x, y, z, k: 0.45,
                             }

decay_scheme = {DecayRate(initial_states='s1', final_states='gs', description='decay s1'): einstein_radiative_decay}


# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 2.9705}

# reorganization energies of the states (eV)
reorganization_energies = {('s1', 'gs'): 0.06,
                           ('gs', 's1'): 0.06}


molecule = Molecule(state_energies=state_energies,
                    transition_moment={('s1', 'gs'): [0.0, -4.278]},  # transition dipole moment of the molecule (Debye)
                    vibrations=MarcusModel(reorganization_energies),
                    decays=decay_scheme,
                    vdw_radius=1.7
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 300.0,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 10.0,             # maximum interaction distance (Angstroms)
              'dexter_k': 1.0}                  # eV

#######################################################################################################################

num_trajectories = 10                          # number of trajectories that will be simulated
max_steps = 100000                              # maximum number of steps for trajectory allowed

system = crystal_system(conditions=conditions,
                        molecule=molecule,
                        scaled_site_coordinates=[[0.0, 0.0],
                                                 [0.5, 0.5]
                                                 ],
                        unitcell=[[ 7.3347, 0.0000],
                                  [-0.2242, 6.0167]],
                        dimensions=[3, 3],
                        orientations=[[ (90-61)*np.pi/180, np.pi/2, 0],
                                      [(90-115)*np.pi/180, np.pi/2, 0]])

# orientations=[[0, 68*np.pi/180, np.pi/2],  # if element is None then random, if list then oriented
#                                      [0, 115*np.pi/180, np.pi/2]])

# system.add_excitation_random('s1', 1)

# molecule.state = 's1'
# print(molecule.decay_rates())

#visualize_system(system, dipole='s1')

import os

def run_trajectory(system, index):

    print('Starting {} {}'.format(index, os.getpid()))

    system = system.copy()
    # np.random.seed(None)  # guarantees a different trajectory by process
    system.add_excitation_random('s1', 1)

    trajectory = Trajectory(system)

    for i in range(max_steps):

        change_step, step_time = do_simulation_step(system)

        if system.is_finished:
            break

        trajectory.add_step(change_step, step_time)

        if i == max_steps-1:
            print('Maximum number of steps reached!!')
            # print(trajectory.get_times()[-10:])
    # trajectory.fh5.close()
    # return trajectory

    print(trajectory.get_times()[-10:])
    # trajectory.plot_graph()
    print('num nodes:', trajectory.get_number_of_nodes())
    print('node_coordinates:')
    print(list(trajectory.get_graph().nodes[0]['coordinates'][:10]))
    print('cell_state:')
    print(list(trajectory.get_graph().nodes[0]['cell_state'][:10]))
    print('time:')
    print(list(trajectory.get_graph().nodes[0]['time'][:10]))
    print('index:')
    print(list(trajectory.get_graph().nodes[0]['index'][:10]))

    print('-------------------------------------------------')
    trajectory.get_number_of_nodes()
    print('trajectory {} done!'.format(index))

    return trajectory


if False:
    trajectories = []
    for i in range(3):
        np.random.seed(i)
        trajectories.append(run_trajectory(system, i))
else:
    # executor = futures.ThreadPoolExecutor(max_workers=4)
    executor = futures.ProcessPoolExecutor(max_workers=10)

    futures_list = []
    for i in range(num_trajectories):
        futures_list.append(executor.submit(run_trajectory, system, i))

    trajectories = []
    for f in futures.as_completed(futures_list):
        trajectories.append(f.result())


analysis = TrajectoryAnalysis(trajectories)


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))
    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_square_tensor(state))

    eval, evec = np.linalg.eig(analysis.diffusion_coeff_tensor(state))
    print('eval', eval)
    print('evec', evec)

exit()
plt = analysis.plot_excitations('s1')
analysis.plot_excitations('t1')
analysis.plot_excitations()
plt.figure()

# plt = analysis.plot_2d('t1')
# plt.figure()
# plt = analysis.plot_2d('t1')
# plt.figure()
analysis.plot_distances('s1')
plt.figure()
analysis.plot_histogram('s1')


plt.show()
