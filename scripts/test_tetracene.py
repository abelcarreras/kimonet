from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, intermolecular_vector, unit_vector
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay, Direct
import kimonet.core.processes as processes
import concurrent.futures as futures


import numpy as np
# np.random.seed(1)  # for testing


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
                             Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='singlet transport'): forster_coupling, # lambda x, y, z, k: 0.04048,
                             Transfer(initial=('t1', 'gs'), final=('gs', 't1'), description='triplet transport'): dexter_coupling # lambda x, y, z, k: 0.00586,
                             # Direct(initial=('s1', 'gs'), final=('tp', 'tp'), description='singlet fission'): lambda x, y, z, k: 8.3,
                             # Direct(initial=('tp', 'tp'), final=('s1', 'gs'), description='triplet fusion'): lambda x, y, z, k: 1.0,
                             #Direct(initial=('tp', 'tp'), final=('t1', 't1'), description='triplet dissociation'): lambda x, y, z, k: 2.0,
                             # Direct(initial=('tp', 'tp'), final=('s1', 'gs'), description='triplet annihilation'): lambda x, y, z, k: 0.45,
                             }

def decay_singlet(x):
    return 1/0.15

def decay_triplet(x):
    return 1/0.35 # (1.37e3)

decay_scheme2 = {Decay(initial='s1', final='gs', description='decay s1'): decay_singlet,
                 Decay(initial='t1', final='gs', description='decay s2'): decay_triplet
}

decay_scheme = {Decay(initial='s1', final='gs', description='decay s1'): einstein_singlet_decay,
                Decay(initial='t1', final='gs', description='decay s2'): decay_triplet
}


# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 2.9705,
                  't1': 2.3772,
                  'tp': 2.3772}

# reorganization energies of the states (eV)
reorganization_energies = {'gs': 0,
                           's1': 0.12,  # 0.12,
                           't1': 0.23,
                           'tp': 0.2}

molecule = Molecule(state_energies=state_energies,
                    reorganization_energies=reorganization_energies,
                    transition_moment={('s1', 'gs'): [0.0, -4.278, 0], ('t1', 'gs'): [0.0, 0.0, 0.0]},  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme2,
                    vdw_radius=1.7
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 10.0,             # maximum interaction distance (Angstroms)
              'dexter_k': 1.0}                  # eV

#######################################################################################################################

num_trajectories = 1500                          # number of trajectories that will be simulated
max_steps = 100000                              # maximum number of steps for trajectory allowed

system = crystal_system(conditions=conditions,
                        molecule=molecule,
                        scaled_coordinates=[[0.0, 0.0, 0.0],
                                            [0.5, 0.5, 0.0]
                                            ],
                        unitcell=[[ 7.3347, 0.0000, -3.1436],
                                  [-0.2242, 6.0167, -1.203],
                                  [ 0.0000, 0.0000,  130.57]],
                        dimensions=[3, 3, 3],
                        orientations=[[ (90-61)*np.pi/180, np.pi/2, 0],
                                      [(90-115)*np.pi/180, np.pi/2, 0]])

#orientations=[[0, 68*np.pi/180, np.pi/2],  # if element is None then random, if list then oriented
#                                      [0, 115*np.pi/180, np.pi/2]])



# visualize_system(system, dipole='t1')


def run_trajectory(system, index):

    system = system.copy()
    system.add_excitation_random('t1', 1)

    trajectory = Trajectory(system)
    for i in range(max_steps):

        change_step, step_time = do_simulation_step(system)

        if system.is_finished:
            break

        trajectory.add_step(change_step, step_time)

        if i == max_steps-1:
            print('Maximum number of steps reached!!')

    print('trajectory {} done!'.format(index))
    return trajectory


# executor = futures.ThreadPoolExecutor(max_workers=4)
executor = futures.ProcessPoolExecutor(max_workers=12)

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
    print(analysis.diffusion_length_tensor(state))


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
analysis.plot_distances('t1')
plt.figure()
analysis.plot_histogram('s1')
plt.figure()
analysis.plot_histogram('t1')

plt.show()
