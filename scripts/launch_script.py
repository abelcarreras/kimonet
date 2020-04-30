from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step, system_test_info
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay, Direct
import kimonet.core.processes as processes
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis.diffusion.diffusion_plots import plot_polar_plot

import numpy as np
#np.random.seed(1)  # for testing
DEBYE_TO_AU = 0.393430

processes.transfer_scheme = {
                             Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='Forster'): forster_coupling,
                             #Transfer(initial=('s2', 'gs'), final=('gs', 's2'), description='Dexter'): forster_coupling,
                             #Transfer(initial=('s1', 'gs'), final=('s2t', 's2t'), description='transition'): lambda x, y, z, k: 1/100,
                             #Direct(initial=('s2t', 's2t'), final=('s2', 's2'), description='split'): lambda x, y, z, k: 1 / 10,
                             #Direct(initial=('s1', 's2'), final=('s2', 's1'), description='cross'): lambda x, y, z, k: 1 / 10,
                             #Direct(initial=('s2', 's2'), final=('gs', 's1'), description='merge'): lambda x, y, z, k: 1/10
                             }

decay_scheme = {
                Decay(initial='s1', final='gs', description='singlet_radiative_decay'): einstein_singlet_decay,
                #Decay(initial='s1', final='gs', description='decay s1'): lambda x: 1/50,
                #Decay(initial='s2', final='gs', description='decay s2'): lambda x: 1/30
}

# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 4.37,
                  's2': 4.92,
                  's2t': 1}

# reorganization energies of the states (eV)
reorganization_energies = {('s1', 'gs'): 0.08,
                           ('gs', 's1'): 0.08,
                           ('s2', 'gs'): 0.2,
                           ('gs', 's2'): 0.2,
                           }

from scipy.interpolate import interp1d
from scipy.integrate import simps

data_abs = np.loadtxt('naphthalene_abs.txt').T
data_em = np.loadtxt('naphthalene_em.txt').T

n_abs = simps(y=data_abs[1], x=data_abs[0])
n_em = simps(y=data_em[1], x=data_em[0])

f_abs = interp1d(data_abs[0], data_abs[1]/n_abs, fill_value=0, bounds_error=False)
f_em = interp1d(data_em[0], data_em[1]/n_em, fill_value=0, bounds_error=False)

molecule = Molecule(state_energies=state_energies,
                    # vibrations=MarcusModel(reorganization_energies),
                    vibrations=EmpiricalModel({('gs', 's1'): f_abs,
                                               ('s1', 'gs'): f_em}),
                    transition_moment={
                        # ('s1', 'gs'): [0.07259229/DEBYE_TO_AU, 0.01227128/DEBYE_TO_AU, 1.76580646/DEBYE_TO_AU],
                        #('s1', 'gs'): [0.07259229 / DEBYE_TO_AU, 1.76580646 / DEBYE_TO_AU, 0.01227128 / DEBYE_TO_AU],
                        ('s1', 'gs'): [2.0, 0.0, 0.0],
                        ('s2', 'gs'): [0., 1., 0.]},  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme,
                    vdw_radius=1.7
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 300.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 7,             # maximum interaction distance (Angstroms)
              'dexter_k': 1.0}                  # eV

#######################################################################################################################

num_trajectories = 50                          # number of trajectories that will be simulated
max_steps = 10000                              # maximum number of steps for trajectory allowed

system_1 = regular_system(conditions=conditions,
                          molecule=molecule,
                          lattice={'size': [4, 4, 4], 'parameters': [3.0, 3.0, 3.0]},  # Angstroms
                          orientation=[0.8492, 1.0803, 0.4389])  # (Rx, Ry, Rz) if None then random orientation


system_2 = crystal_system(conditions=conditions,
                          molecule=molecule,
                          scaled_coordinates=[[0, 0, 0],
                                              [0.5, 0.5, 0.0]
                                              ],
                          unitcell=[[6.32367864, 0.0000000, -4.35427391],
                                    [0.00000000, 5.7210000,  0.00000000],
                                    [0.00000000, 0.0000000,  8.39500000]],
                          dimensions=[4, 4, 4],
                          orientations=[[0.1569345, 0.1761809, -0.42690],  # if element is None then random, if list then oriented
                                        [-0.156866, 0.1761865,  0.42686],
                                        None])

system = system_2  # choose 2

system.add_excitation_center('s1')

# Donor: 558 / Acceptor: 447
# Donor: 558 / Acceptor: 540
from kimonet.core.processes.couplings import forster_coupling
#print(system.molecules[558].state)
#ec = forster_coupling(system.molecules[558], system.molecules[540], conditions, system.supercell)
#print(ec)


#print('-----')
# Donor: 558 / Acceptor: 1448 <- normal
#ec = forster_coupling(system.molecules[558], system.molecules[1448], conditions, system.supercell)
#print('n', ec)



#from kimonet.utils.units import HBAR_PLANCK
#print(HBAR_PLANCK)
#print(2*np.pi/HBAR_PLANCK * ec**2 * 0.627905689626639)

system_test_info(system)

visualize_system(system, dipole='s1')


trajectories = []
for j in range(num_trajectories):

    system.add_excitation_center('s1')
    #system.add_excitation_index('s1', 1)
    #system.add_excitation_random('s2', 5)

    # visualize_system(system)

    print('iteration: ', j)
    trajectory = Trajectory(system)

    for i in range(max_steps):

        change_step, step_time = do_simulation_step(system)

        if system.is_finished:
            break

        trajectory.add_step(change_step, step_time)

        # visualize_system(system)

        if i == max_steps-1:
            print('Maximum number of steps reached!!')

    system.reset()

    trajectories.append(trajectory)

    # trajectory.plot_graph()
    # plt = trajectory.plot_2d()
    # plt.show()


# diffusion properties
analysis = TrajectoryAnalysis(trajectories)


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))
    print(analysis.diffusion_coeff_tensor(state, unit_cell=system.supercell))

    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_square_tensor(state))
    print(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell))

    plot_polar_plot(analysis.diffusion_coeff_tensor(state, unit_cell=system.supercell),
                    title='Diffusion', crystal_labels=True, plane=[0, 2])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell),
                    title='Length square', crystal_labels=True, plane=[0, 1])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell),
                    title='Length square', crystal_labels=True, plane=[0, 2])

store_trajectory_list(trajectories, 'test.h5')

trajectory_list = load_trajectory_list('test.h5')


print('--------**************----------')
analysis = TrajectoryAnalysis(trajectory_list)


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))
    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_square_tensor(state))

exit()


#plt.figure(1)
#analysis.plot_excitations('s1')
#plt.show()
#analysis.plot_excitations('s2')
#analysis.plot_excitations()

analysis.plot_2d('s1').show()
#plt.figure()
#plt = analysis.plot_2d('s2')
analysis.plot_distances('s1').show()
#plt.figure()
#analysis.plot_distances('s2')
#analysis.plot_histogram('s1').savefig('test.png')
analysis.plot_histogram('s1').show()

#plt.figure()
#analysis.plot_histogram('s2')

