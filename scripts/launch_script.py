from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay, Direct
import kimonet.core.processes as processes
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list

import numpy as np
np.random.seed(1)  # for testing


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
                  's1': 1,
                  's2': 1,
                  's2t': 1}

# reorganization energies of the states (eV)
reorganization_energies = {('s1', 'gs'): 0.2,
                           ('gs', 's1'): 0.2,
                           ('s2', 'gs'): 0.2,
                           ('gs', 's2'): 0.2,
                           }

molecule = Molecule(state_energies=state_energies,
                    vibrations=MarcusModel(reorganization_energies),
                    transition_moment={('s1', 'gs'): [2.0, 0], ('s2', 'gs'): [2.0, 0]},  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme,
                    vdw_radius=1.7
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 3.1,             # maximum interaction distance (Angstroms)
              'dexter_k': 1.0}                  # eV

#######################################################################################################################

num_trajectories = 5                          # number of trajectories that will be simulated
max_steps = 100                              # maximum number of steps for trajectory allowed

system_1 = regular_system(conditions=conditions,
                          molecule=molecule,
                          lattice={'size': [4, 4], 'parameters': [3.0, 3.0]},  # Angstroms
                          orientation=[0, 0, 0])  # (Rx, Ry, Rz) if None then random orientation


system_2 = crystal_system(conditions=conditions,
                          molecule=molecule,
                          scaled_coordinates=[[0, 0],],
                          unitcell=[[3.0, 0.5],
                                    [0.5, 2.0]],
                          dimensions=[4, 4],
                          orientations=[[0, 0, np.pi],  # if element is None then random, if list then oriented
                                        None])


system = system_2  # choose 2

visualize_system(system)


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
    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_tensor(state))


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
    print(analysis.diffusion_length_tensor(state))

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

