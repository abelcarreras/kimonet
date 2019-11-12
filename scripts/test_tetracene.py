from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay, Direct
import kimonet.core.processes as processes

import numpy as np
np.random.seed(1)  # for testing


processes.transfer_scheme = {
                             Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='singlet transport'): forster_coupling,
                             Transfer(initial=('t1', 'gs'), final=('gs', 't1'), description='triplet transport'): dexter_coupling,
                             Direct(initial=('s1', 'gs'), final=('tp', 'tp'), description='singlet fission'): lambda x, y, z, k: 8.3,
                             Direct(initial=('tp', 'tp'), final=('s1', 'gs'), description='triplet fusion'): lambda x, y, z, k: 1.0,
                             Direct(initial=('tp', 'tp'), final=('t1', 't1'), description='triplet dissociation'): lambda x, y, z, k: 1 / 10,
                             Direct(initial=('t1', 't1'), final=('s1', 'gs'), description='triplet annihilation'): lambda x, y, z, k: 1/10
                             }

decay_scheme = {Decay(initial='s1', final='gs', description='decay s1'): lambda x: 8.0e-2,
                Decay(initial='t1', final='gs', description='decay s2'): lambda x: 1.6e-5
}

# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 1,
                  't1': 2,
                  'tp': 2}

# reorganization energies of the states (eV)
reorganization_energies = {'gs': 0,
                           's1': 0.2,
                           't1': 0.2,
                           'tp': 0.2}

molecule = Molecule(state_energies=state_energies,
                    reorganization_energies=reorganization_energies,
                    transition_moment=[2.0, 0],  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme,
                    vdw_radius=1.7
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 8.0,             # maximum interaction distance (Angstroms)
              'dexter_k': 1.0}                  # eV

#######################################################################################################################

num_trajectories = 50                          # number of trajectories that will be simulated
max_steps = 10000                              # maximum number of steps for trajectory allowed

system = crystal_system(conditions=conditions,
                        molecule=molecule,
                        scaled_coordinates=[[0.0, 0.0],
                                            [0.5, 0.5]],
                        unitcell=[[7.3347, 0.0],
                                  [-0.2242, 6.0167]],
                        dimensions=[2, 2],
                        orientations=[[0, 0, 68*np.pi/180],  # if element is None then random, if list then oriented
                                      [0, 0, 115*np.pi/180]])



visualize_system(system)
print(system.get_volume())
exit()

trajectories = []
for j in range(num_trajectories):

    system.add_excitation_random('s1', 1)

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

    #trajectory.plot_graph()
    #plt = trajectory.plot_2d()
    #plt.show()


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


plt = analysis.plot_excitations('s1')
analysis.plot_excitations('s2')
analysis.plot_excitations()
plt.figure()

plt = analysis.plot_2d('s1')
plt.figure()
plt = analysis.plot_2d('s2')
plt.figure()
analysis.plot_distances('s1')
plt.figure()
analysis.plot_distances('s2')
plt.figure()
analysis.plot_histogram('s1')
plt.figure()
analysis.plot_histogram('s2')

plt.show()
