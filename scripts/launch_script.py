from kimonet.system.generators import ordered_system, disordered_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import update_system
import numpy as np
np.random.seed(0)  # for testing


"""
Generic molecule initialization
Possible states: 
    'gs': ground state 
    's1': first singlet state 
All energies must be given in eV. By default initialized at gs.
"""

# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 1,
                  's2': 1}

# reorganization energies of the states (eV)
reorganization_energies = {'gs': 0,
                           's1': 0.2,
                           's2': 0.2}

molecule = Molecule(state_energies=state_energies,
                    reorganization_energies=reorganization_energies,
                    transition_moment=[2.0, 0]  # transition dipole moment of the molecule (Debye)
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 3.1}             # maximum interaction distance (Angstroms)

#######################################################################################################################

num_trajectories = 50                          # number of trajectories that will be simulated
max_steps = 100000                              # maximum number of steps for trajectory allowed

system = ordered_system(conditions=conditions,
                        molecule=molecule,
                        lattice={'size': [3, 3], 'parameters': [3.0, 3.0]},  # Angstroms
                        orientation=[0, 0, 0])

# visualize_system(system)

trajectories = []
for j in range(num_trajectories):

    system.add_excitation_center('s1')
    # system.add_excitation_index('s1', 0)
    # system.add_excitation_random('s2', 3)

    # visualize_system(system)

    print('iteration: ', j)
    trajectory = Trajectory(system)
    for i in range(max_steps):

        change_step, step_time = update_system(system)

        if system.is_finished:
            break

        trajectory.add(change_step, step_time)

        # visualize_system(system)

        if i == max_steps-1:
            print('Maximum number of steps reached!!')

    system.reset()

    # print(trajectory.get_lifetime_ratio('s1'), trajectory.get_lifetime_ratio('s2'), trajectory.get_lifetime_ratio('s3'))
    # print(trajectory.get_lifetime_ratio('s3'))

    trajectories.append(trajectory)


# diffusion properties
analysis = TrajectoryAnalysis(trajectories)

print(analysis)

print('diffusion coefficient (average): {} angs^2/ns'.format(analysis.diffusion_coefficient()))
print('diffusion coefficient (s1): {} angs^2/ns'.format(analysis.diffusion_coefficient('s1')))

print('lifetime: {} ns'.format(analysis.lifetime()))
print('diffusion length: {} angs'.format(analysis.diffusion_length()))

print('diffusion tensor')
print(analysis.diffusion_coeff_tensor('s1'))
print('diffusion length tensor')
print(analysis.diffusion_length_tensor('s1'))
# print(np.sqrt(analysis.diffusion_coeff_tensor()*analysis.lifetime()*2))

plt = analysis.plot_2d()
plt.figure()
analysis.plot_distances()

plt.show()
